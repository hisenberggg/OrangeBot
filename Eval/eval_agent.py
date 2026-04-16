"""
OrangeBot Agent Evaluation Script
==================================
Compares new LangSmith traces against golden (reference) traces.

Evaluates on 3 dimensions:
  1. Route Accuracy    — Did the agent pick the correct route? (deterministic)
  2. Response Quality  — Is the new response as good as the reference? (LLM-as-judge)
  3. Completeness      — Does the new response cover all key facts? (LLM-as-judge)

Usage:
  python eval_agent.py --golden golden_dataset.json --new_traces ./new_traces/

  Judge defaults match the OrangeBot app (planner / wiki): gpt-4o-mini + OPENAI_API_KEY from .env.
  Or with Anthropic judge:
  python eval_agent.py --golden golden_dataset.json --new_traces ./new_traces/ --model claude --anthropic-key YOUR_KEY
"""

import json
import os
import glob
import argparse
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional

try:
    from config import settings as _app_settings
except ImportError:
    _app_settings = None

# ─── Configuration ───────────────────────────────────────────────────────────

# Same chat model as agent/planner.py and agent/wiki_agent.py; override with --model
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"


# ─── Data Classes ────────────────────────────────────────────────────────────

@dataclass
class GoldenExample:
    id: str
    input: str
    expected_route: str
    reference_response: str
    extra: dict = field(default_factory=dict)


@dataclass
class NewTrace:
    id: str
    input: str
    actual_route: str
    actual_response: str
    route_rationale: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    test_id: str
    input: str
    expected_route: str
    actual_route: str
    route_match: bool
    response_quality_score: int  # 1-5
    response_quality_reasoning: str
    completeness_score: int      # 1-5
    completeness_reasoning: str
    overall_score: float         # weighted average


# ─── Loaders ─────────────────────────────────────────────────────────────────

def load_golden_dataset(path: str) -> list[GoldenExample]:
    """Load the golden dataset from JSON."""
    with open(path, "r") as f:
        data = json.load(f)
    return [GoldenExample(**item) for item in data]


def load_new_traces(folder: str) -> list[NewTrace]:
    """Load new LangSmith trace JSON files from a folder."""
    traces = []
    for filepath in glob.glob(os.path.join(folder, "*.json")):
        with open(filepath, "r") as f:
            data = json.load(f)

        # Extract fields from LangSmith trace format
        input_msg = data["inputs"]["messages"][0]["content"]
        outputs = data["outputs"]

        trace = NewTrace(
            id=os.path.basename(filepath).replace(".json", ""),
            input=input_msg,
            actual_route=outputs.get("route", "unknown"),
            actual_response=outputs.get("final_response", ""),
            route_rationale=outputs.get("route_rationale", ""),
            extra={
                k: v for k, v in outputs.items()
                if k not in ("route", "final_response", "route_rationale", "messages")
            },
        )
        traces.append(trace)
    return traces


# ─── Matching ────────────────────────────────────────────────────────────────

def match_traces(
    golden: list[GoldenExample], new: list[NewTrace]
) -> list[tuple[GoldenExample, Optional[NewTrace]]]:
    """Match golden examples to new traces by input text (fuzzy)."""
    pairs = []
    for g in golden:
        best_match = None
        g_normalized = g.input.strip().lower()
        for n in new:
            if n.input.strip().lower() == g_normalized:
                best_match = n
                break
        pairs.append((g, best_match))
    return pairs


# ─── LLM Judge ───────────────────────────────────────────────────────────────

JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator for a university FAQ chatbot called OrangeBot (Syracuse University).

You will be given:
- A student's QUESTION
- A REFERENCE response (known good answer from a previous version)
- A NEW response (from the current version being evaluated)

Evaluate the NEW response on two dimensions:

## 1. Response Quality (1-5)
How good is the new response compared to the reference?
- 5: Equal or better than reference — accurate, well-structured, helpful
- 4: Slightly worse but still good — minor issues in tone or structure
- 3: Acceptable but noticeably worse — missing some nuance or clarity
- 2: Poor — significant issues in accuracy or helpfulness
- 1: Bad — wrong information, unhelpful, or off-topic

## 2. Completeness (1-5)
Does the new response cover all the key facts from the reference?
- 5: All key facts covered, possibly with additional useful info
- 4: Most key facts covered, 1 minor detail missing
- 3: Some key facts missing but core answer is present
- 2: Major facts missing — answer is incomplete
- 1: Completely misses the point

Respond ONLY in this exact JSON format (no markdown, no backticks):
{{"response_quality_score": <int>, "response_quality_reasoning": "<1-2 sentences>", "completeness_score": <int>, "completeness_reasoning": "<1-2 sentences>"}}

---
QUESTION: {question}

REFERENCE RESPONSE:
{reference}

NEW RESPONSE:
{new_response}
"""


def judge_with_openai(question: str, reference: str, new_response: str, model: str, api_key: str) -> dict:
    """Use OpenAI as the LLM judge."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question, reference=reference, new_response=new_response
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if present
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def judge_with_anthropic(question: str, reference: str, new_response: str, api_key: str) -> dict:
    """Use Anthropic Claude as the LLM judge."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question, reference=reference, new_response=new_response
    )

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def judge(question: str, reference: str, new_response: str, args) -> dict:
    """Route to the correct LLM judge based on args."""
    if args.model.startswith("claude"):
        api_key = args.anthropic_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY or use --anthropic-key")
        return judge_with_anthropic(question, reference, new_response, api_key)
    else:
        api_key = (
            args.openai_key
            or (_app_settings.openai_api_key if _app_settings else "")
            or os.getenv("OPENAI_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY in .env (see config/settings.py) or use --openai-key"
            )
        return judge_with_openai(question, reference, new_response, args.model, api_key)


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_pair(
    golden: GoldenExample, new: NewTrace, args
) -> EvalResult:
    """Evaluate a single golden/new trace pair."""

    # 1. Route accuracy (deterministic)
    route_match = golden.expected_route.lower() == new.actual_route.lower()

    # 2 & 3. LLM-as-judge for quality and completeness
    scores = judge(golden.input, golden.reference_response, new.actual_response, args)

    # Weighted overall: route=30%, quality=40%, completeness=30%
    route_score = 5 if route_match else 1
    overall = (
        0.30 * route_score
        + 0.40 * scores["response_quality_score"]
        + 0.30 * scores["completeness_score"]
    )

    return EvalResult(
        test_id=golden.id,
        input=golden.input,
        expected_route=golden.expected_route,
        actual_route=new.actual_route,
        route_match=route_match,
        response_quality_score=scores["response_quality_score"],
        response_quality_reasoning=scores["response_quality_reasoning"],
        completeness_score=scores["completeness_score"],
        completeness_reasoning=scores["completeness_reasoning"],
        overall_score=round(overall, 2),
    )


# ─── Reporting ───────────────────────────────────────────────────────────────

def print_report(results: list[EvalResult]):
    """Print a formatted evaluation report to console."""
    print("\n" + "=" * 80)
    print("  ORANGEBOT AGENT EVALUATION REPORT")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    total = len(results)
    route_matches = sum(1 for r in results if r.route_match)
    avg_quality = sum(r.response_quality_score for r in results) / total
    avg_completeness = sum(r.completeness_score for r in results) / total
    avg_overall = sum(r.overall_score for r in results) / total

    # Summary
    print(f"\n  SUMMARY ({total} test cases)")
    print(f"  {'─' * 50}")
    print(f"  Route Accuracy:       {route_matches}/{total} ({100*route_matches/total:.0f}%)")
    print(f"  Avg Response Quality: {avg_quality:.2f}/5")
    print(f"  Avg Completeness:     {avg_completeness:.2f}/5")
    print(f"  Avg Overall Score:    {avg_overall:.2f}/5")
    print()

    # Per-case details
    for i, r in enumerate(results, 1):
        route_icon = "✅" if r.route_match else "❌"
        print(f"  [{i}] {r.input[:60]}...")
        print(f"      Route: {r.expected_route} → {r.actual_route}  {route_icon}")
        print(f"      Quality: {r.response_quality_score}/5  |  Completeness: {r.completeness_score}/5  |  Overall: {r.overall_score}/5")
        print(f"      Quality reason:      {r.response_quality_reasoning}")
        print(f"      Completeness reason: {r.completeness_reasoning}")
        print()

    print("=" * 80)


def save_report(results: list[EvalResult], output_path: str):
    """Save evaluation results to JSON."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "num_cases": len(results),
        "summary": {
            "route_accuracy": sum(1 for r in results if r.route_match) / len(results),
            "avg_response_quality": sum(r.response_quality_score for r in results) / len(results),
            "avg_completeness": sum(r.completeness_score for r in results) / len(results),
            "avg_overall": sum(r.overall_score for r in results) / len(results),
        },
        "results": [asdict(r) for r in results],
    }
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to: {output_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OrangeBot Agent Evaluation")
    parser.add_argument("--golden", required=True, help="Path to golden_dataset.json")
    parser.add_argument("--new_traces", required=True, help="Folder with new LangSmith trace JSONs")
    parser.add_argument("--model", default=DEFAULT_JUDGE_MODEL, help="Judge model (gpt-4o, claude, etc.)")
    parser.add_argument("--openai-key", default=None, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--anthropic-key", default=None, help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--output", default=None, help="Path to save JSON report (optional)")
    args = parser.parse_args()

    # Load data
    print("\n  Loading golden dataset...")
    golden = load_golden_dataset(args.golden)
    print(f"  Loaded {len(golden)} golden examples")

    print("  Loading new traces...")
    new_traces = load_new_traces(args.new_traces)
    print(f"  Loaded {len(new_traces)} new traces")

    # Match golden → new
    pairs = match_traces(golden, new_traces)
    matched = [(g, n) for g, n in pairs if n is not None]
    unmatched = [(g, n) for g, n in pairs if n is None]

    if unmatched:
        print(f"\n  ⚠️  {len(unmatched)} golden examples had no matching new trace:")
        for g, _ in unmatched:
            print(f"      - {g.input[:70]}")

    if not matched:
        print("\n  ❌ No matching traces found. Make sure you asked the same questions.")
        return

    print(f"\n  Evaluating {len(matched)} matched pairs...\n")

    # Run evaluation
    results = []
    for i, (g, n) in enumerate(matched, 1):
        print(f"  Evaluating [{i}/{len(matched)}]: {g.input[:50]}...")
        try:
            result = evaluate_pair(g, n, args)
            results.append(result)
        except Exception as e:
            print(f"    ⚠️  Error evaluating: {e}")

    # Report
    if results:
        print_report(results)
        output_path = args.output or f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_report(results, output_path)


if __name__ == "__main__":
    main()
