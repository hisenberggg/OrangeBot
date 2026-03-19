"use client";

import { useState, useEffect } from "react";
import type { ThinkingEvent } from "@/lib/types";
import styles from "./ThinkingSteps.module.css";

interface ThinkingStepsProps {
  steps: ThinkingEvent[];
  streaming: boolean;
}

function stepIcon(step: ThinkingEvent): string {
  switch (step.type) {
    case "route":
      return "↗";
    case "tool_call":
      return "🔍";
    case "status":
    default:
      return "⟳";
  }
}

function stepLabel(step: ThinkingEvent): React.ReactNode {
  switch (step.type) {
    case "route":
      return (
        <>
          Routed to <strong>{step.route}</strong>
          {step.rationale && (
            <span className={styles.rationale}> — {step.rationale}</span>
          )}
        </>
      );
    case "tool_call":
      return (
        <>
          Called <strong>{step.tool}</strong>
          {step.input && (
            <span className={styles.rationale}> — &quot;{step.input}&quot;</span>
          )}
        </>
      );
    case "status":
      return step.message;
  }
}

export default function ThinkingSteps({ steps, streaming }: ThinkingStepsProps) {
  const [collapsed, setCollapsed] = useState(false);

  useEffect(() => {
    if (!streaming && steps.length > 0) {
      setCollapsed(true);
    }
  }, [streaming, steps.length]);

  if (steps.length === 0) return null;

  return (
    <div className={styles.container}>
      <button
        className={styles.toggle}
        onClick={() => setCollapsed((c) => !c)}
        aria-expanded={!collapsed}
      >
        <span className={styles.chevron} data-open={!collapsed}>
          ▸
        </span>
        <span className={styles.toggleLabel}>
          Thinking{streaming ? "..." : ""}
        </span>
        <span className={styles.count}>{steps.length} steps</span>
      </button>

      {!collapsed && (
        <ul className={styles.list}>
          {steps.map((step, i) => (
            <li key={i} className={styles.step}>
              <span className={styles.icon}>{stepIcon(step)}</span>
              <span className={styles.text}>{stepLabel(step)}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
