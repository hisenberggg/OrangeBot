"""Regression tests for Confluence wiki public URL normalization."""
import os

os.environ.setdefault(
    "CONFLUENCE_BASE_URL",
    "https://answers.atlassian.syr.edu/wiki/rest/api/content",
)

from mcp_servers.wiki.url_utils import (  # noqa: E402
    canonical_wiki_public_base,
    normalize_confluence_webui_url,
)

API_BASE = "https://answers.atlassian.syr.edu/wiki/rest/api/content"
EXPECTED_PREFIX = "https://answers.atlassian.syr.edu/wiki/spaces/Maxwell/pages/160794437/Registering+for+Classes"
REL_PATH = "/spaces/Maxwell/pages/160794437/Registering+for+Classes"


def test_canonical_wiki_public_base_from_rest_url():
    assert canonical_wiki_public_base(API_BASE) == "https://answers.atlassian.syr.edu/wiki"


def test_relative_spaces_path_gets_wiki_prefix():
    assert normalize_confluence_webui_url(REL_PATH, api_base=API_BASE) == EXPECTED_PREFIX


def test_dict_href_relative():
    assert (
        normalize_confluence_webui_url({"href": REL_PATH}, api_base=API_BASE)
        == EXPECTED_PREFIX
    )


def test_already_canonical_unchanged():
    url = EXPECTED_PREFIX
    assert normalize_confluence_webui_url(url, api_base=API_BASE) == url


def test_wrong_host_answers_syr_edu():
    bad = "https://answers.syr.edu/spaces/Maxwell/pages/160794437/Registering+for+Classes"
    assert normalize_confluence_webui_url(bad, api_base=API_BASE) == EXPECTED_PREFIX


def test_atlassian_host_missing_wiki_segment():
    bad = "https://answers.atlassian.syr.edu/spaces/Maxwell/pages/160794437/Registering+for+Classes"
    assert normalize_confluence_webui_url(bad, api_base=API_BASE) == EXPECTED_PREFIX


def test_relative_path_starting_with_wiki():
    path = "/wiki/spaces/Maxwell/pages/160794437/Registering+for+Classes"
    assert (
        normalize_confluence_webui_url(path, api_base=API_BASE)
        == "https://answers.atlassian.syr.edu" + path
    )


def test_empty_and_none():
    assert normalize_confluence_webui_url("", api_base=API_BASE) == ""
    assert normalize_confluence_webui_url(None, api_base=API_BASE) == ""
