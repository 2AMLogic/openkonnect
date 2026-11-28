"""Docs vs code: spec C-3, the README tool table and the kipy confinement
must match the code (#14, #105)."""

from __future__ import annotations

import re
from pathlib import Path

import anyio
import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from openkonnect.errors import ErrorCode
from openkonnect.server import build_server

from .conftest import make_session

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "spec" / "capabilities.md"
README = ROOT / "README.md"

RESERVED_MARKER = "Reserved, not yet emitted:"


def _codes_in_table(text: str) -> set[str]:
    """Every backticked code in the first column of the markdown table rows."""
    codes: set[str] = set()
    for line in text.splitlines():
        if not line.startswith("| `"):
            continue
        first_cell = line.split("|")[1]
        codes.update(re.findall(r"`([a-z_]+)`", first_cell))
    return codes


def _c3_tables() -> tuple[set[str], set[str]]:
    text = SPEC.read_text(encoding="utf-8")
    start = text.index("**C-3 Errors.**")
    end = text.index("**C-4 ", start)
    section = text[start:end]
    split = section.index(RESERVED_MARKER)
    return _codes_in_table(section[:split]), _codes_in_table(section[split:])


def test_every_error_code_is_in_the_c3_emitted_table():
    emitted, _ = _c3_tables()
    values = {c.value for c in ErrorCode}
    assert values - emitted == set(), "ErrorCode values missing from spec C-3"
    assert emitted - values == set(), "C-3 lists as emitted a code that ErrorCode lacks"


def test_c3_reserved_codes_are_not_emitted():
    _, reserved = _c3_tables()
    assert reserved, "C-3 reserved table is empty or was not found"
    assert reserved & {c.value for c in ErrorCode} == set()


TOOL_ROW = re.compile(r"^\|\s*`([a-z_]+)`\s*\|\s*([A-Z]{3}-\d{2})\s*\|", re.M)
SPEC_HEADING = re.compile(r"^### ([A-Z]{3}-\d{2}) (.+)$", re.M)


def _readme_tool_rows(text: str) -> list[tuple[str, str]]:
    """(tool, spec id) for each row of the README tool table; padding in cells is allowed."""
    return TOOL_ROW.findall(text)


def _spec_tool_ids(text: str) -> dict[str, set[str]]:
    """{tool: every spec id whose heading names it} from the spec's entry headings.

    Every backticked name in a heading maps to that heading's ID, so
    `### SES-15 `highlight_net` / `clear_highlight`` yields two tools and
    prose after the name (`SES-06`) is ignored. A name under two headings
    keeps both IDs, so the caller can report the duplicate.
    """
    ids: dict[str, set[str]] = {}
    for spec_id, title in SPEC_HEADING.findall(text):
        for name in re.findall(r"`([a-z_]+)`", title):
            ids.setdefault(name, set()).add(spec_id)
    return ids


def _readme_id_mismatches(rows: list[tuple[str, str]], spec_ids: dict[str, set[str]]) -> list[str]:
    """Problems for README tools only, so prose tokens such as `kct` under two headings pass."""
    problems = []
    for name, readme_id in rows:
        found = spec_ids.get(name, set())
        if not found:
            problems.append(f"`{name}` (README {readme_id}) has no heading in spec/capabilities.md")
        elif len(found) > 1:
            problems.append(
                f"`{name}` appears under several spec headings: {', '.join(sorted(found))}"
            )
        else:
            (spec_id,) = found
            if spec_id != readme_id:
                problems.append(f"`{name}`: README says {readme_id}, spec heading says {spec_id}")
    return problems


def _assert_rows_parsed(rows: list[tuple[str, str]]) -> None:
    assert rows, "README tool table was not parsed: no `| `tool` | ABC-NN |` rows found"


def test_readme_tool_table_matches_registered_tools():
    async def go():
        server = build_server(make_session(sockets=[], running=False))
        async with create_connected_server_and_client_session(server._mcp_server) as client:
            return await client.list_tools()

    registered = {t.name for t in anyio.run(go).tools}
    rows = _readme_tool_rows(README.read_text(encoding="utf-8"))
    _assert_rows_parsed(rows)
    names = [name for name, _ in rows]
    assert len(names) == len(set(names)), "a tool appears twice in the README table"
    assert set(names) == registered


def test_readme_tool_table_spec_ids_match_spec_headings():
    rows = _readme_tool_rows(README.read_text(encoding="utf-8"))
    _assert_rows_parsed(rows)
    problems = _readme_id_mismatches(rows, _spec_tool_ids(SPEC.read_text(encoding="utf-8")))
    assert not problems, "README tool table spec IDs disagree with the spec:\n" + "\n".join(
        problems
    )


def test_spec_tool_ids_handles_multi_tool_and_prose_headings():
    ids = _spec_tool_ids(SPEC.read_text(encoding="utf-8"))
    assert ids["run_kct_handoff"] == {"SES-06"}
    assert ids["highlight_net"] == ids["clear_highlight"] == {"SES-15"}
    assert ids["view_state"] == ids["set_view_state"] == {"SES-16"}
    assert ids["begin_change"] == ids["commit_change"] == ids["abort_change"] == {"SES-17"}


def test_readme_row_parser_tolerates_padded_cells():
    rows = _readme_tool_rows("|  `session_status`  |  SES-01  | Whether KiCad runs |\n")
    assert rows == [("session_status", "SES-01")]
    assert _readme_id_mismatches(rows, {"session_status": {"SES-01"}}) == []


def test_readme_id_mismatch_names_tool_and_both_ids():
    rows = [("run_kct_handoff", "SES-05")]
    (problem,) = _readme_id_mismatches(rows, {"run_kct_handoff": {"SES-06"}})
    assert "run_kct_handoff" in problem and "SES-05" in problem and "SES-06" in problem


def test_tool_under_two_spec_headings_is_reported():
    spec = "### LAY-01 `board_summary`\n\n### LAY-99 `board_summary` draft\n"
    ids = _spec_tool_ids(spec)
    assert ids["board_summary"] == {"LAY-01", "LAY-99"}
    for readme_id in ("LAY-01", "LAY-99"):
        (problem,) = _readme_id_mismatches([("board_summary", readme_id)], ids)
        assert "board_summary" in problem and "LAY-01" in problem and "LAY-99" in problem
        assert "several spec headings" in problem


def test_prose_token_under_two_headings_is_ignored_unless_in_readme():
    ids = _spec_tool_ids("### SES-06 `run_kct_handoff` via `kct`\n### SES-10 `kct` probe\n")
    assert ids["kct"] == {"SES-06", "SES-10"}
    assert _readme_id_mismatches([("run_kct_handoff", "SES-06")], ids) == []


def test_readme_tool_missing_from_spec_is_reported():
    (problem,) = _readme_id_mismatches([("no_such_tool", "SES-99")], {})
    assert "no_such_tool" in problem and "no heading" in problem


def test_unparsable_readme_table_fails_as_a_parse_error():
    rows = _readme_tool_rows("| Tool | Spec |\n|------|------|\n")
    with pytest.raises(AssertionError, match="README tool table was not parsed"):
        _assert_rows_parsed(rows)


KIPY_IMPORT = re.compile(r"^\s*(?:from|import)\s+kipy\b", re.M)
SRC_ROOT = ROOT / "src" / "openkonnect"


def _kipy_importers(src_root: Path = SRC_ROOT) -> set[str]:
    """Paths (relative to ``src/openkonnect/``) of modules importing kipy anywhere.

    The pattern allows leading whitespace on purpose: every kipy import in
    scope is function-local (`session.py`'s error mapping at 8 spaces, the
    backend modules' helpers inside functions), so a `^from kipy` anchor
    would match none of them and wave a violating repo through.
    """
    return {
        str(p.relative_to(src_root))
        for p in sorted(src_root.rglob("*.py"))
        if KIPY_IMPORT.search(p.read_text(encoding="utf-8"))
    }


def _assert_kipy_importers_parsed(importers: set[str]) -> None:
    assert importers, (
        "the kipy-import guard matched nothing under src/openkonnect/: "
        "kipy is gone (retire the guard) or the pattern broke (it must allow "
        "indented, function-local imports)"
    )


def _stray_kipy_importers(importers: set[str]) -> list[str]:
    """Importers outside the documented shape: top-level `backend*` modules plus `session.py`."""
    return sorted(
        m for m in importers if "/" in m or not (m.startswith("backend") or m == "session.py")
    )


def test_kipy_imports_confined_to_backend_modules_and_session():
    importers = _kipy_importers()
    _assert_kipy_importers_parsed(importers)
    stray = _stray_kipy_importers(importers)
    assert not stray, (
        "modules outside the backend layer import kipy: "
        + ", ".join(stray)
        + " — docs/ipc-client.md states kipy is confined to the backend* "
        "modules plus session.py's error mapping; reconcile the doc or move "
        "the imports"
    )


def test_kipy_import_pattern_matches_indented_and_plain_imports():
    assert KIPY_IMPORT.search("from kipy import KiCad\n")
    assert KIPY_IMPORT.search("import kipy\n")
    assert KIPY_IMPORT.search("        from kipy.errors import ApiError\n")
    assert KIPY_IMPORT.search("\timport kipy.proto.common\n")


def test_kipy_import_pattern_ignores_kipy_prefixed_module_names():
    assert not KIPY_IMPORT.search("import kipy_helpers\n")
    assert not KIPY_IMPORT.search("from kipy_helpers import shim\n")


def test_stray_rule_accepts_backend_siblings_and_session():
    importers = {"backend.py", "backend_outline.py", "backend_readers.py", "session.py"}
    assert _stray_kipy_importers(importers) == []


def test_stray_rule_reports_tools_and_top_level_strays():
    stray = _stray_kipy_importers(
        {"backend.py", "session.py", "tools/list_copper.py", "kicad_text.py"}
    )
    assert stray == ["kicad_text.py", "tools/list_copper.py"]


def test_kipy_guard_fails_loud_when_pattern_matches_nothing():
    with pytest.raises(AssertionError, match="matched nothing"):
        _assert_kipy_importers_parsed(set())
