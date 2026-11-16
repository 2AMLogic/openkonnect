"""SES-09 ``show_findings`` and the SES-10 round trip against a fake PCB editor.

Findings come from the installed kicad-tools: ``Finding`` model objects built
directly, or the versioned payload (``schema_version`` + ``findings``) a real
``kct`` run prints for ``tests/fixtures/selection.kicad_pcb``. No finding
dict is copied into this file except the deliberately malformed ones.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
from kicad_tools.findings import SCHEMA_VERSION, Finding
from kicad_tools.validate.violations import DRCViolation as LegacyViolation

from openkonnect.session import file_state
from openkonnect.tools import findings as F
from openkonnect.tools.get_selection import get_selection
from openkonnect.tools.run_kct_handoff import find_kct
from openkonnect.tools.show_findings import (
    _POINT_MESSAGES,
    BBOX_EMPTY_MESSAGE,
    show_findings,
)

from .conftest import (
    SELECTION_BOARD,
    EditorFake,
    make_session,
    selection_board_index,
)


def show(fake, findings, **kw):
    session = make_session(fake)
    return session.run_tool(lambda s: show_findings(s, findings, **kw))


def selected(fake):
    r = make_session(fake).run_tool(get_selection)
    assert r["ok"], r
    return r


def kct_json(*argv: str) -> dict:
    kct = find_kct()
    if kct is None:
        pytest.skip("kct (kicad-tools) not found")
    p = subprocess.run([kct, *argv], capture_output=True, text=True, timeout=120, check=False)
    err_tail = p.stderr.strip()[-500:]
    if not p.stdout.strip():
        raise AssertionError(
            f"`kct {' '.join(argv)}` wrote nothing on stdout (rc={p.returncode}); "
            f"stderr tail: {err_tail!r}"
        )
    try:
        return json.loads(p.stdout)
    except ValueError as e:
        raise AssertionError(
            f"`kct {' '.join(argv)}` stdout is not JSON (rc={p.returncode}): {e}; "
            f"stderr tail: {err_tail!r}"
        ) from None


def courtyard(ref: str, x: float, y: float) -> Finding:
    return Finding(
        rule_id="courtyard_missing",
        severity="warning",
        message=f"{ref} has no courtyard",
        refs=[ref],
        point=[x, y],
        layer="F.Cu",
    )


def envelope(*fs: Finding, **extra) -> dict:
    """The additive shape every finding-emitting kct command spreads into its JSON."""
    return {
        "schema_version": SCHEMA_VERSION,
        "findings": [f.to_dict() for f in fs],
        "violations": [{"deliberately": "not a finding"}],
        **extra,
    }


# A real `kct drc tests/fixtures/selection.kicad_pcb --format json` response,
# captured verbatim (kicad-tools 0.29.0, kicad-cli 8). `kct drc` shells out to
# kicad-cli, which CI runners do not have, so the live-subprocess loop skips
# drc there and this capture is what the production reader is tested against.
# The capture-run's `source` (kicad-cli's temp report path) and footprint-library
# warnings are kept as captured.
DRC_CAPTURE = Path(__file__).parent / "fixtures" / "kct_drc_selection.json"


# -- the reader reads the versioned format -------------------------------------


def test_supported_schema_version_tracks_upstream():
    assert F.SUPPORTED_SCHEMA_VERSION == SCHEMA_VERSION


def test_finding_objects_and_their_json_read_the_same():
    f = courtyard("R2", 120.0, 60.0)
    assert F.is_finding(f) and F.is_finding(f.to_dict())
    assert F.read(f) == F.read(f.to_dict())
    a = show(EditorFake(), [f])
    b = show(EditorFake(), [f.to_dict()])
    assert a["findings"] == b["findings"]


def test_versioned_payloads_from_every_kct_emitter_resolve():
    # drc is not in this loop: `kct drc` shells out to kicad-cli, which a CI
    # runner does not have (there it exits 0 with empty stdout), so its real
    # payload is covered by the captured file in the test below.
    fake = EditorFake()
    for cmd in ("check", "detect-mistakes", "net-status"):
        payload = kct_json(cmd, SELECTION_BOARD, "--format", "json")
        assert payload["schema_version"] == F.SUPPORTED_SCHEMA_VERSION
        assert payload["findings"], f"{cmd} emitted no findings on the fixture board"
        r = show(fake, payload)
        assert r["ok"] is True, {cmd: r}
        assert r["totals"]["findings"] == len(payload["findings"])


def test_captured_drc_payload_resolves_without_kicad():
    payload = json.loads(DRC_CAPTURE.read_text(encoding="utf-8"))
    assert payload["schema_version"] == F.SUPPORTED_SCHEMA_VERSION
    assert payload["findings"], "the captured drc run emitted no findings"
    fake = EditorFake()
    r = show(fake, payload)
    assert r["ok"] is True, r
    assert r["totals"]["findings"] == len(payload["findings"])
    # The unconnected_items finding carries refs, pads, a net AND a bbox; the
    # names resolve it and the bbox only checks -- it must not widen the
    # selection beyond what the names resolved (the post-fix regression pin,
    # measured against a real kct drc response).
    unconnected = [
        f for f in r["findings"] if f["rule_id"] == "unconnected_items" and f["index"] == 3
    ]
    assert len(unconnected) == 1
    [f] = unconnected
    assert f["status"] == "resolved"
    assert f["resolved"]["footprints"] == ["R1", "R3"]
    assert f["resolved"]["pads"] == ["R1.2", "R3.2"]
    assert f["resolved"]["item_count"] == 4
    assert f["unresolved"] == []
    [box] = [p for p in f["points"] if "x0_mm" in p.get("file_mm", {})]
    assert box["used_for"] == "check" and box["within_resolved_items"] is True


def test_measure_clearance_shaped_finding_resolves():
    f = Finding(
        rule_id="measure_clearance.clearance",
        severity="error",
        message="Clearance between R3.1 and R3.2 is 1.0000 mm",
        refs=["R3"],
        pads=["R3.1", "R3.2"],
        point=[129.0, 65.0],
        layer="F.Cu",
    )
    r = show(EditorFake(), [f])
    assert r["ok"] is True, r
    [f] = r["findings"]
    assert f["resolved"]["pads"] == ["R3.1", "R3.2"]
    [p] = f["points"]
    assert p["used_for"] == "check" and p["within_resolved_items"] is True


def test_drc_point_from_versioned_payload_resolves_to_item():
    # kct drc's location was reported frame_not_stated and never used before
    # the versioned format; on main this payload is refused outright. Now the
    # point is read and checked against the resolved items.
    f = Finding(
        rule_id="drc.clearance",
        severity="error",
        message="",
        refs=["R3"],
        pads=["R3.1"],
        nets=["SIG"],
        point=[129.0, 65.0],
    )
    r = show(EditorFake(), [f])
    assert r["ok"] is True, r
    [f] = r["findings"]
    assert f["status"] == "resolved"
    [p] = f["points"]
    assert p["used_for"] == "check" and p["within_resolved_items"] is True


def test_detect_mistakes_point_from_versioned_payload_resolves_to_item():
    # Same story as the drc point: locate it now instead of discarding it.
    f = Finding(
        rule_id="mistake.power_trace_width",
        severity="warning",
        message="",
        nets=["VIN"],
        point=[109.0, 57.0],
    )
    r = show(EditorFake(), [f])
    assert r["ok"] is True, r
    [f] = r["findings"]
    assert f["status"] == "resolved"  # the VIN copper
    [p] = f["points"]
    assert p["used_for"] == "check" and p["within_resolved_items"] is True


def test_net_status_finding_resolves_without_board_outline():
    # The point arrives already in the file frame (upstream adds the board
    # origin back), so a board with no Edge.Cuts outline cannot block it.
    index = selection_board_index()
    fake = EditorFake(dict(index, outline_nm=None))
    f = Finding(
        rule_id="net_status.incomplete",
        severity="info",
        message="",
        refs=["R3"],
        pads=["R3.2"],
        nets=["GND"],
        point=[131.0, 65.0],
    )
    r = show(fake, [f])
    assert r["ok"] is True, r
    [f] = r["findings"]
    assert f["status"] == "resolved"
    assert f["resolved"]["pads"] == ["R3.2"]
    [p] = f["points"]
    assert p["within_resolved_items"] is True
    assert p["file_mm"] == {"x_mm": 131.0, "y_mm": 65.0}


def test_envelope_with_unknown_schema_version_is_invalid_argument():
    env = envelope(courtyard("R1", 110, 60))
    env["schema_version"] = 2
    r = show(EditorFake(), env)
    assert r["error"]["code"] == "invalid_argument"
    assert r["error"]["details"]["observed_schema_version"] == 2
    assert r["error"]["details"]["supported_schema_version"] == 1


def test_finding_with_unknown_schema_version_is_invalid_argument():
    bad = courtyard("R1", 110, 60).to_dict() | {"schema_version": 99}
    r = show(EditorFake(), [bad])
    assert r["error"]["code"] == "invalid_argument"
    assert r["error"]["details"]["index"] == 0
    assert r["error"]["details"]["observed_schema_version"] == 99
    assert r["error"]["details"]["supported_schema_version"] == 1


def test_legacy_violation_shape_is_rejected_not_matched():
    # The reflective matcher for pre-0.25 output is gone: a legacy finding
    # dict is invalid_argument, not silently matched.
    legacy = LegacyViolation(rule_id="x", severity="error", message="", items=("R1",))
    assert "schema_version" not in legacy.to_dict()
    r = show(EditorFake(), [legacy.to_dict()])
    assert r["error"]["code"] == "invalid_argument"
    assert r["error"]["details"]["index"] == 0
    assert r["error"]["details"]["observed_schema_version"] is None


def test_envelope_ignores_non_finding_lists():
    # A real payload carries the command's legacy keys alongside findings;
    # only the versioned array is read.
    env = envelope(courtyard("R1", 110, 60))
    r = show(EditorFake(), env)
    assert r["ok"] is True, r
    assert [f["rule_id"] for f in r["findings"]] == ["courtyard_missing"]


def test_empty_findings_array_is_invalid_argument():
    r = show(EditorFake(), envelope())
    assert r["error"]["code"] == "invalid_argument"
    assert "empty" in r["error"]["message"]


def test_bare_findings_array_and_single_finding_are_accepted():
    r = show(EditorFake(), [courtyard("R1", 110, 60).to_dict()])
    assert r["ok"] is True, r
    assert r["findings"][0]["resolved"]["footprints"] == ["R1"]
    r = show(EditorFake(), courtyard("R2", 120, 60).to_dict())
    assert r["ok"] is True, r
    assert r["findings"][0]["resolved"]["footprints"] == ["R2"]
    r = show(EditorFake(), courtyard("R3", 130, 65))
    assert r["ok"] is True, r
    assert r["findings"][0]["resolved"]["footprints"] == ["R3"]


def test_no_stale_gap_references_remain_in_src():
    src = Path(__file__).resolve().parents[1] / "src"
    offenders = []
    for p in sorted(src.rglob("*.py")):
        text = p.read_text(encoding="utf-8")
        if "8015" in text or re.search(r"\bG1\b", text):
            offenders.append(str(p.relative_to(src)))
    assert offenders == []


# -- resolution -------------------------------------------------------------------


def test_waived_finding_is_still_resolved_and_reported():
    f = courtyard("R1", 110, 60)
    f.waived = True
    r = show(EditorFake(), [f])
    assert r["ok"] is True, r
    [f] = r["findings"]
    assert f["waived"] is True and f["status"] == "resolved"


@pytest.mark.parametrize("how", ["object", "json_null"])
def test_finding_without_point_resolves_by_name_and_says_so(how):
    obj = Finding(
        rule_id="courtyard_missing",
        severity="warning",
        message="R2 has no courtyard",
        refs=["R2"],
        nets=["VIN"],
        layer="F.Cu",
    )
    finding: object = obj
    if how == "json_null":
        finding = obj.to_dict()
        assert finding["point"] is None
    fake = EditorFake()
    r = show(fake, [finding, courtyard("R1", 110, 60)])
    assert r["ok"] is True, r
    f, located = r["findings"]
    assert f["status"] == "resolved"
    assert f["resolved"]["footprints"] == ["R2"]
    assert f["resolved"]["nets"] == ["VIN"]
    assert f["points"] == [] and f["has_point"] is False
    assert "null or missing" in f["point_note"]
    assert f["unresolved"] == []
    assert located["has_point"] is True and "point_note" not in located
    assert "R2" in selected(fake)["footprints"]


def test_finding_missing_locator_fields_is_invalid_argument():
    bad = courtyard("R1", 110, 60).to_dict()
    del bad["point"]
    r = show(EditorFake(), [bad])
    assert r["error"]["code"] == "invalid_argument"
    assert "missing" in r["error"]["message"]


def test_finding_with_null_point_and_no_name_is_reported_unresolved():
    # Degrades per finding: this finding is unresolved with a no_locator
    # reason, and the rest of the payload is still resolved and selected.
    bare = Finding(rule_id="x", severity="error", message="").to_dict()
    fake = EditorFake()
    r = show(fake, [courtyard("R1", 110, 60), bare])
    assert r["ok"] is True, r
    good, bare_r = r["findings"]
    assert good["status"] == "resolved"
    assert bare_r["status"] == "unresolved"
    assert bare_r["unresolved"][0]["reason"] == "no_locator"
    assert "carries no locator" in bare_r["unresolved"][0]["message"]
    assert bare_r["action"] == "nothing_to_select"
    assert r["totals"]["unresolved"] == 1
    assert selected(fake)["footprints"] == ["R1"]


def test_unrecognised_finding_is_invalid_argument_naming_its_index():
    fake = EditorFake()
    r = show(fake, [courtyard("R1", 110, 60).to_dict(), {"severity": "error", "where": "here"}])
    assert r["ok"] is False
    assert r["error"]["code"] == "invalid_argument"
    assert r["error"]["details"]["index"] == 1
    assert "set_selection" not in fake.calls


def test_finding_without_any_locator_is_reported_unresolved():
    empty = Finding(rule_id="x", severity="error", message="nothing to point at")
    r = show(EditorFake(), [courtyard("R1", 110, 60), empty])
    assert r["ok"] is True, r
    good, empty_r = r["findings"]
    assert good["status"] == "resolved"
    assert empty_r["status"] == "unresolved"
    assert empty_r["unresolved"][0]["reason"] == "no_locator"
    assert r["totals"]["unresolved"] == 1


def test_envelope_without_findings_is_invalid_argument():
    r = show(EditorFake(), {"summary": {"errors": 0}, "rows": [{"a": 1}]})
    assert r["error"]["code"] == "invalid_argument"


def test_partial_resolution_applies_what_resolved_and_says_why_the_rest_did_not():
    finding = Finding(
        rule_id="x",
        severity="error",
        message="",
        refs=["R1", "R9"],
        pads=["R3.7"],
        nets=["OLD_NET"],
    )
    second = Finding(
        rule_id="net_status.incomplete",
        severity="info",
        message="",
        nets=["GND"],
        pads=["R1.7", "Q1.1"],
    )
    fake = EditorFake()
    r = show(fake, [finding, second])
    assert r["ok"] is True, r
    f0, f1 = r["findings"]
    assert f0["status"] == "partial"
    assert f0["resolved"]["footprints"] == ["R1"]
    reasons = {u["locator"]["value"]: u["reason"] for u in f0["unresolved"]}
    # refs and REF.PAD names are typed by upstream; only their misses are reported.
    assert reasons == {
        "R9": "ref_not_on_board",
        "R3.7": "pad_not_on_footprint",
        "OLD_NET": "net_not_on_board",
    }
    reasons1 = {u["locator"]["value"]: u["reason"] for u in f1["unresolved"]}
    assert reasons1 == {"R1.7": "pad_not_on_footprint", "Q1.1": "ref_not_on_board"}
    assert r["totals"]["partial"] == 2
    assert selected(fake)["footprints"] == ["R1"]


def test_ref_matching_two_footprints_is_ambiguous_not_guessed():
    index = selection_board_index()
    dup = dict(index["footprints"][0], id="fp-dup")
    fake = EditorFake(dict(index, footprints=[*index["footprints"], dup]))
    r = show(fake, [Finding(rule_id="x", severity="error", message="", refs=["R1"])])
    [f] = r["findings"]
    assert f["unresolved"][0]["reason"] == "ambiguous_reference"
    assert f["status"] == "unresolved"
    assert f["resolved"]["footprints"] == []
    # With a file-frame point as well, the point resolves the finding instead.
    r = show(
        fake,
        [Finding(rule_id="x", severity="error", message="", refs=["R1"], point=[110.0, 60.0])],
    )
    [f] = r["findings"]
    assert f["unresolved"][0]["reason"] == "ambiguous_reference"
    assert f["points"][0]["used_for"] == "resolve"
    assert f["resolved"]["footprints"] == ["R1"]


def test_point_only_finding_selects_the_smallest_item_there():
    hit = Finding(rule_id="x", severity="error", message="", point=[129.0, 65.0])
    fake = EditorFake()
    r = show(fake, [hit])
    [f] = r["findings"]
    assert f["status"] == "resolved"
    # R3's pad 1 sits inside R3's footprint box; the pad is the smaller one.
    assert f["resolved"]["pads"] == ["R3.1"]
    assert f["points"][0]["used_for"] == "resolve"


def test_bbox_without_point_resolves_the_smallest_overlapping_items():
    # A region spanning both R3 pads: the pads are the smallest overlapped boxes.
    f = Finding(
        rule_id="drc.clearance", severity="error", message="", bbox=[128.5, 64.4, 131.5, 65.6]
    )
    r = show(EditorFake(), [f])
    assert r["ok"] is True, r
    [f] = r["findings"]
    assert f["status"] == "resolved"
    assert f["resolved"]["pads"] == ["R3.1", "R3.2"]
    assert f["has_point"] is False
    assert "point_note" in f
    [p] = f["points"]
    assert p["used_for"] == "resolve" and p["resolved_to"] == 2


def test_bbox_is_preferred_over_point_when_no_name_resolves():
    f = Finding(
        rule_id="drc.clearance",
        severity="error",
        message="",
        point=[129.0, 65.0],
        bbox=[128.5, 64.4, 131.5, 65.6],
    )
    r = show(EditorFake(), [f])
    assert r["ok"] is True, r
    [f] = r["findings"]
    assert f["resolved"]["pads"] == ["R3.1", "R3.2"]
    box, point = f["points"]  # the bbox is processed (and reported) first
    assert point["used_for"] == "check" and point["within_resolved_items"] is True
    assert box["used_for"] == "resolve"


def test_bbox_with_resolving_names_is_checked_not_resolved():
    # Real kct drc findings carry names AND a bbox (upstream builds it with
    # _bbox_of(), which needs >= 2 points). The names resolve the finding;
    # the bbox must neither widen the selection nor fail it.
    wide = Finding(
        rule_id="unconnected_items",
        severity="error",
        message="",
        refs=["R3"],
        bbox=[100.0, 50.0, 140.0, 70.0],
    )
    r = show(EditorFake(), [wide])
    assert r["ok"] is True, r
    [f] = r["findings"]
    assert f["status"] == "resolved"
    assert f["resolved"]["footprints"] == ["R3"]
    assert f["resolved"]["item_count"] == 1
    assert f["unresolved"] == []
    [box] = f["points"]
    assert box["used_for"] == "check" and box["within_resolved_items"] is True

    empty = Finding(
        rule_id="unconnected_items",
        severity="error",
        message="",
        refs=["R3"],
        bbox=[101.0, 51.0, 102.0, 52.0],  # over bare board, overlaps nothing
    )
    r = show(EditorFake(), [empty])
    assert r["ok"] is True, r
    [f] = r["findings"]
    assert f["status"] == "resolved"  # the names still resolve it
    assert f["resolved"]["footprints"] == ["R3"]
    assert f["resolved"]["item_count"] == 1
    assert f["unresolved"] == []
    [box] = f["points"]
    assert box["used_for"] == "check" and box["within_resolved_items"] is False


def test_bbox_over_nothing_resolves_nothing_and_leaves_the_selection():
    miss = Finding(rule_id="x", severity="error", message="", bbox=[101.0, 51.0, 102.0, 52.0])
    fake = EditorFake()
    fake.selected = [fake.id_of("R2")]
    r = show(fake, [miss])
    assert r["ok"] is True
    [f] = r["findings"]
    assert f["status"] == "unresolved"
    assert f["unresolved"][0]["reason"] == "empty_area"
    assert f["unresolved"][0]["message"] == BBOX_EMPTY_MESSAGE
    assert f["action"] == "nothing_to_select"
    assert r["selection_changed"] is False
    assert "set_selection" not in fake.calls
    assert selected(fake)["footprints"] == ["R2"]


def test_empty_area_resolves_nothing_and_leaves_the_selection():
    miss = Finding(rule_id="x", severity="error", message="", point=[101.0, 51.0])
    fake = EditorFake()
    fake.selected = [fake.id_of("R2")]
    r = show(fake, [miss])
    assert r["ok"] is True
    [f] = r["findings"]
    assert f["status"] == "unresolved"
    assert f["unresolved"][0]["reason"] == "empty_area"
    assert f["action"] == "nothing_to_select"
    assert r["selection_changed"] is False
    assert "set_selection" not in fake.calls
    assert selected(fake)["footprints"] == ["R2"]


def test_unrouted_net_without_named_pads_selects_its_pads():
    fake = EditorFake()
    r = show(fake, [Finding(rule_id="x", severity="error", message="", nets=["SIG"])])
    [f] = r["findings"]
    assert f["resolved"]["pads"] == ["R2.2", "R3.1"]
    assert "no copper" in f["net_basis"]["SIG"]


# -- mode, zoom, focus, staleness ----------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "zoom", "parts"),
    [
        ("select", False, []),
        ("highlight", False, ["highlight"]),
        ("both", True, ["highlight", "zoom"]),
        ("select", True, ["zoom"]),
    ],
)
def test_highlight_and_zoom_are_unsupported_but_selection_still_happens(mode, zoom, parts):
    fake = EditorFake()
    r = show(fake, [courtyard("R1", 110, 60)], mode=mode, zoom=zoom)
    assert r["ok"] is True
    assert [u["part"] for u in r["unsupported"]] == parts
    for u in r["unsupported"]:
        assert u["error"]["code"] == "unsupported_on_this_kicad"
    assert r["selection"]["footprints"] == ["R1"]


def test_bad_mode_is_invalid_argument():
    r = show(EditorFake(), [courtyard("R1", 110, 60)], mode="flash")
    assert r["error"]["code"] == "invalid_argument"


def test_focus_index_selects_one_finding_and_reports_the_rest():
    fs = [courtyard("R1", 110, 60), courtyard("R2", 120, 60), courtyard("R3", 130, 65)]
    fake = EditorFake()
    r = show(fake, fs, focus=1)
    assert [f["action"] for f in r["findings"]] == ["not_focused", "selected", "not_focused"]
    assert r["selection"]["footprints"] == ["R2"]
    # Every finding is still resolved and counted.
    assert r["totals"]["resolved"] == 3 and r["totals"]["focused"] == 1

    bad = show(fake, fs, focus=3)
    assert bad["error"]["code"] == "invalid_argument"


def test_focus_next_and_previous_walk_the_list():
    fs = [courtyard("R1", 110, 60), courtyard("R2", 120, 60)]
    fake = EditorFake()
    session = make_session(fake)

    def go(focus):
        r = session.run_tool(lambda s: show_findings(s, fs, focus=focus))
        assert r["ok"], r
        return r["focus"]["index"], r["focus"]["wrapped"], r["selection"]["footprints"]

    assert go("next") == (0, False, ["R1"])
    assert go("next") == (1, False, ["R2"])
    assert go("next") == (0, True, ["R1"])
    assert go("previous") == (1, True, ["R2"])
    assert go("previous") == (0, False, ["R1"])


def test_input_hash_mismatch_flags_possibly_stale_but_still_resolves():
    report = {
        "pcb_sha256": "0" * 64,
        "findings": [courtyard("R1", 110, 60).to_dict()],
    }
    fake = EditorFake()
    r = show(fake, report)
    assert r["ok"] is True
    assert r["staleness"]["possibly_stale"] is True
    assert r["findings"][0]["possibly_stale"] is True
    assert r["findings"][0]["resolved"]["footprints"] == ["R1"]
    assert r["totals"]["possibly_stale"] == 1


def test_input_hash_of_the_saved_file_is_not_stale():
    report = {
        "pcb_sha256": file_state(SELECTION_BOARD)["sha256"],
        "findings": [courtyard("R1", 110, 60).to_dict()],
    }
    r = show(EditorFake(), report)
    assert r["staleness"]["possibly_stale"] is False
    assert r["staleness"]["matches"] == "saved_file"


def test_no_board_open_is_no_document():
    r = show(EditorFake(open_boards=[[]]), [courtyard("R1", 110, 60)])
    assert r["error"]["code"] == "no_document"


def test_point_messages_no_longer_carry_frame_reasons() -> None:
    # Points arrive in a known frame (file), so the old frame_not_stated and
    # no_board_outline reasons are unreachable; #41's "board-level" wording
    # went with them.
    assert set(_POINT_MESSAGES) == {"empty_area"}


# -- SES-10: selection -> real kct -> show_findings -> same selection ------------------


def _refs(sel):
    return set(sel["footprints"]), set(sel["pads"]), set(sel["nets"])


def _assert_round_trip(before, after):
    diff = {
        name: {"lost": sorted(b - a), "gained": sorted(a - b)}
        for name, b, a in zip(("footprints", "pads", "nets"), before, after, strict=True)
        if a != b
    }
    assert not diff, f"round trip changed the selection: {diff}"


def test_round_trip_footprints_through_kct_check():
    fake = EditorFake()
    fake.selected = [fake.id_of("R1"), fake.id_of("R2")]
    sel = selected(fake)
    wanted = set(sel["footprints"])

    report = kct_json("check", SELECTION_BOARD, "--format", "json")
    assert report["pcb_sha256"] == file_state(SELECTION_BOARD)["sha256"]
    # "check these parts": keep the findings whose refs are all selected refs,
    # with their violation entries (a real payload carries both arrays; only
    # findings is read).
    keep = [i for i, f in enumerate(report["findings"]) if f["refs"] and set(f["refs"]) <= wanted]
    assert keep, "kct check reported nothing on the selected footprints"
    report["findings"] = [report["findings"][i] for i in keep]
    report["violations"] = [report["violations"][i] for i in keep]

    r = show(fake, report)
    assert r["ok"] is True, r
    assert r["staleness"]["possibly_stale"] is False
    _assert_round_trip(_refs(sel), _refs(selected(fake)))

    # A finding's point lands on the footprint it concerns (1 um), in the
    # file frame; a finding without a point is reported, not invented.
    geo = make_session(fake).run_tool(lambda s: get_selection(s, include_geometry=True))
    for f, d in zip(r["findings"], report["findings"], strict=True):
        assert f["resolved"]["footprints"] == d["refs"]
        if d["point"] is None:
            assert f["has_point"] is False and f["points"] == []
            assert "point_note" in f
            continue
        assert f["has_point"] is True
        [ref] = d["refs"]
        pos = geo["geometry"][ref]["position_mm"]
        assert abs(pos["x_mm"] - d["point"][0]) <= 1e-3
        assert abs(pos["y_mm"] - d["point"][1]) <= 1e-3


def test_round_trip_pad_and_net_through_kct_net_status():
    fake = EditorFake()
    # The finding names the footprint (refs), the stranded pad (pads) and the
    # net (nets), so all three resolve and must be selected beforehand.
    fake.selected = [fake.id_of("R3"), fake.id_of("R3.2")]
    sel = selected(fake)
    assert sel["nets"] == ["GND"]

    report = kct_json("net-status", SELECTION_BOARD, "--format", "json", "--net", sel["nets"][0])
    assert report["schema_version"] == F.SUPPORTED_SCHEMA_VERSION
    r = show(fake, report)
    assert r["ok"] is True, r
    after = selected(fake)
    _assert_round_trip(_refs(sel), _refs(after))

    # The finding's point is the unconnected pad's position, already in the
    # file frame (1 um).
    [f] = r["findings"]
    [p] = f["points"]
    assert p["within_resolved_items"] is True
    geo = make_session(fake).run_tool(lambda s: get_selection(s, include_geometry=True))
    pos = geo["geometry"]["R3.2"]["position_mm"]
    assert abs(p["file_mm"]["x_mm"] - pos["x_mm"]) <= 1e-3
    assert abs(p["file_mm"]["y_mm"] - pos["y_mm"]) <= 1e-3


def test_round_trip_pads_through_kct_check_connectivity():
    fake = EditorFake()
    fake.selected = [fake.id_of("R3"), fake.id_of("R3.1"), fake.id_of("R3.2")]
    sel = selected(fake)

    report = kct_json("check", SELECTION_BOARD, "--format", "json")
    # The connectivity findings for R3's stranded pads; typed refs, pads and
    # nets resolve all of what each finding concerns.
    conn = [f for f in report["findings"] if f["rule_id"] == "connectivity"]
    assert {tuple(f["pads"]) for f in conn} == {("R3.1",), ("R3.2",)}
    r = show(fake, {"pcb_sha256": report["pcb_sha256"], "findings": conn})
    assert r["ok"] is True, r
    _assert_round_trip(_refs(sel), _refs(selected(fake)))
