"""LAY-05 live: ``list_copper`` against ``tests/fixtures/selection.kicad_pcb`` open in KiCad.

Never runs in CI. Setup as for ``test_live_selection.py``: KiCad 10 with 'Enable
KiCad API' on, the fixture open and unmodified. Run ``uv run pytest --live
-m live_kicad -rs``. Read-only.

This is the only test that proves the straight-track decode reads KiCad's own
objects correctly: the real ``kipy`` field names and units, in both frames.
``tests/test_list_copper.py``'s decode tests build ``kipy`` objects by hand,
which cannot catch a field KiCad populates differently from the binding's
default.

Coverage limit: the fixture has no arcs and no vias, so this live run proves
nothing about the via drill units, the via layer spans, ``via_type`` or the
arc length -- those rest only on the hand-built-object decode tests in
``tests/test_list_copper.py``. A green run here does not cover them.

The fixture routes VIN with three F.Cu segments (3 + 10 + 3 mm) and leaves GND
and SIG unrouted; the assertions below are written against that and are
deliberate about what the fixture does NOT contain.
"""

from __future__ import annotations

import pytest

from openkonnect.tools.list_copper import list_copper

pytestmark = pytest.mark.live_kicad


def test_live_copper_matches_the_fixture_route(opened):
    r = opened.run_tool(lambda s: list_copper(s))
    assert r["ok"], r
    kinds = sorted(c["kind"] for c in r["copper"])
    assert kinds == ["track", "track", "track"], r["copper"]
    for c in r["copper"]:
        assert c["net"] == "VIN" and c["layer"] == "F.Cu"
        assert c["id_scope"] == "session" and c["no_file_reference"] is True
        assert c["id"] and isinstance(c["id"], str)
        assert c["width_mm"] == 0.125
        assert c["start_mm"]["frame"] == "file" and c["end_mm"]["frame"] == "file"
        # the fixture has an Edge.Cuts outline, so the board frame is defined
        assert c["start_board_mm"]["frame"] == "board"
        assert c["locked"] is False
    assert r["frame_offset"]["present"] is True
    # 3 + 10 + 3 mm of VIN track; KiCad's own lengths, to 1 um
    assert r["total_length_mm_by_net"].keys() == {"VIN"}
    assert abs(r["total_length_mm_by_net"]["VIN"] - 16.0) < 0.001


def test_live_filters(opened):
    assert opened.run_tool(lambda s: list_copper(s, net="VIN"))["count"] == 3
    # GND and SIG are unrouted in the fixture: an empty list, not an error
    r = opened.run_tool(lambda s: list_copper(s, net="GND"))
    assert r["ok"] and r["count"] == 0 and r["total_length_mm_by_net"] == {}
    assert opened.run_tool(lambda s: list_copper(s, layer="B.Cu"))["count"] == 0
    # the three segments run between (109, 57) and (119, 60) in the file frame
    box = {"x0_mm": 108, "y0_mm": 56, "x1_mm": 112, "y1_mm": 61, "frame": "file"}
    assert opened.run_tool(lambda s: list_copper(s, box=box))["count"] == 2
    # nothing in the fixture is locked
    assert opened.run_tool(lambda s: list_copper(s, locked_only=True))["count"] == 0


def test_live_unknown_layer_is_invalid_argument(opened):
    r = opened.run_tool(lambda s: list_copper(s, layer="F.Copper"))
    assert r["ok"] is False and r["error"]["code"] == "invalid_argument"
