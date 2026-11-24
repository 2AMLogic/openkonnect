"""LAY-03 live: ``footprint_pads`` against ``tests/fixtures/selection.kicad_pcb`` open in KiCad.

Never runs in CI. Setup as for ``test_live_selection.py``: KiCad 10 with 'Enable
KiCad API' on, the fixture open and unmodified. Run ``uv run pytest --live
-m live_kicad -rs``. Read-only.
"""

from __future__ import annotations

import pytest

from openkonnect.tools.footprint_pads import footprint_pads

pytestmark = pytest.mark.live_kicad


def test_live_pads_match_the_fixture(opened):
    r = opened.run_tool(lambda s: footprint_pads(s, ["R1", "R2", "R3"]))
    assert r["ok"], r
    assert {p["name"]: p["net"] for p in r["pads"]} == {
        "R1.1": "VIN",
        "R1.2": "GND",
        "R2.1": "VIN",
        "R2.2": "SIG",
        "R3.1": "SIG",
        "R3.2": "GND",
    }
    for p in r["pads"]:
        assert p["layers"] == ["F.Cu", "F.Mask", "F.Paste"]
    r1 = {p["name"]: p for p in r["pads"]}["R1.1"]
    assert r1["position_mm"] == {"frame": "file", "x_mm": 109.0, "y_mm": 60.0}


def test_live_unknown_pad_is_not_found(opened):
    r = opened.run_tool(lambda s: footprint_pads(s, ["R1.9"]))
    assert r["error"]["code"] == "not_found"
