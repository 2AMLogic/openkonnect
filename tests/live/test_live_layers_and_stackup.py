"""LAY-07 live: ``layers_and_stackup`` against ``tests/fixtures/selection.kicad_pcb`` open in KiCad.

Never runs in CI. Setup as for ``test_live_selection.py``: KiCad 10 with 'Enable
KiCad API' on, the fixture open and unmodified. Run ``uv run pytest --live
-m live_kicad -rs``. Read-only.

The fixture is a 2-copper-layer board. The exact stackup rows and their
materials/thicknesses depend on the KiCad build (a KiCad before 10.0.6 reports
rows with a type but no detail submessage), so only shape, order and the null
convention are asserted here, never specific material values.
"""

from __future__ import annotations

import pytest

from openkonnect.tools.layers_and_stackup import layers_and_stackup

pytestmark = pytest.mark.live_kicad


def test_live_layers_and_stackup(opened):
    r = opened.run_tool(lambda s: layers_and_stackup(s))
    assert r["ok"], r
    assert r["source"] == "live_editor"
    assert r["copper_layers"] == 2
    names = [layer["name"] for layer in r["layers"]]
    assert {"F.Cu", "B.Cu", "Edge.Cuts"} <= set(names)
    for layer in r["layers"]:
        assert isinstance(layer["name"], str) and isinstance(layer["user_name"], str)
    kinds = [row["kind"] for row in r["stackup"]]
    # not the top row: whether KiCad emits a disabled F.SilkS row at all is
    # an open question, so only non-emptiness is asserted here
    assert kinds
    assert kinds.count("copper") == 2
    assert kinds.count("dielectric") >= 1
    for row in r["stackup"]:
        # null, never a guessed default: unset thickness stays null
        assert row["thickness_mm"] is None or isinstance(row["thickness_mm"], float)
        if row["kind"] == "dielectric":
            assert row["layer"] is None
            # KiCad >= 10.0.6 populates the details oneof; before it, null
            if row["dielectric"] is not None:
                assert row["dielectric"]["type"] in ("core", "prepreg", "none", "unknown")
                for sub in row["dielectric"]["sub_layers"]:
                    # null convention: a KiCad reporting 0 maps to null
                    assert sub["epsilon_r"] is None or isinstance(sub["epsilon_r"], float)
                    assert sub["loss_tangent"] is None or isinstance(sub["loss_tangent"], float)
    assert r["total_thickness_mm"] is None or isinstance(r["total_thickness_mm"], float)
