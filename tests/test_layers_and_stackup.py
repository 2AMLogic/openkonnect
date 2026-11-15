"""LAY-07 ``layers_and_stackup`` against a fake PCB editor, plus the decode path.

The decode tests follow ``test_edge_outline.py``: real kipy objects built from
raw protos, handed to a minimal board fake, and fed to the module-level
``_layers_and_stackup`` with no KiCad running. They pin the two rules a fake
backend cannot prove: unset vs explicit-zero thickness must not collapse
(``null`` vs ``0.0``), and a stackup row whose detail oneof is empty (KiCad
before 10.0.6) reports ``dielectric: null`` instead of raising.
"""

from __future__ import annotations

from typing import Any

from kipy.board import BoardStackup
from kipy.proto.board import board_pb2
from kipy.proto.board.board_types_pb2 import BoardLayer

from openkonnect.backend import _layers_and_stackup
from openkonnect.tools.layers_and_stackup import describe_layers_and_stackup, layers_and_stackup

from .conftest import EditorFake, FakeBackend, board_ref, make_session


def run(backend=None):
    return make_session(backend if backend is not None else EditorFake()).run_tool(
        lambda s: layers_and_stackup(s)
    )


# -- fake backend (tool level) ---------------------------------------------------


def test_enabled_layers_copper_count_and_board():
    r = run(EditorFake(open_boards=[[board_ref()]]))
    assert r["ok"] is True, r
    assert r["source"] == "live_editor"
    assert r["copper_layers"] == 2
    assert {"name": "F.Cu", "user_name": "F.Cu"} in r["layers"]
    assert {"name": "Edge.Cuts", "user_name": "Edge.Cuts"} in r["layers"]
    assert r["board"]["path"] == "/proj/demo.kicad_pcb"


def test_stackup_top_to_bottom_with_mm_and_total():
    r = run()
    rows = r["stackup"]
    assert [row["kind"] for row in rows] == [
        "silkscreen",
        "copper",
        "dielectric",
        "copper",
        "soldermask",
    ]
    cu, di = rows[1], rows[2]
    assert cu["layer"] == "F.Cu" and cu["thickness_mm"] == 0.035 and cu["material"] is None
    assert di["layer"] is None and di["thickness_mm"] == 1.51 and di["material"] == "FR4"
    assert di["dielectric"] == {
        "type": "core",
        "sub_layers": [
            {"material": "FR4", "epsilon_r": 4.6, "loss_tangent": 0.02, "thickness_mm": 1.51}
        ],
    }
    # 35 um + 1.51 mm + 35 um; rows without a thickness do not contribute.
    assert r["total_thickness_mm"] == 1.58


def test_no_board_is_no_document():
    r = run(FakeBackend(open_boards=[[]]))
    assert r["ok"] is False and r["error"]["code"] == "no_document"


def test_unsaved_board_reports_null_path():
    ref = board_ref()
    ref.path = None
    r = run(EditorFake(open_boards=[[ref]]))
    assert r["ok"] is True and r["board"]["path"] is None


# -- decode path (real kipy protos on a board fake) --------------------------------


def stackup_layer_proto() -> board_pb2.BoardStackupLayer:
    return board_pb2.BoardStackupLayer()


def copper_row(layer: int, thickness_nm: int | None = 35_000, enabled: bool = True) -> Any:
    cu = stackup_layer_proto()
    cu.type = board_pb2.BSLT_COPPER
    cu.layer = layer
    cu.enabled = enabled
    if thickness_nm is not None:
        cu.thickness.value_nm = thickness_nm
    return cu


def dielectric_row(
    sub_layers: list[dict[str, Any]] | None, type_name: int = board_pb2.BSDT_CORE
) -> Any:
    di = stackup_layer_proto()
    di.type = board_pb2.BSLT_DIELECTRIC
    di.layer = BoardLayer.BL_UNDEFINED
    di.enabled = True
    di.thickness.value_nm = 1_510_000
    di.material_name = "FR4"
    if sub_layers is not None:
        detail = board_pb2.BoardStackupDielectricLayer()
        detail.type = type_name
        for sub in sub_layers:
            entry = detail.layer.add()
            entry.material_name = sub.get("material", "")
            entry.epsilon_r = sub.get("epsilon_r", 0.0)
            entry.loss_tangent = sub.get("loss_tangent", 0.0)
            if (nm := sub.get("thickness_nm")) is not None:
                entry.thickness.value_nm = nm
        di.dielectric.CopyFrom(detail)
    return di


def board_fake(stackup_rows: list[Any], names: dict[int, str] | None = None) -> Any:
    """A board carrying just what ``_layers_and_stackup`` calls, with real protos."""

    stackup = board_pb2.BoardStackup()
    for row in stackup_rows:
        stackup.layers.append(row)

    enabled = [
        BoardLayer.BL_F_Cu,
        BoardLayer.BL_B_Cu,
        BoardLayer.BL_F_SilkS,
        BoardLayer.BL_F_Mask,
        BoardLayer.BL_Edge_Cuts,
        BoardLayer.BL_Dwgs_User,
    ]

    class Board:
        def get_stackup(self) -> BoardStackup:
            return BoardStackup(stackup)

        def get_enabled_layers(self) -> list[int]:
            return list(enabled)

        def get_copper_layer_count(self) -> int:
            return 2

        def get_layer_name(self, layer: int) -> str:
            return (names or {}).get(layer, "F.Cu")

    class Backend:
        def _board(self, ref: Any) -> Board:
            return Board()

    return Backend()


def decode(rows: list[Any], names: dict[int, str] | None = None) -> dict[str, Any]:
    return _layers_and_stackup(board_fake(rows, names), board_ref())


def test_decode_copper_row():
    d = decode([copper_row(BoardLayer.BL_F_Cu)])
    assert d["copper_layers"] == 2
    assert d["stackup"] == [
        {
            "kind": "copper",
            "layer": "F.Cu",
            "user_name": None,
            "enabled": True,
            "thickness_nm": 35_000,
            "material": None,
            "dielectric": None,
        }
    ]


def test_decode_dielectric_row_has_null_layer_and_sub_layers():
    d = decode([dielectric_row([{"material": "FR4", "epsilon_r": 4.6, "loss_tangent": 0.02}])])
    row = d["stackup"][0]
    assert row["kind"] == "dielectric"
    assert row["layer"] is None  # a dielectric is on no board layer (BL_UNDEFINED)
    assert row["material"] == "FR4"
    assert row["dielectric"]["type"] == "core"
    assert row["dielectric"]["sub_layers"] == [
        {"material": "FR4", "epsilon_r": 4.6, "loss_tangent": 0.02, "thickness_nm": None}
    ]


def test_decode_multi_sub_layer_dielectric():
    subs = [
        {"material": "FR4", "epsilon_r": 4.6, "loss_tangent": 0.02, "thickness_nm": 1_000_000},
        {
            "material": "Prepreg 2116",
            "epsilon_r": 4.2,
            "loss_tangent": 0.015,
            "thickness_nm": 510_000,
        },
    ]
    d = decode([dielectric_row(subs, type_name=board_pb2.BSDT_PREPREG)])
    row = d["stackup"][0]
    assert row["dielectric"]["type"] == "prepreg"
    assert [s["thickness_nm"] for s in row["dielectric"]["sub_layers"]] == [1_000_000, 510_000]
    assert row["dielectric"]["sub_layers"][1]["material"] == "Prepreg 2116"


def test_decode_row_without_thickness_vs_explicit_zero():
    d = decode([copper_row(BoardLayer.BL_F_Cu, thickness_nm=None)])
    assert d["stackup"][0]["thickness_nm"] is None  # KiCad gave nothing -> null
    d = decode([copper_row(BoardLayer.BL_F_Cu, thickness_nm=0)])
    assert d["stackup"][0]["thickness_nm"] == 0  # KiCad gave 0 -> 0, not null


def test_decode_soldermask_row_without_detail_oneof():
    sm = stackup_layer_proto()
    sm.type = board_pb2.BSLT_SOLDERMASK
    sm.enabled = True  # pre-10.0.6 shape: a type, no details submessage
    d = decode([sm])
    row = d["stackup"][0]
    assert row["kind"] == "soldermask"
    assert row["dielectric"] is None  # degraded to null, not raised
    assert row["material"] is None


def test_decode_disabled_row_is_reported_with_its_flag():
    d = decode([copper_row(BoardLayer.BL_B_Cu, enabled=False)])
    assert d["stackup"][0]["enabled"] is False


def test_decode_empty_stackup_and_enabled_layers():
    d = decode([])
    assert d["stackup"] == []
    assert d["layers"] == [
        {"name": "F.Cu", "user_name": "F.Cu"},
        {"name": "B.Cu", "user_name": "F.Cu"},  # names are canned per-layer in the fake
        {"name": "F.SilkS", "user_name": "F.Cu"},
        {"name": "F.Mask", "user_name": "F.Cu"},
        {"name": "Edge.Cuts", "user_name": "F.Cu"},
        {"name": "Dwgs.User", "user_name": "F.Cu"},
    ]


def test_decode_user_names_come_from_the_board_per_layer():
    names = {
        BoardLayer.BL_F_Cu: "Front",
        BoardLayer.BL_B_Cu: "Back",
        BoardLayer.BL_F_SilkS: "F.SilkS",
        BoardLayer.BL_F_Mask: "F.Mask",
        BoardLayer.BL_Edge_Cuts: "Edge.Cuts",
        BoardLayer.BL_Dwgs_User: "Dwgs.User",
    }
    d = decode([], names=names)
    assert d["layers"] == [
        {"name": "F.Cu", "user_name": "Front"},
        {"name": "B.Cu", "user_name": "Back"},
        {"name": "F.SilkS", "user_name": "F.SilkS"},
        {"name": "F.Mask", "user_name": "F.Mask"},
        {"name": "Edge.Cuts", "user_name": "Edge.Cuts"},
        {"name": "Dwgs.User", "user_name": "Dwgs.User"},
    ]


# -- the null rule through the tool decode ----------------------------------------


def backend_row(**over: Any) -> dict[str, Any]:
    """A full stackup row in the backend's form, overridden per test."""
    row: dict[str, Any] = {
        "kind": "copper",
        "layer": "F.Cu",
        "user_name": None,
        "enabled": True,
        "thickness_nm": None,
        "material": None,
        "dielectric": None,
    }
    row.update(over)
    return row


def test_unset_and_zero_thickness_do_not_collapse_in_mm():
    data = {
        "layers": [],
        "copper_layers": 2,
        "stackup": [backend_row(), backend_row(thickness_nm=0)],
    }
    with_unset, with_zero = describe_layers_and_stackup(data)["stackup"]
    assert with_unset["thickness_mm"] is None
    assert with_zero["thickness_mm"] == 0.0
    # the total counts the row that carries a thickness, including the zero
    assert describe_layers_and_stackup(data)["total_thickness_mm"] == 0.0


def test_no_thickness_anywhere_is_a_null_total_not_zero():
    data = {"layers": [], "copper_layers": 2, "stackup": [backend_row()]}
    r = describe_layers_and_stackup(data)
    assert r["total_thickness_mm"] is None


def test_empty_stackup_is_an_empty_list_not_a_zero_board():
    data = {"layers": [], "copper_layers": 2, "stackup": []}
    r = describe_layers_and_stackup(data)
    assert r["stackup"] == [] and r["total_thickness_mm"] is None
