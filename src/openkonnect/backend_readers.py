"""Selection, board index and the LAY-03..LAY-07 board readers for the kipy backend.

The kipy item decoders, selection (#5), board index and net / zone / stackup /
copper / pad reader block moved here out of ``backend.py``: it is pure kipy
decoding over the board a ``KipyBackend`` holds, reached only through that
class's one-line delegators. See the banner below for why the selection path
sends its own requests instead of calling kipy's ``get_selection``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .backend_outline import _edge_outline

if TYPE_CHECKING:
    from .backend import BoardRef, KipyBackend


# -- selection (#5) ----------------------------------------------------------
#
# Verified against kicad-python 0.8.0's own source (MIT):
#   Board.get_selection(types)   -> editor GetSelection / SelectionResponse
#   Board.clear_selection()      -> editor ClearSelection
#   Board.add_to_selection(items)-> editor AddToSelection (sends item ids)
#   Board.get_items_by_id(ids)   -> GetItemsById (KiCad 10.0.0)
#   Board.get_item_bounding_box  -> editor GetBoundingBox
# kipy's get_selection unwraps every item with an assert, so one item of a
# kind the client does not know fails the whole call. ``_read_selection``
# sends the same GetSelection request and unwraps item by item instead, so
# such items are counted as undescribed rather than failing or vanishing.
# kipy's list form of get_item_bounding_box drops items without a box and
# so loses the item-to-box pairing; ``_boxes`` sends the same GetBoundingBox
# request and keeps the pairing from the response's ``items`` field.
# No zoom, view or highlight request is used: kicad-python 0.8 documents none
# (only KiCad.run_action, marked unstable), see spec SES-14/15.


def _kind_types() -> dict[str, Any]:
    from kipy.proto.common.types import KiCadObjectType as T

    return {
        "footprint": T.KOT_PCB_FOOTPRINT,
        "pad": T.KOT_PCB_PAD,
        "track": T.KOT_PCB_TRACE,
        "arc": T.KOT_PCB_ARC,
        "via": T.KOT_PCB_VIA,
        "zone": T.KOT_PCB_ZONE,
        "shape": T.KOT_PCB_SHAPE,
        "text": T.KOT_PCB_TEXT,
        "textbox": T.KOT_PCB_TEXTBOX,
        "field": T.KOT_PCB_FIELD,
        "dimension": T.KOT_PCB_DIMENSION,
        "barcode": T.KOT_PCB_BARCODE,
        "reference_image": T.KOT_PCB_REFERENCE_IMAGE,
        "group": T.KOT_PCB_GROUP,
    }


def _kind_of(item: Any) -> str | None:
    from kipy import board_types as bt

    for cls, kind in (
        (bt.FootprintInstance, "footprint"),
        (bt.Pad, "pad"),
        (bt.ArcTrack, "arc"),
        (bt.Track, "track"),
        (bt.Via, "via"),
        (bt.Zone, "zone"),
        (bt.BoardShape, "shape"),
        (bt.BoardTextBox, "textbox"),
        (bt.BoardText, "text"),
        (bt.Field, "field"),
        (bt.Dimension, "dimension"),
        (bt.Barcode, "barcode"),
        (bt.ReferenceImage, "reference_image"),
        (bt.Group, "group"),
    ):
        if isinstance(item, cls):
            return kind
    return None


def _layer_name(layer: Any) -> str | None:
    from kipy.util.board_layer import canonical_name

    name = canonical_name(layer)
    return None if name == "Unknown" else name


def _vec(v: Any) -> dict[str, int]:
    return {"x": int(v.x), "y": int(v.y)}


def _net_name(net: Any) -> str | None:
    return (net.name or None) if net is not None else None


def _footprint_ref(fp: Any) -> str:
    return fp.reference_field.text.value


def _item_id(item: Any) -> str | None:
    """A board item's KIID as a string; fields have no KIID of their own."""
    from kipy.board_types import Field

    if isinstance(item, Field):
        return None
    value = getattr(item.id, "value", None)
    return value or None


def _boxes(kicad_client: Any, doc: Any, items: list[Any]) -> dict[str, dict[str, int]]:
    from kipy.proto.common.commands import editor_commands_pb2 as ed

    with_ids = [i for i in items if _item_id(i)]
    if not with_ids:
        return {}
    cmd = ed.GetBoundingBox()
    cmd.header.document.CopyFrom(doc)
    cmd.mode = ed.BoundingBoxMode.BBM_ITEM_ONLY
    cmd.items.extend([i.id for i in with_ids])
    resp = kicad_client.send(cmd, ed.GetBoundingBoxResponse)
    out = {}
    for kiid, box in zip(resp.items, resp.boxes, strict=False):
        x, y = int(box.position.x_nm), int(box.position.y_nm)
        out[kiid.value] = {
            "x0_nm": x,
            "y0_nm": y,
            "x1_nm": x + int(box.size.x_nm),
            "y1_nm": y + int(box.size.y_nm),
        }
    return out


def _pad_owners(footprints: list[Any]) -> dict[str, tuple[str, str]]:
    """Pad KIID -> (footprint reference, pad number), from footprint definitions."""
    owners: dict[str, tuple[str, str]] = {}
    for fp in footprints:
        ref = _footprint_ref(fp)
        for pad in fp.definition.pads:
            if pad.id.value:
                owners[pad.id.value] = (ref, pad.number)
    return owners


def _footprint_details(fp: Any) -> dict[str, Any]:
    """The LAY-02 extras of a footprint: value, library ID, rotation, lock, attributes, fields."""
    from kipy.board_types import Field
    from kipy.proto.board import board_types_pb2 as bt

    lib = fp.definition.id
    library_id = f"{lib.library}:{lib.name}" if lib.library else (lib.name or None)
    attrs = fp.attributes
    mounting = {
        bt.FootprintMountingStyle.FMS_SMD: "smd",
        bt.FootprintMountingStyle.FMS_THROUGH_HOLE: "through_hole",
    }.get(attrs.mounting_style)
    fields: dict[str, str] = {}
    for f in (*fp.definition.items, fp.datasheet_field, fp.description_field):
        if isinstance(f, Field) and f.name and f.text.value:
            fields[f.name] = f.text.value
    return {
        "value": fp.value_field.text.value,
        "library_id": library_id,
        "orientation_deg": round(fp.orientation.degrees, 6),
        "locked": bool(fp.locked),
        "attributes": {
            "do_not_populate": bool(attrs.do_not_populate),
            "exclude_from_bom": bool(attrs.exclude_from_bill_of_materials),
            "exclude_from_position_files": bool(attrs.exclude_from_position_files),
            "not_in_schematic": bool(attrs.not_in_schematic),
            "mounting": mounting,
        },
        "fields": fields,
    }


def _describe(item: Any, kind: str, owners: dict[str, tuple[str, str]]) -> dict[str, Any]:
    d: dict[str, Any] = {"id": _item_id(item), "kind": kind, "net": None, "layer": None}
    if kind == "footprint":
        d["ref"] = _footprint_ref(item)
        d["layer"] = _layer_name(item.layer)
        d["position_nm"] = _vec(item.position)
    elif kind == "pad":
        owner = owners.get(d["id"] or "")
        if owner is not None:
            d["ref"], d["pad"] = owner
        d["net"] = _net_name(item.net)
        d["position_nm"] = _vec(item.position)
    elif kind in ("track", "arc"):
        d["net"] = _net_name(item.net)
        d["layer"] = _layer_name(item.layer)
    elif kind == "via":
        d["net"] = _net_name(item.net)
        d["position_nm"] = _vec(item.position)
    elif kind == "zone":
        d["net"] = _net_name(item.net)
        d["layers"] = [n for n in (_layer_name(lyr) for lyr in item.layers) if n]
        d["name"] = item.name or None
    elif kind == "field":
        d["name"] = item.name
        d["layer"] = _layer_name(item.layer)
    elif kind == "group":
        d["name"] = item.name or None
    else:
        layer = getattr(item, "layer", None)
        d["layer"] = _layer_name(layer) if layer is not None else None
        if kind == "shape":
            d["net"] = _net_name(item.net)
    return d


def _read_selection(
    kicad_client: Any, doc: Any, types: list[Any]
) -> tuple[list[Any], dict[str, int]]:
    from kipy.board_types import (
        BoardShape,
        Dimension,
        to_concrete_board_shape,
        to_concrete_dimension,
        unwrap,
    )
    from kipy.proto.common.commands import editor_commands_pb2 as ed

    cmd = ed.GetSelection()
    cmd.header.document.CopyFrom(doc)
    cmd.types.extend(types)
    resp = kicad_client.send(cmd, ed.SelectionResponse)
    items: list[Any] = []
    undescribed: dict[str, int] = {}
    for raw in resp.items:
        name = raw.type_url.rsplit("/", 1)[-1] or "unknown"
        try:
            item = unwrap(raw)
        except (AssertionError, NotImplementedError, ValueError):
            undescribed[name] = undescribed.get(name, 0) + 1
            continue
        if isinstance(item, BoardShape):
            item = to_concrete_board_shape(item)
        elif isinstance(item, Dimension):
            item = to_concrete_dimension(item)
        if _kind_of(item) is None:
            undescribed[name] = undescribed.get(name, 0) + 1
            continue
        items.append(item)
    return items, undescribed


def _selection(self: KipyBackend, ref: BoardRef, kinds: list[str] | None = None) -> dict[str, Any]:
    board = self._board(ref)
    client = self._kicad._client
    table = _kind_types()
    types = [table[k] for k in (kinds or [])]
    items, undescribed = _read_selection(client, ref.handle, types)
    owners = (
        _pad_owners(list(board.get_footprints()))
        if any(_kind_of(i) == "pad" for i in items)
        else {}
    )
    boxes = _boxes(client, ref.handle, items)
    described = []
    for item in items:
        kind = _kind_of(item) or "unknown"
        d = _describe(item, kind, owners)
        if kind == "footprint":
            d["pads"] = [
                {"id": p.id.value or None, "number": p.number, "net": _net_name(p.net)}
                for p in item.definition.pads
            ]
        d["bbox_nm"] = boxes.get(d["id"] or "")
        described.append(d)
    _, outline = _edge_outline(board)
    return {"items": described, "undescribed": undescribed, "outline_nm": outline}


def _board_index(self: KipyBackend, ref: BoardRef) -> dict[str, Any]:
    board = self._board(ref)
    client = self._kicad._client
    footprints = list(board.get_footprints())
    pads = list(board.get_pads())
    copper = [*board.get_tracks(), *board.get_vias(), *board.get_zones()]
    owners = _pad_owners(footprints)
    boxes = _boxes(client, ref.handle, [*footprints, *pads, *copper])
    fps = []
    by_fp: dict[str, list[dict[str, Any]]] = {}
    for p in pads:
        d = _describe(p, "pad", owners)
        d["bbox_nm"] = boxes.get(d["id"] or "")
        if "ref" in d:
            by_fp.setdefault(d["ref"], []).append(d)
    for fp in footprints:
        d = _describe(fp, "footprint", owners)
        d.update(_footprint_details(fp))
        d["bbox_nm"] = boxes.get(d["id"] or "")
        d["pads"] = by_fp.get(d["ref"], [])
        fps.append(d)
    cu = []
    for item in copper:
        d = _describe(item, _kind_of(item) or "unknown", owners)
        d["bbox_nm"] = boxes.get(d["id"] or "")
        cu.append(d)
    nets = [n.name for n in board.get_nets() if n.name]
    _, outline = _edge_outline(board)
    return {"footprints": fps, "copper": cu, "nets": nets, "outline_nm": outline}


def _nets(self: KipyBackend, ref: BoardRef, include_items: bool = False) -> dict[str, Any]:
    board = self._board(ref)
    nets = [n for n in board.get_nets() if n.name]
    classes = board.get_netclass_for_nets(nets) if nets else {}
    pads = list(board.get_pads())
    pad_counts: dict[str, int] = {}
    for p in pads:
        name = _net_name(p.net)
        if name:
            pad_counts[name] = pad_counts.get(name, 0) + 1
    items: dict[str, list[dict[str, Any]]] = {}
    if include_items:
        owners = _pad_owners(list(board.get_footprints()))
        for p in pads:
            d = _describe(p, "pad", owners)
            if d["net"]:
                items.setdefault(d["net"], []).append(
                    {k: d[k] for k in ("id", "kind", "ref", "pad") if k in d}
                )
        for c in [*board.get_tracks(), *board.get_vias(), *board.get_zones()]:
            net = _net_name(c.net)
            if net:
                items.setdefault(net, []).append({"id": _item_id(c), "kind": _kind_of(c)})
    out = []
    for n in nets:
        cls = classes.get(n.name)
        entry: dict[str, Any] = {
            "name": n.name,
            "net_class": (cls.name or None) if cls is not None else None,
            "pad_count": pad_counts.get(n.name, 0),
        }
        if include_items:
            entry["items"] = items.get(n.name, [])
        out.append(entry)
    return {"nets": out}


def _zones(self: KipyBackend, ref: BoardRef) -> dict[str, Any]:
    from kipy.proto.board.board_types_pb2 import ZoneConnectionStyle, ZoneType

    board = self._board(ref)
    out = []
    for z in board.get_zones():
        rule_area = z.type == ZoneType.ZT_RULE_AREA
        keepouts = pad_connection = None
        if rule_area:
            ra = z.proto.rule_area_settings
            keepouts = {
                "copper": bool(ra.keepout_copper),
                "vias": bool(ra.keepout_vias),
                "tracks": bool(ra.keepout_tracks),
                "pads": bool(ra.keepout_pads),
                "footprints": bool(ra.keepout_footprints),
            }
        else:
            conn = z.connection
            if conn is not None:
                pad_connection = (
                    ZoneConnectionStyle.Name(conn.zone_connection).removeprefix("ZCS_").lower()
                )
        kind = (
            "copper"
            if z.type == ZoneType.ZT_COPPER
            else "rule_area"
            if rule_area
            else ZoneType.Name(z.type).removeprefix("ZT_").lower()
        )
        out.append(
            {
                "id": _item_id(z),
                "kind": kind,
                "name": z.name or None,
                "net": None if rule_area else _net_name(z.net),
                "layers": [n for n in (_layer_name(lyr) for lyr in z.layers) if n],
                "priority": int(z.priority),
                "pad_connection": pad_connection,
                "filled": bool(z.filled),
                "locked": bool(z.locked),
                "keepouts": keepouts,
            }
        )
    return {"zones": out}


def _stackup_dielectric(dielectric: Any) -> dict[str, Any] | None:
    """A stackup row's dielectric detail as plain data, or None when KiCad gives none.

    The detail submessage sits in a ``details`` oneof that is independent of the
    row's ``type``: KiCad 10.0.0-10.0.5 reports the type but leaves the oneof
    empty, and an empty oneof is ``null``, never a guessed default (C-2).
    ``epsilon_r`` / ``loss_tangent`` / ``material_name`` are plain proto3 scalars
    with no presence, so an unset one maps to ``null`` on the falsy convention;
    ``thickness`` does have presence (``HasField``), so unset vs explicit zero
    stay distinct.
    """
    if dielectric is None:
        return None
    from kipy.proto.board.board_pb2 import BoardStackupDielectricType

    sub_layers = []
    for sub in dielectric.layers:
        sub_layers.append(
            {
                "material": sub.material_name or None,
                "epsilon_r": sub.epsilon_r or None,
                "loss_tangent": sub.loss_tangent or None,
                "thickness_nm": int(sub.thickness) if sub.proto.HasField("thickness") else None,
            }
        )
    return {
        "type": BoardStackupDielectricType.Name(dielectric.type).removeprefix("BSDT_").lower(),
        "sub_layers": sub_layers,
    }


def _layers_and_stackup(self: KipyBackend, ref: BoardRef) -> dict[str, Any]:
    from kipy.proto.board.board_pb2 import BoardStackupLayerType

    board = self._board(ref)
    layers = []
    for lyr in board.get_enabled_layers():
        name = _layer_name(lyr)
        if name is None:
            continue
        layers.append({"name": name, "user_name": board.get_layer_name(lyr)})

    rows = []
    for entry in board.get_stackup().layers:
        rows.append(
            {
                "kind": BoardStackupLayerType.Name(entry.type).removeprefix("BSLT_").lower(),
                "layer": _layer_name(entry.layer),
                "user_name": entry.user_name or None,
                "enabled": bool(entry.enabled),
                "thickness_nm": (
                    int(entry.thickness) if entry.proto.HasField("thickness") else None
                ),
                "material": entry.material_name or None,
                "dielectric": _stackup_dielectric(entry.dielectric),
            }
        )
    return {"layers": layers, "copper_layers": board.get_copper_layer_count(), "stackup": rows}


def _via_layers(via: Any) -> list[str]:
    """A via's layer span: the canonical names of its drill's start and end layers."""
    drill = via.padstack.drill
    names = [_layer_name(lyr) for lyr in (drill.start_layer, drill.end_layer)]
    return [n for n in names if n]


def _via_diameter(via: Any) -> int | None:
    """The via's copper diameter, or None when KiCad reports no copper layer for it.

    ``Via.diameter`` reads the first copper layer of the padstack and raises when
    there is none; a diameter is never guessed (C-2), so that case is ``null``.
    """
    try:
        return int(via.diameter)
    except (ValueError, IndexError):
        return None


def _copper(self: KipyBackend, ref: BoardRef) -> dict[str, Any]:
    from kipy.board_types import ArcTrack
    from kipy.proto.board.board_types_pb2 import ViaType

    board = self._board(ref)
    out: list[dict[str, Any]] = []
    for t in board.get_tracks():
        is_arc = isinstance(t, ArcTrack)
        entry: dict[str, Any] = {
            "id": _item_id(t),
            "kind": "arc" if is_arc else "track",
            "net": _net_name(t.net),
            "layer": _layer_name(t.layer),
            "width_nm": int(t.width),
            "start_nm": _vec(t.start),
            "end_nm": _vec(t.end),
            # kipy's ArcTrack.length() is the true arc length (angle * radius),
            # falling back to the chord only for a degenerate arc.
            "length_nm": float(t.length()),
            "locked": bool(t.locked),
        }
        if is_arc:
            entry["mid_nm"] = _vec(t.mid)
        out.append(entry)
    for v in board.get_vias():
        out.append(
            {
                "id": _item_id(v),
                "kind": "via",
                "net": _net_name(v.net),
                "layers": _via_layers(v),
                "position_nm": _vec(v.position),
                "diameter_nm": _via_diameter(v),
                "drill_nm": int(v.drill_diameter),
                "via_type": ViaType.Name(v.type).removeprefix("VT_").lower(),
                "locked": bool(v.locked),
            }
        )
    _, outline = _edge_outline(board)
    return {"copper": out, "outline_nm": outline}


def _pad_geometry(pad: Any) -> dict[str, Any]:
    """Layers, shape and size of a pad, from its padstack (the first copper layer entry)."""
    from kipy.proto.board.board_types_pb2 import PadStackShape

    stack = pad.padstack
    layers = [n for n in (_layer_name(lyr) for lyr in stack.layers) if n]
    shape = size = None
    copper = stack.copper_layers
    if copper:
        shape = PadStackShape.Name(copper[0].shape).removeprefix("PSS_").lower()
        size = {"x": int(copper[0].size.x), "y": int(copper[0].size.y)}
    return {"layers": layers, "shape": shape, "size_nm": size}


def _pad_net(pad: Any) -> dict[str, Any]:
    """The pad's net and whether it could be read; unreadable is not unconnected."""
    try:
        name = _net_name(pad.net)
    except Exception as e:  # a net that cannot be decoded must not read as unconnected
        return {"net": None, "net_state": "unknown", "net_reason": f"{type(e).__name__}: {e}"}
    if name is None:
        return {"net": None, "net_state": "unconnected"}
    return {"net": name, "net_state": "connected"}


def _pads(self: KipyBackend, ref: BoardRef, refs: list[str] | None = None) -> dict[str, Any]:
    board = self._board(ref)
    footprints = list(board.get_footprints())
    owners = _pad_owners(footprints)
    wanted = set(refs) if refs is not None else None
    out = []
    for p in board.get_pads():
        owner = owners.get(p.id.value or "")
        if owner is None or (wanted is not None and owner[0] not in wanted):
            continue
        d: dict[str, Any] = {"ref": owner[0], "pad": owner[1], "position_nm": _vec(p.position)}
        d.update(_pad_net(p))
        d.update(_pad_geometry(p))
        out.append(d)
    _, outline = _edge_outline(board)
    return {
        "footprints": [_footprint_ref(fp) for fp in footprints],
        "pads": out,
        "outline_nm": outline,
    }


def _set_selection(self: KipyBackend, ref: BoardRef, ids: list[str]) -> None:
    from kipy.proto.common.types import KIID

    board = self._board(ref)
    board.clear_selection()
    if ids:
        items = board.get_items_by_id([KIID(value=i) for i in ids])
        if items:
            board.add_to_selection(list(items))
