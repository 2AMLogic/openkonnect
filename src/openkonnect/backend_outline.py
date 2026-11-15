"""Board-outline geometry for the kipy IPC backend.

The LAY-08 outline/edge-chaining block moved here out of ``backend.py``: it is
pure geometry over kipy shapes (extents, arc/circle/bezier sampling, shape
classification, endpoint chaining) consumed via :func:`_outline` and
:func:`_edge_outline`. See the banner below for why this code exists.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from .errors import ErrorCode, OpenKonnectError

if TYPE_CHECKING:
    from .backend import BoardRef, KipyBackend


def _edge_outline(board: Any) -> tuple[list[Any], dict[str, int] | None]:
    """Board-level Edge.Cuts shapes and the box of their geometry (file frame, nm).

    The box is the shapes' own geometry (corners, endpoints, arc, circle and
    curve extents), not ``get_item_bounding_box``: KiCad's box includes the
    drawn line width, which moved the board-frame origin by half a stroke (#32).

    ``Board.get_shapes`` would give the same shapes (kipy 0.8.0 keeps an
    unclassified shape as a bare ``BoardShape``); ``get_items`` is used so that
    shape is visible here, and an Edge.Cuts shape that is not one of the six
    kipy shape classes raises ``internal_error`` instead of being skipped, so
    the outline is never silently wrong.

    Only board-level Edge.Cuts graphics count; footprint-owned ones are
    excluded to match ``kct``'s board origin (C-1, #41). No filter is needed:
    KiCad 10.0.7 does not return footprint children from board-level
    ``GetItems(KOT_PCB_SHAPE)`` (they come only through
    ``FootprintInstance.definition``; docs/ipc-client.md).
    """
    from kipy.board_types import to_concrete_board_shape
    from kipy.proto.board.board_types_pb2 import BoardLayer
    from kipy.proto.common.types import KiCadObjectType

    edge = [
        to_concrete_board_shape(s)
        for s in board.get_items([KiCadObjectType.KOT_PCB_SHAPE])
        if s.layer == BoardLayer.BL_Edge_Cuts
    ]
    extents = [e for e in (_shape_extent(s) for s in edge) if e is not None]
    outline = None
    if extents:
        outline = {
            "x0_nm": min(e[0] for e in extents),
            "y0_nm": min(e[1] for e in extents),
            "x1_nm": max(e[2] for e in extents),
            "y1_nm": max(e[3] for e in extents),
        }
    return edge, outline


Extent = tuple[int, int, int, int]  # x0, y0, x1, y1 in nm


def _points_extent(points: list[Any]) -> Extent:
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _box_extent(box: Any) -> Extent:
    return box.pos.x, box.pos.y, box.pos.x + box.size.x, box.pos.y + box.size.y


def _shape_extent(shape: Any) -> Extent | None:
    """One Edge.Cuts shape's geometric extent, without stroke width.

    ``None`` only for a polygon with no outline points (it bounds nothing).
    """
    from kipy.board_types import (
        BoardArc,
        BoardBezier,
        BoardCircle,
        BoardPolygon,
        BoardRectangle,
        BoardSegment,
    )
    from kipy.geometry import arc_bounding_box

    # Exact type checks: kipy leaves an unclassified shape as a bare BoardShape.
    kind = type(shape)
    if kind is BoardRectangle:
        return _points_extent([shape.top_left, shape.bottom_right])
    if kind is BoardSegment:
        return _points_extent([shape.start, shape.end])
    if kind is BoardArc:
        return _box_extent(arc_bounding_box(shape.start, shape.mid, shape.end))
    if kind is BoardCircle:
        return _box_extent(shape.bounding_box())
    if kind is BoardBezier:
        return _bezier_extent(shape.start, shape.control1, shape.control2, shape.end)
    if kind is BoardPolygon:
        extents: list[Extent] = []
        for poly in shape.polygons:  # holes lie inside their outline
            for node in poly.outline:
                if node.has_point:
                    extents.append(_points_extent([node.point]))
                elif node.has_arc:
                    a = node.arc
                    extents.append(_box_extent(arc_bounding_box(a.start, a.mid, a.end)))
        if not extents:
            return None
        return (
            min(e[0] for e in extents),
            min(e[1] for e in extents),
            max(e[2] for e in extents),
            max(e[3] for e in extents),
        )
    proto = getattr(shape, "_proto", None)
    geometry = None
    with contextlib.suppress(Exception):
        geometry = proto.shape.WhichOneof("geometry")  # type: ignore[union-attr]
    shape_id = None
    with contextlib.suppress(Exception):
        shape_id = shape.id.value or None
    raise OpenKonnectError(
        ErrorCode.INTERNAL,
        "An Edge.Cuts shape has a geometry openkonnect cannot measure, so the board "
        "outline (and the board frame) cannot be computed. Report it with these details.",
        details={"shape_id": shape_id, "geometry": geometry, "class": kind.__name__},
    )


def _bezier_extent(p0: Any, p1: Any, p2: Any, p3: Any) -> Extent:
    """Extent of a cubic Bezier: its endpoints plus its interior extrema per axis.

    B'(t) is proportional to (a - 2b + c) t^2 + 2 (b - a) t + a with
    a = P1 - P0, b = P2 - P1, c = P3 - P2; its roots in (0, 1) are the extrema.
    The control points themselves only bound the curve, so they are not used.
    """

    def axis(v0: float, v1: float, v2: float, v3: float) -> tuple[int, int]:
        a, b, c = v1 - v0, v2 - v1, v3 - v2
        qa, qb, qc = a - 2 * b + c, 2 * (b - a), a
        ts: list[float] = []
        if qa == 0:
            if qb != 0:
                ts.append(-qc / qb)
        else:
            disc = qb * qb - 4 * qa * qc
            if disc >= 0:
                r = disc**0.5
                ts += [(-qb + r) / (2 * qa), (-qb - r) / (2 * qa)]
        values = [float(v0), float(v3)]
        for t in ts:
            if 0 < t < 1:
                u = 1 - t
                values.append(u**3 * v0 + 3 * u * u * t * v1 + 3 * u * t * t * v2 + t**3 * v3)
        return round(min(values)), round(max(values))

    x0, x1 = axis(p0.x, p1.x, p2.x, p3.x)
    y0, y1 = axis(p0.y, p1.y, p2.y, p3.y)
    return x0, y0, x1, y1


# -- board outline polygons (LAY-08, #88) ------------------------------------
#
# kipy 0.8.0 has no board-outline helper: ``Board`` offers no
# ``get_board_outline()`` and ``kipy.geometry`` only measures single arcs. The
# chaining below is therefore ours. ``kicad-tools``' two implementations were
# read and deliberately not reused: both fall back to a bounding box (one to a
# hardcoded 100 x 100 mm board) when chaining fails, which LAY-08 forbids, and
# both read the sexp file rather than the open editor (C-7). Their greedy
# endpoint matching and 0.01 mm tolerance are the reference for the tolerance
# below.

#: Endpoints this far apart or closer are the same point when chaining (0.01 mm).
#: The value ``kct``'s ``_chain_lines_to_polygon`` uses, so the two agree about
#: which outlines close.
_CHAIN_TOLERANCE_NM = 10_000

#: Sampling step for arcs and circles. Curves become polylines because LAY-08
#: asks for polygons; the step is reported so a consumer knows the loop is an
#: approximation.
_ARC_STEP_DEG = 5.0

#: Segments a cubic bezier is sampled into (beziers have no angular measure).
_BEZIER_STEPS = 24


def _pt(p: Any) -> dict[str, int]:
    return {"x": int(p.x), "y": int(p.y)}


def _near(a: dict[str, int], b: dict[str, int], tol: int = _CHAIN_TOLERANCE_NM) -> bool:
    return abs(a["x"] - b["x"]) <= tol and abs(a["y"] - b["y"]) <= tol


def _gap_nm(a: dict[str, int], b: dict[str, int]) -> int:
    return round(((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2) ** 0.5)


def _arc_points(start: Any, mid: Any, end: Any) -> list[dict[str, int]]:
    """An arc sampled into points, start and end kept exact.

    The sweep is the one that passes through ``mid``, so the direction KiCad
    drew is preserved. A degenerate arc (no center) falls back to its three
    defining points rather than being dropped.
    """
    import math

    from kipy.geometry import arc_center, arc_radius

    first, last = _pt(start), _pt(end)
    center = arc_center(start, mid, end)
    if center is None:
        return [first, _pt(mid), last]
    radius = arc_radius(start, mid, end)
    if not radius:
        return [first, _pt(mid), last]
    cx, cy = float(center.x), float(center.y)

    def angle(p: Any) -> float:
        return math.atan2(float(p.y) - cy, float(p.x) - cx)

    a0 = angle(start)
    turn = math.tau
    to_mid = (angle(mid) - a0) % turn
    to_end = (angle(end) - a0) % turn
    if to_end == 0.0:
        sweep = turn if to_mid > 0 else -turn  # start == end: a full circle
    elif to_mid <= to_end:
        sweep = to_end
    else:
        sweep = to_end - turn
    steps = max(2, math.ceil(abs(sweep) / math.radians(_ARC_STEP_DEG)))
    points = [first]
    for i in range(1, steps):
        t = a0 + sweep * i / steps
        points.append(
            {"x": round(cx + radius * math.cos(t)), "y": round(cy + radius * math.sin(t))}
        )
    points.append(last)
    return points


def _circle_points(center: Any, radius: float) -> list[dict[str, int]]:
    """A circle sampled into a closed ring (the first point is not repeated)."""
    import math

    steps = max(8, math.ceil(360.0 / _ARC_STEP_DEG))
    cx, cy = float(center.x), float(center.y)
    return [
        {
            "x": round(cx + radius * math.cos(math.tau * i / steps)),
            "y": round(cy + radius * math.sin(math.tau * i / steps)),
        }
        for i in range(steps)
    ]


def _bezier_points(p0: Any, p1: Any, p2: Any, p3: Any) -> list[dict[str, int]]:
    """A cubic bezier sampled into points, start and end kept exact."""
    points = [_pt(p0)]
    for i in range(1, _BEZIER_STEPS):
        t = i / _BEZIER_STEPS
        u = 1 - t
        points.append(
            {
                "x": round(u**3 * p0.x + 3 * u * u * t * p1.x + 3 * u * t * t * p2.x + t**3 * p3.x),
                "y": round(u**3 * p0.y + 3 * u * u * t * p1.y + 3 * u * t * t * p2.y + t**3 * p3.y),
            }
        )
    points.append(_pt(p3))
    return points


def _polyline_points(line: Any) -> list[dict[str, int]]:
    """A kipy ``PolyLine``'s nodes as points; arc nodes are sampled."""
    points: list[dict[str, int]] = []
    for node in line:
        if node.has_point:
            piece = [_pt(node.point)]
        elif node.has_arc:
            a = node.arc
            piece = _arc_points(a.start, a.mid, a.end)
        else:  # pragma: no cover - kipy sets one of the two
            continue
        for p in piece:
            if not points or not _near(points[-1], p, 0):
                points.append(p)
    while len(points) > 1 and _near(points[0], points[-1]):
        points.pop()  # a polyline KiCad closed explicitly; rings do not repeat
    return points


def _rectangle_points(shape: Any) -> list[dict[str, int]]:
    """A rectangle's four corners, clockwise from the top left (corners may be unordered)."""
    xs = sorted((int(shape.top_left.x), int(shape.bottom_right.x)))
    ys = sorted((int(shape.top_left.y), int(shape.bottom_right.y)))
    return [
        {"x": xs[0], "y": ys[0]},
        {"x": xs[1], "y": ys[0]},
        {"x": xs[1], "y": ys[1]},
        {"x": xs[0], "y": ys[1]},
    ]


def _shape_id(shape: Any) -> str:
    with contextlib.suppress(Exception):
        return str(shape.id.value or "")
    return ""  # pragma: no cover - kipy always has an id field


def _loop(points: list[dict[str, int]], role: str, shapes: int) -> dict[str, Any]:
    return {
        "closed": True,
        "role": role,
        "points_nm": points,
        "source_shapes": shapes,
        "gap_nm": None,
    }


def _classify_shapes(edge: list[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split Edge.Cuts shapes into rings that close on their own and open pieces.

    A rectangle, a circle and each ring of a polygon are already closed, so they
    are loops in their own right; a polygon's holes stay separate loops rather
    than being merged into its outline. A segment, arc or bezier is an open
    piece for :func:`_chain_pieces`. An unclassifiable shape reaches
    :func:`_shape_extent`, which raises ``internal_error`` (``_edge_outline``
    has already done so; this keeps the two paths honest together).
    """
    from kipy.board_types import (
        BoardArc,
        BoardBezier,
        BoardCircle,
        BoardPolygon,
        BoardRectangle,
        BoardSegment,
    )

    rings: list[dict[str, Any]] = []
    pieces: list[dict[str, Any]] = []
    for shape in edge:
        kind = type(shape)
        if kind is BoardRectangle:
            rings.append(_loop(_rectangle_points(shape), "outline", 1))
        elif kind is BoardCircle:
            rings.append(_loop(_circle_points(shape.center, float(shape.radius())), "outline", 1))
        elif kind is BoardPolygon:
            for poly in shape.polygons:
                points = _polyline_points(poly.outline)
                if len(points) >= 3:
                    rings.append(_loop(points, "outline", 1))
                for hole in poly.holes:
                    hole_points = _polyline_points(hole)
                    if len(hole_points) >= 3:
                        rings.append(_loop(hole_points, "hole", 1))
        elif kind is BoardSegment:
            pieces.append({"points": [_pt(shape.start), _pt(shape.end)], "id": _shape_id(shape)})
        elif kind is BoardArc:
            pieces.append(
                {"points": _arc_points(shape.start, shape.mid, shape.end), "id": _shape_id(shape)}
            )
        elif kind is BoardBezier:
            pieces.append(
                {
                    "points": _bezier_points(
                        shape.start, shape.control1, shape.control2, shape.end
                    ),
                    "id": _shape_id(shape),
                }
            )
        else:
            _shape_extent(shape)  # raises internal_error, naming the shape
    return rings, pieces


def _piece_key(piece: dict[str, Any]) -> tuple[Any, ...]:
    """A deterministic order for chaining, from geometry then id.

    Direction-independent, so the same board gives the same loops whatever
    order KiCad lists the shapes in, and a junction where three or more
    endpoints coincide is resolved the same way on every call.
    """
    a = (piece["points"][0]["x"], piece["points"][0]["y"])
    b = (piece["points"][-1]["x"], piece["points"][-1]["y"])
    lo, hi = sorted((a, b))
    return (lo, hi, len(piece["points"]), piece["id"])


def _prepare_pieces(pieces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop zero-length pieces and exact duplicates, then order deterministically.

    A zero-length segment would otherwise end a chain at a point it cannot
    leave, and a shape drawn twice would double its loop.
    """
    kept: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for piece in sorted(pieces, key=_piece_key):
        points = piece["points"]
        if _near(points[0], points[-1]) and len(points) <= 2:
            continue  # zero-length: nothing to chain through
        fingerprint = tuple((p["x"], p["y"]) for p in points)
        if fingerprint in seen or tuple(reversed(fingerprint)) in seen:
            continue
        seen.add(fingerprint)
        kept.append(piece)
    return kept


def _chain_pieces(
    pieces: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Greedy endpoint chaining into closed loops and leftover open chains.

    Each chain grows from both ends so a piece in the middle of a run still
    completes, and candidates are tried in :func:`_piece_key` order so the
    result does not depend on the order KiCad listed the shapes. A chain whose
    ends meet within :data:`_CHAIN_TOLERANCE_NM` is a closed loop; one that
    does not is returned as it stands, with the gap that defeated it. Several
    closed loops are normal: a board with a cutout has two.
    """
    remaining = _prepare_pieces(pieces)
    loops: list[dict[str, Any]] = []
    open_chains: list[dict[str, Any]] = []
    while remaining:
        head = remaining.pop(0)
        points = list(head["points"])
        used = 1
        growing = True
        while growing and not (len(points) >= 4 and _near(points[0], points[-1])):
            growing = False
            for i, piece in enumerate(remaining):
                other = piece["points"]
                if _near(points[-1], other[0]):
                    points.extend(other[1:])
                elif _near(points[-1], other[-1]):
                    points.extend(reversed(other[:-1]))
                elif _near(points[0], other[-1]):
                    points[:0] = other[:-1]
                elif _near(points[0], other[0]):
                    points[:0] = list(reversed(other[1:]))
                else:
                    continue
                remaining.pop(i)
                used += 1
                growing = True
                break
        if len(points) >= 4 and _near(points[0], points[-1]):
            loops.append(_loop(points[:-1], "outline", used))
        else:
            open_chains.append(
                {
                    "closed": False,
                    "role": "outline",
                    "points_nm": points,
                    "source_shapes": used,
                    "gap_nm": _gap_nm(points[0], points[-1]),
                }
            )
    return loops, open_chains


def _origins(board: Any) -> tuple[dict[str, int] | None, dict[str, int] | None, str | None]:
    """KiCad's grid and drill/place origins (file frame, nm), or why they are absent.

    ``Board.get_origin`` is kipy 0.3.0+ and KiCad answers it on 10.0. Only an
    "KiCad does not implement this" reply is turned into data; a busy or timed
    out request is raised so ``session.py`` can retry and map it (C-3).
    """
    from kipy.errors import ApiError
    from kipy.proto.board.board_commands_pb2 import BoardOriginType
    from kipy.proto.common import ApiStatusCode

    try:
        grid = _pt(board.get_origin(BoardOriginType.BOT_GRID))
        drill = _pt(board.get_origin(BoardOriginType.BOT_DRILL))
    except ApiError as e:
        if e.code in (ApiStatusCode.AS_UNIMPLEMENTED, ApiStatusCode.AS_UNHANDLED):
            return (
                None,
                None,
                "This KiCad does not answer GetBoardOrigin, so the grid and drill "
                f"origins are unknown ({ApiStatusCode.Name(e.code)}).",
            )
        raise
    return grid, drill, None


def _outline(self: KipyBackend, ref: BoardRef) -> dict[str, Any]:
    board = self._board(ref)
    edge, extents = _edge_outline(board)
    rings, pieces = _classify_shapes(edge)
    chained, open_chains = _chain_pieces(pieces)
    grid, drill, reason = _origins(board)
    return {
        "shape_count": len(edge),
        "loops": rings + chained,
        "open_chains": open_chains,
        "outline_nm": extents,
        "grid_origin_nm": grid,
        "drill_origin_nm": drill,
        "origins_reason": reason,
        "chain_tolerance_nm": _CHAIN_TOLERANCE_NM,
        "arc_step_deg": _ARC_STEP_DEG,
    }
