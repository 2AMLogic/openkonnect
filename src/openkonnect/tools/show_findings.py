"""SES-09 ``show_findings``: point the user at the items kicad-tools findings concern.

Takes the versioned finding payload a finding-emitting ``kct`` command prints
with ``--format json`` (``kct check``, ``kct drc``, ``kct detect-mistakes``,
``kct net-status``, or the ``measure_clearance`` MCP tool): every emitter
spreads ``schema_version`` and a ``findings`` array into its JSON. That array
alone, a bare findings array, or one finding is accepted too. Each finding's
locators (references, ``REF.PAD`` names, net names, a point or bbox in the
file frame, a layer) are read from the finding's explicit fields
(``findings.py``), resolved against the live board, and the resolved items
are **selected** in the PCB editor (replace semantics).

KiCad 10 gap (spec limitation 7, SES-14, SES-15): kicad-python 0.8 documents
no request to zoom the view or highlight a net. The only route is
``KiCad.run_action``, which kicad-python marks unstable and not for general
use, and this tool does not call it. ``mode`` ``highlight``/``both`` and
``zoom: true`` therefore report ``unsupported_on_this_kicad`` for that part,
while the selection still happens; the editor draws the selection
highlighted.

How locators become items:

* a reference designator selects the footprint; ``REF.PAD`` selects the pad;
* a net selects its copper (tracks, arcs, vias, zones). A net with no copper
  is represented by the finding's own pads on it if it names any, else by all
  of the net's pads;
* when the finding resolved no name and carries a ``bbox``, the smallest
  item(s) whose bounding box overlaps that region are selected;
* otherwise a point is used only when the finding resolved no name: it
  selects the smallest item whose bounding box contains it (on the finding's
  layer, when given). Otherwise it is checked against the resolved items.

A point is optional. A finding whose point is null, missing or not an
``(x, y)`` pair is resolved from its references, pads and nets alone, or
from the region its ``bbox`` spans when no name resolved; its result has
``has_point: false`` and a ``point_note`` saying which, and whether a
bbox was only checked against the resolved items (or matched none), and
no coordinate is invented. A finding with no point, no bbox and no name
is reported ``unresolved`` with a ``no_locator`` reason and no
``point_note``; the rest of the payload is still resolved and shown.
"""

from __future__ import annotations

import hashlib
import json
import weakref
from typing import Any

from .. import errors as E
from ..backend import NM_PER_MM
from ..session import Session, file_state, read_board_text, text_sha256
from . import findings as F
from .common import pad_name, require_board
from .get_selection import describe_selection

DESCRIPTION = (
    "Point the user at kicad-tools findings in KiCad's PCB editor: resolve each finding's "
    "locators (reference designators, 'REF.PAD', net names, a point or region) against the "
    "live board and select the affected items (replacing the selection). findings is the "
    "versioned JSON a finding-emitting kct command printed with --format json (kct check, "
    "kct drc, kct detect-mistakes, kct net-status, or the measure_clearance MCP tool), its "
    "findings array, a bare findings array, or one finding. mode: select | highlight | both; "
    "focus: 'all', a finding index, 'next' or 'previous'; zoom: true/false. On KiCad 10 "
    "there is no documented zoom or net-highlight API, so highlight and zoom report "
    "unsupported_on_this_kicad while the selection still happens. Per finding the result "
    "lists the resolved items, the locators that did not resolve with a reason, and the "
    "action taken, plus totals. Findings whose pcb_sha256 no longer matches the board are "
    "still resolved and flagged possibly_stale. A finding's point is optional: one with a "
    "null or missing point is resolved from its names or its bbox region and reported "
    "with has_point false."
)

MODES = ("select", "highlight", "both")
#: Points within this distance of an item's box count as inside it (SES-10: 1 um).
TOLERANCE_NM = 1_000

_CURSORS: weakref.WeakKeyDictionary[Session, dict[str, Any]] = weakref.WeakKeyDictionary()


# -- input --------------------------------------------------------------------


def _fingerprint(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, default=lambda o: repr(o))
    return hashlib.sha256(raw.encode()).hexdigest()


def split_input(findings: Any) -> tuple[list[Any], str | None]:
    """The finding list and the input board hash (``pcb_sha256``), if any.

    ``findings`` may be the whole JSON object a kct command printed (its
    ``findings`` array is taken; the command's other keys are left alone),
    a bare findings array, one finding, or that JSON as a string. An
    envelope claiming a ``schema_version`` this tool does not read is
    refused with the observed and supported versions.
    """
    if isinstance(findings, str):
        try:
            findings = json.loads(findings)
        except ValueError as e:
            raise E.invalid_argument(f"findings is not valid JSON: {e}") from None
    input_hash = None
    if isinstance(findings, dict):
        input_hash = findings.get("pcb_sha256")
        input_hash = input_hash if isinstance(input_hash, str) and input_hash else None
        version = findings.get("schema_version")
        if version is not None and version != F.SUPPORTED_SCHEMA_VERSION:
            raise E.invalid_argument(
                f"findings carries schema_version {version!r}; this tool reads schema "
                f"version {F.SUPPORTED_SCHEMA_VERSION}.",
                observed_schema_version=version,
                supported_schema_version=F.SUPPORTED_SCHEMA_VERSION,
            )
        array = findings.get("findings")
        if isinstance(array, list):
            return array, input_hash
        if F.is_finding(findings):
            return [findings], input_hash
        raise E.invalid_argument(
            "findings is a JSON object whose 'findings' key is not an array of versioned "
            "kicad-tools findings. Pass the output of a supported kct command printed with "
            "--format json, a findings array, or one finding.",
            observed_schema_version=version,
            supported_schema_version=F.SUPPORTED_SCHEMA_VERSION,
        )
    if isinstance(findings, list):
        return findings, None
    if F.is_finding(findings):
        return [findings], None
    raise E.invalid_argument(
        "findings must be a kct --format json object, a findings array, or one versioned "
        "kicad-tools finding.",
        got=type(findings).__name__,
    )


def extract_all(items: list[Any]) -> list[list[F.Locator]]:
    """Each finding's locators; an empty list for a finding that carries none.

    A versioned finding with no reference, pad or net and a null, missing or
    malformed point and bbox is not a payload error: it degrades to an empty
    locator list, and that one finding is reported ``unresolved``
    (``no_locator``) while the rest of the payload is still resolved and
    shown. Only an element that is not a versioned finding at all is refused.
    """
    out = []
    for i, finding in enumerate(items):
        try:
            locs = F.read(finding)
        except F.Unrecognised as u:
            d = F.as_dict(finding)
            raise E.invalid_argument(
                f"Finding {i} is not a versioned kicad-tools finding: {u.reason}.",
                index=i,
                observed_schema_version=d.get("schema_version") if d else None,
                supported_schema_version=F.SUPPORTED_SCHEMA_VERSION,
            ) from None
        out.append(locs)
    return out


def parse_focus(focus: Any, n: int) -> str | int:
    if isinstance(focus, bool):
        raise E.invalid_argument("focus must be 'all', 'next', 'previous' or an index.")
    if isinstance(focus, str) and focus.strip().lstrip("-").isdigit():
        focus = int(focus.strip())
    if isinstance(focus, int):
        if not 0 <= focus < n:
            raise E.invalid_argument(
                f"focus index {focus} is out of range for {n} finding(s).", index=focus, count=n
            )
        return focus
    if focus in ("all", "next", "previous"):
        return focus
    raise E.invalid_argument(
        "focus must be 'all', 'next', 'previous' or a finding index.", got=focus
    )


# -- resolution ---------------------------------------------------------------


class BoardIndex:
    """The live board's names, for resolving locators (built from ``board_index``)."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.refs: dict[str, list[dict[str, Any]]] = {}
        self.pads: dict[str, list[dict[str, Any]]] = {}
        self.pads_by_net: dict[str, list[dict[str, Any]]] = {}
        self.copper_by_net: dict[str, list[dict[str, Any]]] = {}
        self.nets = set(data.get("nets") or [])
        for fp in data.get("footprints") or []:
            self.refs.setdefault(fp["ref"], []).append(fp)
            for pad in fp.get("pads") or []:
                name = pad_name(pad)
                if name:
                    self.pads.setdefault(name, []).append(pad)
                if pad.get("net"):
                    self.pads_by_net.setdefault(pad["net"], []).append(pad)
                    self.nets.add(pad["net"])
        self.copper = list(data.get("copper") or [])
        for item in self.copper:
            if item.get("net"):
                self.copper_by_net.setdefault(item["net"], []).append(item)
                self.nets.add(item["net"])

    def all_items(self) -> list[dict[str, Any]]:
        out = []
        for fps in self.refs.values():
            for fp in fps:
                out.append(fp)
                out.extend(fp.get("pads") or [])
        return out + self.copper


def _candidates(index: BoardIndex, kind: str, name: str) -> list[tuple[str, Any]]:
    c: list[tuple[str, Any]] = []
    if kind == "ref" and name in index.refs:
        c.extend(("footprint", fp) for fp in index.refs[name])
    if kind == "pad" and name in index.pads:
        c.extend(("pad", p) for p in index.pads[name])
    if kind == "net" and name in index.nets:
        c.append(("net", name))
    return c


def _miss_reason(index: BoardIndex, kind: str, name: str) -> tuple[str, str]:
    if kind == "ref":
        return "ref_not_on_board", f"No footprint {name} on the board."
    if kind == "pad":
        ref, _, num = name.partition(".")
        if ref in index.refs:
            return "pad_not_on_footprint", f"Footprint {ref} has no pad {num}."
        return "ref_not_on_board", f"No footprint {ref} on the board."
    return "net_not_on_board", f"No net {name} on the board (renamed or removed?)."


def _area(box: dict[str, int]) -> int:
    return (box["x1_nm"] - box["x0_nm"]) * (box["y1_nm"] - box["y0_nm"])


def _on_layer(item: dict[str, Any], layer: str | None) -> bool:
    if not layer:
        return True
    if item.get("layers"):
        return layer in item["layers"]
    return item.get("layer") in (None, layer)


def _nm(v: float) -> int:
    return round(v * NM_PER_MM)


def _intersects(box: dict[str, int], x0: int, y0: int, x1: int, y1: int) -> bool:
    t = TOLERANCE_NM
    return (
        box["x0_nm"] - t <= x1
        and x0 <= box["x1_nm"] + t
        and box["y0_nm"] - t <= y1
        and y0 <= box["y1_nm"] + t
    )


def _smallest_hits(
    index: BoardIndex, x0: int, y0: int, x1: int, y1: int, layer: str | None
) -> list[dict[str, Any]]:
    """Every item tied for the smallest bounding box meeting the bounds.

    The one spatial selection rule for both locator shapes: a point is the
    bounds (x, y, x, y), which the intersection predicate treats exactly as
    containment of the point. Tied items keep board order.
    """
    hits = [
        i
        for i in index.all_items()
        if i.get("bbox_nm") and _intersects(i["bbox_nm"], x0, y0, x1, y1) and _on_layer(i, layer)
    ]
    if not hits:
        return []
    smallest = min(_area(i["bbox_nm"]) for i in hits)
    return [i for i in hits if _area(i["bbox_nm"]) == smallest]


NO_POINT_NOTE = (
    "The finding carries no point: its point is null or missing (optional in the "
    "versioned finding format). It was resolved from its references, pads and nets "
    "only, and no location was checked."
)

NO_POINT_BBOX_NOTE = (
    "The finding carries no point: its point is null or missing (optional in the "
    "versioned finding format). It was resolved from the region (bbox) the finding "
    "carries, and no single point was checked."
)

NO_POINT_NAMES_BBOX_NOTE = (
    "The finding carries no point: its point is null or missing (optional in the "
    "versioned finding format). It was resolved from its references, pads and nets "
    "only, and no single point was checked; the region (bbox) the finding carries "
    "was checked against those items."
)

NO_POINT_BBOX_EMPTY_NOTE = (
    "The finding carries no point: its point is null or missing (optional in the "
    "versioned finding format). Nothing was resolved from references, pads or "
    "nets; the region (bbox) the finding carries was checked, but no item's "
    "bounding box overlaps it."
)

NO_LOCATOR_MESSAGE = (
    "The finding carries no locator: no reference, pad or net, and its point and "
    "bbox are null, missing or not well formed. Nothing could be selected for it; "
    "the other findings were still resolved."
)

_POINT_MESSAGES = {
    "empty_area": "No item's bounding box contains this point.",
}

BBOX_EMPTY_MESSAGE = "No item's bounding box overlaps this finding's region."


def _point_note(points: list[dict[str, Any]], resolved: dict[str, Any]) -> str | None:
    """A no-point finding's note, from what resolved it and what was checked.

    States what resolved the finding -- its names, the bbox region, or
    nothing -- and what was checked; ``None`` when nothing resolved and
    nothing was checked, where the unresolved report already says why.
    """
    if any(
        p.get("kind") == "bbox" and p.get("used_for") == "resolve" and p.get("resolved_to")
        for p in points
    ):
        return NO_POINT_BBOX_NOTE
    names_resolved = bool(resolved.get("item_count")) or bool(resolved.get("nets"))
    # Past the resolved-by-bbox return above, every bbox entry was either
    # only checked against the resolved items, or probed and matched nothing.
    bbox_seen = any(p.get("kind") == "bbox" for p in points)
    if names_resolved:
        return NO_POINT_NAMES_BBOX_NOTE if bbox_seen else NO_POINT_NOTE
    return NO_POINT_BBOX_EMPTY_NOTE if bbox_seen else None


def resolve(index: BoardIndex, locs: list[F.Locator]) -> dict[str, Any]:
    """Resolve one finding's locators. Pure: no IPC."""
    chosen: dict[str, dict[str, Any]] = {}
    footprints: set[str] = set()
    pads: set[str] = set()
    nets: set[str] = set()
    unresolved: list[dict[str, Any]] = []

    def take(item: dict[str, Any]) -> None:
        if item.get("id"):
            chosen.setdefault(item["id"], item)

    def record(items: list[dict[str, Any]]) -> int:
        for i in items:
            take(i)
            if i["kind"] == "footprint":
                footprints.add(i["ref"])
            elif i["kind"] == "pad" and pad_name(i):
                pads.add(pad_name(i))
            elif i.get("net"):
                nets.add(i["net"])
        return len(items)

    for loc in locs:
        if loc.kind in ("point", "bbox"):
            continue
        cands = _candidates(index, loc.kind, loc.value)
        if not cands:
            reason, msg = _miss_reason(index, loc.kind, loc.value)
            unresolved.append({"locator": loc.to_dict(), "reason": reason, "message": msg})
            continue
        if len(cands) > 1:
            unresolved.append(
                {
                    "locator": loc.to_dict(),
                    "reason": "ambiguous_reference",
                    "message": f"{loc.value!r} matches more than one item: "
                    + ", ".join(sorted({k for k, _ in cands}))
                    + ". Nothing was guessed.",
                }
            )
            continue
        kind, target = cands[0]
        if kind == "footprint":
            footprints.add(target["ref"])
            take(target)
        elif kind == "pad":
            pads.add(loc.value)
            take(target)
        else:
            nets.add(target)

    net_basis: dict[str, str] = {}
    for net in sorted(nets):
        copper = index.copper_by_net.get(net, [])
        if copper:
            for item in copper:
                take(item)
            net_basis[net] = f"copper ({len(copper)} item(s))"
            continue
        own = [p for p in index.pads_by_net.get(net, []) if pad_name(p) in pads]
        if own:
            net_basis[net] = "no copper; represented by the finding's pads on it"
            continue
        all_pads = index.pads_by_net.get(net, [])
        for p in all_pads:
            take(p)
            name = pad_name(p)
            if name:
                pads.add(name)
        net_basis[net] = f"no copper; its {len(all_pads)} pad(s)"

    named = bool(chosen) or bool(nets)
    points: list[dict[str, Any]] = []

    # Upstream's bbox spans the locations a finding concerns; when it is
    # supplied and no name resolved, it resolves instead of the point. When
    # names did resolve, it only checks that the region overlaps what the
    # names selected -- it never widens the selection or fails the finding.
    for loc in locs:
        if loc.kind != "bbox":
            continue
        entry = loc.to_dict()
        x0, y0, x1, y1 = (_nm(c) for c in loc.value)
        entry["file_mm"] = {
            "x0_mm": x0 / NM_PER_MM,
            "y0_mm": y0 / NM_PER_MM,
            "x1_mm": x1 / NM_PER_MM,
            "y1_mm": y1 / NM_PER_MM,
        }
        if named:
            entry["used_for"] = "check"
            entry["within_resolved_items"] = any(
                i.get("bbox_nm") and _intersects(i["bbox_nm"], x0, y0, x1, y1)
                for i in chosen.values()
            )
        else:
            entry["used_for"] = "resolve"
            selected = _smallest_hits(index, x0, y0, x1, y1, loc.layer)
            if not selected:
                entry["reason"] = "empty_area"
                unresolved.append(
                    {
                        "locator": loc.to_dict(),
                        "reason": "empty_area",
                        "message": BBOX_EMPTY_MESSAGE,
                    }
                )
            else:
                entry["resolved_to"] = record(selected)
                named = True
        points.append(entry)

    for loc in locs:
        if loc.kind != "point":
            continue
        entry = loc.to_dict()
        x, y = (_nm(c) for c in loc.value)
        entry["file_mm"] = {"x_mm": x / NM_PER_MM, "y_mm": y / NM_PER_MM}
        if named:
            entry["used_for"] = "check"
            entry["within_resolved_items"] = any(
                i.get("bbox_nm") and _intersects(i["bbox_nm"], x, y, x, y) for i in chosen.values()
            )
        else:
            entry["used_for"] = "resolve"
            selected = _smallest_hits(index, x, y, x, y, loc.layer)
            if not selected:
                entry["reason"] = "empty_area"
                unresolved.append(
                    {
                        "locator": loc.to_dict(),
                        "reason": "empty_area",
                        "message": _POINT_MESSAGES["empty_area"],
                    }
                )
            else:
                entry["resolved_to"] = record(selected)
        points.append(entry)

    if not chosen and not nets:
        status = "unresolved"
    elif unresolved:
        status = "partial"
    else:
        status = "resolved"
    return {
        "status": status,
        "ids": list(chosen),
        "resolved": {
            "footprints": sorted(footprints),
            "pads": sorted(pads),
            "nets": sorted(nets),
            "item_count": len(chosen),
        },
        "net_basis": net_basis,
        "unresolved": unresolved,
        "points": points,
    }


# -- staleness ----------------------------------------------------------------


def staleness(session: Session, ref: Any, input_hash: str | None) -> dict[str, Any]:
    if input_hash is None:
        return {
            "possibly_stale": False,
            "input_hash": None,
            "basis": "The findings carry no input hash (pcb_sha256), so staleness is not known.",
        }
    # A board that has never been saved has no file to compare against.
    disk = file_state(ref.path).get("sha256") if ref.path is not None else None
    if disk == input_hash:
        return {
            "possibly_stale": False,
            "input_hash": input_hash,
            "matches": "saved_file",
            "basis": "The findings were computed from the board file as it is now on disk. "
            "Unsaved editor changes since then cannot be ruled out on KiCad 10.",
        }
    text = read_board_text(session, ref)
    if text is not None and text_sha256(text) == input_hash:
        return {
            "possibly_stale": False,
            "input_hash": input_hash,
            "matches": "editor",
            "basis": "The findings were computed from the board as the editor holds it now.",
        }
    return {
        "possibly_stale": True,
        "input_hash": input_hash,
        "matches": None,
        "saved_file_sha256": disk,
        "basis": "The input hash matches neither the saved board file nor the editor's "
        "board; the board changed after the findings were computed. They were still "
        "resolved against the current board.",
    }


# -- the tool -----------------------------------------------------------------


def _focus_indices(
    session: Session, focus: str | int, n: int, fingerprint: str
) -> tuple[list[int], dict[str, Any]]:
    if focus == "all":
        return list(range(n)), {"requested": "all"}
    if isinstance(focus, int):
        _CURSORS[session] = {"fingerprint": fingerprint, "index": focus}
        return [focus], {"requested": focus, "index": focus}
    cur = _CURSORS.get(session)
    same = cur is not None and cur["fingerprint"] == fingerprint
    wrapped = False
    if focus == "next":
        i = cur["index"] + 1 if same else 0
        if i >= n:
            i, wrapped = 0, True
    else:
        i = cur["index"] - 1 if same else n - 1
        if i < 0:
            i, wrapped = n - 1, True
    _CURSORS[session] = {"fingerprint": fingerprint, "index": i}
    return [i], {
        "requested": focus,
        "index": i,
        "wrapped": wrapped,
        "cursor": "continued" if same else "started (new or changed finding list)",
    }


def _unsupported(mode: str, zoom: bool) -> list[dict[str, Any]]:
    out = []
    if mode in ("highlight", "both"):
        err = E.unsupported(
            "KiCad 10's IPC API documents no net-highlight request (kicad-python 0.8 has "
            "none; the unstable run_action route is not used). The items were selected "
            "instead, and the editor draws the selection highlighted.",
            spec="SES-15",
        )
        out.append({"part": "highlight", "error": err.to_dict(), "fallback": "selected"})
    if zoom:
        err = E.unsupported(
            "KiCad 10's IPC API documents no zoom or view request (kicad-python 0.8 has "
            "none; the unstable run_action route is not used). The view was not moved.",
            spec="SES-14",
        )
        out.append({"part": "zoom", "error": err.to_dict(), "fallback": "none"})
    return out


def show_findings(
    session: Session,
    findings: Any,
    mode: str = "select",
    focus: Any = "all",
    zoom: bool = False,
) -> dict[str, Any]:
    if mode not in MODES:
        raise E.invalid_argument(f"mode must be one of {', '.join(MODES)}.", got=mode)
    items, input_hash = split_input(findings)
    if not items:
        raise E.invalid_argument("findings is empty: there is nothing to show.")
    extracted = extract_all(items)
    focus_v = parse_focus(focus, len(items))

    ref = require_board(session)
    index = BoardIndex(session.call(lambda b: b.board_index(ref)))
    stale = staleness(session, ref, input_hash)
    focused, focus_info = _focus_indices(session, focus_v, len(items), _fingerprint(items))

    per: list[dict[str, Any]] = []
    ids: list[str] = []
    for i, locs in enumerate(extracted):
        d = F.as_dict(items[i]) or {}
        r = resolve(index, locs)
        if not locs:
            r["unresolved"] = [
                {"locator": None, "reason": "no_locator", "message": NO_LOCATOR_MESSAGE}
            ]
        in_focus = i in focused
        if in_focus:
            ids.extend(x for x in r["ids"] if x not in ids)
        has_point = any(p.get("kind") == "point" for p in r["points"])
        point_note = None if has_point else _point_note(r["points"], r["resolved"])
        per.append(
            {
                "index": i,
                "rule_id": d.get("rule_id"),
                "severity": d.get("severity"),
                "waived": bool(d.get("waived", False)),
                "status": r["status"],
                "resolved": r["resolved"],
                "net_basis": r["net_basis"],
                "unresolved": r["unresolved"],
                "points": r["points"],
                "has_point": has_point,
                **({} if has_point or point_note is None else {"point_note": point_note}),
                "possibly_stale": stale["possibly_stale"],
                "_ids": r["ids"],
                "_focus": in_focus,
            }
        )

    selection: dict[str, Any] | None = None
    if ids:
        session.call(lambda b: b.set_selection(ref, ids))
        after = session.call(lambda b: b.selection(ref, None))
        observed = {i["id"] for i in after["items"] if i.get("id")}
        described = describe_selection(after)
        selection = {
            "footprints": described["footprints"],
            "pads": described["pads"],
            "nets": described["nets"],
            "count": described["count"],
            "bbox": described["bbox"],
            "frame_offset": described["frame_offset"],
            "requested_items": len(ids),
            "missing_after_select": [x for x in ids if x not in observed],
        }
    for f in per:
        fids, in_focus = f.pop("_ids"), f.pop("_focus")
        if not in_focus:
            f["action"] = "not_focused"
        elif not fids:
            f["action"] = "nothing_to_select"
        else:
            f["action"] = "selected"

    totals = {
        "findings": len(per),
        "resolved": sum(f["status"] == "resolved" for f in per),
        "partial": sum(f["status"] == "partial" for f in per),
        "unresolved": sum(f["status"] == "unresolved" for f in per),
        "focused": len(focused),
        "selected_items": len(ids),
        "possibly_stale": len(per) if stale["possibly_stale"] else 0,
    }
    return {
        "source": "live_editor",
        "board": {"path": ref.path},
        "mode": mode,
        "zoom": bool(zoom),
        "focus": focus_info,
        "selection_changed": bool(ids),
        "selection_note": (
            "The editor selection was replaced with the focused findings' items."
            if ids
            else "Nothing resolved in the focused findings, so the selection was left as it was."
        ),
        "selection": selection,
        "unsupported": _unsupported(mode, bool(zoom)),
        "staleness": stale,
        "findings": per,
        "totals": totals,
    }
