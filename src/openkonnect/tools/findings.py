"""Locators from kicad-tools' shared, versioned finding format.

kicad-tools ships one finding model for every command that emits findings
(``kicad_tools.findings.Finding``; its docstring names this consumer as the
reason the format exists). Each emitter keeps its existing JSON keys and
additionally spreads ``{"schema_version": N, "findings": [...]}`` into its
``--format json`` payload, where each element is a ``Finding.to_dict()``:

* ``rule_id``, ``severity``, ``message`` and ``waived`` describe the finding;
* ``refs``, ``pads`` (``REF.PAD``) and ``nets`` name the items it concerns,
  typed by upstream, so nothing is guessed from free text here;
* ``point`` (``[x, y]``) and ``bbox`` (``[x0, y0, x1, y1]``) locate it, both
  millimetres in the KiCad *file* frame: the emitters convert before building
  the Finding (``kct net-status`` adds ``PCB.board_origin`` back to the
  board-relative pad positions). No frame is left unstated, so no board
  outline is needed to place one;
* ``layer`` names the layer, when the finding has one.

This module reads those explicit fields into :class:`Locator` values for
``show_findings`` to resolve. It does not guess: an element that is not a
version-1 finding (wrong ``schema_version``, missing fields) raises
:class:`Unrecognised`, which the caller turns into ``invalid_argument``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: The finding schema this reader implements. Upstream carries the same
#: number as ``kicad_tools.findings.SCHEMA_VERSION``; it bumps when a field
#: is renamed, removed or retyped, and this reader must learn the new shape
#: before payloads bearing the new version resolve again.
SUPPORTED_SCHEMA_VERSION = 1

#: The locator fields every versioned finding carries.
_LOCATOR_FIELDS = ("refs", "pads", "nets", "point", "bbox", "layer")


@dataclass(frozen=True)
class Locator:
    """One way a finding points at the board.

    ``kind``: ``ref`` (reference designator), ``pad`` (``REF.PAD``), ``net``,
    ``point`` (file-frame mm), or ``bbox`` (file-frame mm corners).
    """

    kind: str
    value: Any
    field: str
    frame: str | None = None
    layer: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"kind": self.kind, "field": self.field}
        if self.kind == "point":
            d["x_mm"], d["y_mm"] = self.value
            d["frame"] = self.frame
        elif self.kind == "bbox":
            d["x0_mm"], d["y0_mm"], d["x1_mm"], d["y1_mm"] = self.value
            d["frame"] = self.frame
        else:
            d["value"] = self.value
        if self.layer:
            d["layer"] = self.layer
        return d


class Unrecognised(Exception):
    """Why an element is not a finding this reader can use."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def as_dict(finding: Any) -> dict[str, Any] | None:
    """The finding's JSON form, accepting a live model with ``to_dict()``."""
    if isinstance(finding, dict):
        return finding
    to_dict = getattr(finding, "to_dict", None)
    if callable(to_dict):
        d = to_dict()
        if isinstance(d, dict):
            return d
    return None


def is_finding(finding: Any) -> bool:
    """Whether ``finding`` is one versioned finding (dict or model object)."""
    d = as_dict(finding)
    return (
        d is not None
        and d.get("schema_version") == SUPPORTED_SCHEMA_VERSION
        and all(f in d for f in _LOCATOR_FIELDS)
    )


def _seq(v: Any) -> list[Any]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


def _xy(v: Any) -> tuple[float, float] | None:
    """A point given as ``[x, y]``."""
    if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(c, (int, float)) for c in v):
        return float(v[0]), float(v[1])
    return None


def _bbox(v: Any) -> tuple[float, float, float, float] | None:
    """A box given as ``[x0, y0, x1, y1]``."""
    if isinstance(v, (list, tuple)) and len(v) == 4 and all(isinstance(c, (int, float)) for c in v):
        return tuple(float(c) for c in v)
    return None


def _strs(v: Any) -> list[str]:
    return [s for s in _seq(v) if isinstance(s, str) and s]


def read(finding: Any) -> list[Locator]:
    """One finding's locators, read from its explicit versioned fields.

    Raises :class:`Unrecognised` when the element is not a versioned finding:
    not a dict or model object, a ``schema_version`` this reader does not
    know, or missing locator fields.
    """
    d = as_dict(finding)
    if d is None:
        raise Unrecognised(
            f"not a JSON object or a model with to_dict() ({type(finding).__name__})"
        )
    observed = d.get("schema_version")
    if observed != SUPPORTED_SCHEMA_VERSION:
        raise Unrecognised(
            f"its schema_version is {observed!r}, this reader supports {SUPPORTED_SCHEMA_VERSION}"
        )
    missing = [f for f in _LOCATOR_FIELDS if f not in d]
    if missing:
        raise Unrecognised(f"its fields are missing: {', '.join(missing)}")
    layer = d["layer"] if isinstance(d["layer"], str) and d["layer"] else None
    out = [Locator("ref", s, "refs") for s in _strs(d["refs"])]
    out += [Locator("pad", s, "pads") for s in _strs(d["pads"])]
    out += [Locator("net", s, "nets") for s in _strs(d["nets"])]
    p = _xy(d["point"])
    if p is not None:
        out.append(Locator("point", p, "point", "file", layer))
    box = _bbox(d["bbox"])
    if box is not None:
        out.append(Locator("bbox", box, "bbox", "file", layer))
    return out
