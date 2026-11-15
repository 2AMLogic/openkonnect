"""LAY-07 ``layers_and_stackup``: the open board's layers and physical build-up.

Read live over IPC: the enabled layers with their names, and the stackup
(copper, dielectric, mask and silk rows) top to bottom, as currently set in
the editor -- unsaved changes included.
"""

from __future__ import annotations

from typing import Any

from ..session import Session
from .common import mm, require_board

DESCRIPTION = (
    "Describe the open board's layers and physical build-up, live, unsaved changes "
    "included. Reports the enabled layers (canonical and user names), the copper "
    "layer count, and the stackup from top to bottom: per row its kind (copper, "
    "dielectric, silkscreen, soldermask, solderpaste), board layer (canonical "
    "name; null for dielectric rows), user name, enabled flag, thickness and "
    "material; a dielectric row also carries its type (core or prepreg) and each "
    "sub-layer's material, epsilon_r, loss_tangent and thickness. Lengths are in "
    "millimetres. Anything KiCad does not report is null, never a guessed default."
)


def _mm_or_none(nm: float | None) -> float | None:
    return None if nm is None else mm(nm)


def describe_layers_and_stackup(data: dict[str, Any]) -> dict[str, Any]:
    """Turn the backend's layer/stackup data into the LAY-07 result (no IPC)."""
    stackup: list[dict[str, Any]] = []
    for row in data["stackup"]:
        dielectric = row["dielectric"]
        stackup.append(
            {
                "kind": row["kind"],
                "layer": row["layer"],
                "user_name": row["user_name"],
                "enabled": row["enabled"],
                "thickness_mm": _mm_or_none(row["thickness_nm"]),
                "material": row["material"],
                "dielectric": None
                if dielectric is None
                else {
                    "type": dielectric["type"],
                    "sub_layers": [
                        {
                            "material": sub["material"],
                            "epsilon_r": sub["epsilon_r"],
                            "loss_tangent": sub["loss_tangent"],
                            "thickness_mm": _mm_or_none(sub["thickness_nm"]),
                        }
                        for sub in dielectric["sub_layers"]
                    ],
                },
            }
        )
    known = [row["thickness_nm"] for row in data["stackup"] if row["thickness_nm"] is not None]
    total = mm(sum(known)) if known else None
    return {
        "source": "live_editor",
        "layers": [dict(layer) for layer in data["layers"]],
        "copper_layers": data["copper_layers"],
        "stackup": stackup,
        "total_thickness_mm": total,
    }


def layers_and_stackup(session: Session) -> dict[str, Any]:
    ref = require_board(session)
    data = session.call(lambda b: b.layers_and_stackup(ref))
    result = describe_layers_and_stackup(data)
    result["board"] = {"path": ref.path}
    return result
