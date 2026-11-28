"""The narrow IPC surface openkonnect needs, and its kicad-python implementation.

The session layer (``session.py``) talks only to :class:`IpcBackend`. The real
implementation, :class:`KipyBackend`, wraps KiCad's official Python bindings
(``kicad-python``, imported as ``kipy``; MIT, see ``docs/ipc-client.md``).
Tests substitute a fake backend that raises the same ``kipy`` exception types.

Backends raise ``kipy.errors.ConnectionError`` and ``kipy.errors.ApiError``
unchanged; ``session.py`` maps them onto C-3 codes.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from .backend_outline import _edge_outline, _outline
from .backend_readers import (
    _board_index,
    _copper,
    _layers_and_stackup,
    _nets,
    _pads,
    _selection,
    _set_selection,
    _zones,
)

NM_PER_MM = 1_000_000


@dataclass
class BoardRef:
    """One board open in the PCB editor, as KiCad reports it.

    ``path`` is None for a board that has never been saved (a new board, with
    no file and no project). Tools that need the file refuse it with
    ``no_document``; see :func:`openkonnect.tools.common.require_saved`.
    """

    path: str | None
    file_name: str
    project_dir: str | None = None
    project_name: str | None = None
    handle: Any = field(default=None, repr=False, compare=False)


class IpcBackend(Protocol):
    """What the session layer needs from an IPC client."""

    socket: str

    def ping(self) -> None: ...

    def kicad_version(self) -> str: ...

    def instance_token(self) -> str | None: ...

    def open_boards(self) -> list[BoardRef]: ...

    def board_data(self, ref: BoardRef) -> dict[str, Any]: ...

    def board_text(self, ref: BoardRef) -> str:
        """The open board as KiCad native-format text, including unsaved changes."""
        ...

    def save_board(self, ref: BoardRef) -> None:
        """Ask the PCB editor to save the board to its own file."""
        ...

    def revert_board(self, ref: BoardRef) -> None:
        """Ask the PCB editor to reload the board from its file, dropping unsaved changes."""
        ...

    def selection(self, ref: BoardRef, kinds: list[str] | None = None) -> dict[str, Any]:
        """The editor's current selection (#5), as plain data.

        Returns ``{"items": [item, ...], "undescribed": {type_name: count},
        "outline_nm": box | None}``. Each item is a dict with ``id``, ``kind``
        (one of :data:`SELECTION_KINDS`), ``ref``/``pad`` where the item has a
        file-level name, ``net``, ``layer``, ``position_nm`` and ``bbox_nm``
        (file frame, nm). Items the client cannot decode are counted in
        ``undescribed``, never dropped.
        """
        ...

    def board_index(self, ref: BoardRef) -> dict[str, Any]:
        """Footprints (with their pads), copper items and nets, for resolving references.

        Returns ``{"footprints": [...], "copper": [...], "nets": [...],
        "outline_nm": box | None}``; items use the same dict form as
        :meth:`selection`. A footprint item also carries ``value``, ``library_id``,
        ``orientation_deg``, ``locked``, ``attributes`` and ``fields`` (LAY-02).
        """
        ...

    def nets(self, ref: BoardRef, include_items: bool = False) -> dict[str, Any]:
        """The board's nets (LAY-04), as plain data.

        Returns ``{"nets": [{"name", "net_class", "pad_count", "items"?}, ...]}``.
        ``net_class`` is the name of the net's class (None if KiCad reports none).
        With ``include_items``, ``items`` lists the pads (``kind: "pad"``, ``ref``,
        ``pad``, ``id``) and copper (``kind``, ``id``) on the net.
        """
        ...

    def zones(self, ref: BoardRef) -> dict[str, Any]:
        """Copper zones and rule areas (LAY-06), as plain data.

        Returns ``{"zones": [zone, ...]}``. Each zone has ``id`` (session ID), ``kind``
        (``"copper"``, ``"rule_area"`` or another zone type name), ``name`` (None if
        unnamed), ``net`` (None for a rule area or an unconnected zone), ``layers``
        (canonical names), ``priority``, ``pad_connection`` (``"inherited"``, ``"none"``,
        ``"thermal"``, ``"full"``, ``"pth_thermal"``; None for a rule area), ``filled``,
        ``locked`` and ``keepouts`` (None for a copper zone; for a rule area a dict of
        ``copper``, ``vias``, ``tracks``, ``pads``, ``footprints`` booleans).
        """
        ...

    def layers_and_stackup(self, ref: BoardRef) -> dict[str, Any]:
        """Enabled layers and the physical stackup (LAY-07), as plain data.

        Returns ``{"layers": [layer, ...], "copper_layers": int, "stackup": [row, ...]}``.
        ``layers`` is the board's enabled layer set (copper and non-copper), each
        entry ``{"name": canonical, "user_name": ...}``. Each stackup row is the
        corresponding build-up entry, in order from top to bottom: ``kind``
        (``"copper"``, ``"dielectric"``, ``"silkscreen"``, ``"soldermask"``,
        ``"solderpaste"``, ...), ``layer`` (canonical name; None for a row on no
        board layer, e.g. a dielectric), ``user_name`` (None when unset), ``enabled``,
        ``thickness_nm`` (None when KiCad reports no thickness), ``material``
        (None when unset) and ``dielectric`` — None when KiCad reports no detail
        submessage, else ``{"type": "core"/"prepreg"/..., "sub_layers": [sub, ...]}``
        with each sub-layer ``{"material", "epsilon_r", "loss_tangent", "thickness_nm"}``
        (thickness None when unreported; the other fields None when unset). Lengths
        in nanometres.
        """
        ...

    def outline(self, ref: BoardRef) -> dict[str, Any]:
        """The board outline as polygons, with the extents and KiCad's origins (LAY-08).

        Returns ``{"shape_count", "loops", "open_chains", "outline_nm",
        "grid_origin_nm", "drill_origin_nm", "origins_reason",
        "chain_tolerance_nm", "arc_step_deg"}``.

        ``loops`` are the closed rings, ``open_chains`` the pieces that did not
        close; each is ``{"closed", "role" ("outline" or "hole"),
        "points_nm": [{"x", "y"}, ...], "source_shapes", "gap_nm"}``. A closed
        ring does not repeat its first point and has ``gap_nm`` None; an open
        chain's ``gap_nm`` is the distance from its last point back to its
        first, so a caller can say what defeated the chain. Arcs, beziers and
        circles are sampled into segments at ``arc_step_deg``, so a loop that
        contains one is an approximation. Chaining joins endpoints within
        ``chain_tolerance_nm``.

        ``outline_nm`` is the extents box :meth:`board_data` reports, on the
        same basis (geometry, not stroke; board-level shapes only). The origins
        are ``{"x", "y"}`` in the ``file`` frame, or None with
        ``origins_reason`` set when this KiCad has no origin query.
        """
        ...

    def copper(self, ref: BoardRef) -> dict[str, Any]:
        """Tracks, arcs and vias (LAY-05), as plain data.

        Returns ``{"copper": [item, ...], "outline_nm": box | None}``. Every item has
        ``id`` (session ID), ``kind`` (``"track"``, ``"arc"`` or ``"via"``), ``net``
        (None when unconnected or unreadable) and ``locked``. A track or arc also has
        ``layer`` (canonical name), ``width_nm``, ``start_nm``, ``end_nm`` and
        ``length_nm`` (an arc's true arc length, not its chord), and an arc has
        ``mid_nm``. A via instead has ``layers`` (the canonical names of its drill span),
        ``position_nm``, ``diameter_nm`` (None if KiCad reports no copper layer),
        ``drill_nm`` and ``via_type`` (``"through"``, ``"blind_buried"``,
        ``"micro"``, ``"blind"`` or ``"buried"`` -- the last two new in KiCad 10
        -- or ``"unknown"`` for an unrecognised type);
        a via has no length. ``outline_nm`` is the C-1 board-frame origin box.
        """
        ...

    def pads(self, ref: BoardRef, refs: list[str] | None = None) -> dict[str, Any]:
        """Pads with their geometry (LAY-03), as plain data.

        Returns ``{"footprints": [ref, ...], "pads": [pad, ...], "outline_nm": box | None}``.
        ``footprints`` lists every footprint reference on the board. Each pad has ``ref``,
        ``pad`` (number), ``net`` (name, None if unconnected or unreadable), ``net_state``
        (``"connected"``, ``"unconnected"`` or ``"unknown"``), ``net_reason`` when unknown,
        ``position_nm``, ``layers`` (canonical names), ``shape`` and ``size_nm``
        (``{"x", "y"}``; None if KiCad gives no copper layer). With ``refs``, only pads of
        those footprints are returned.
        """
        ...

    def set_selection(self, ref: BoardRef, ids: list[str]) -> None:
        """Replace the editor selection with the items whose session IDs are ``ids``."""
        ...

    def close(self) -> None: ...


#: Item kinds ``selection`` reports, and the filter values ``get_selection`` accepts.
SELECTION_KINDS = (
    "footprint",
    "pad",
    "track",
    "arc",
    "via",
    "zone",
    "shape",
    "text",
    "textbox",
    "field",
    "dimension",
    "barcode",
    "reference_image",
    "group",
)

#: Kinds that carry copper connectivity (a net).
COPPER_KINDS = ("track", "arc", "via", "zone")


def client_api_version() -> str | None:
    """The KiCad API version the installed kicad-python was built against."""
    try:
        from kipy.kicad_api_version import KICAD_API_VERSION
    except Exception:  # pragma: no cover - only if kipy is broken
        return None
    return str(KICAD_API_VERSION)


class KipyBackend:
    """:class:`IpcBackend` over ``kipy.KiCad``. One instance holds one connection."""

    def __init__(self, socket: str, timeout_ms: int, client_name: str = "openkonnect") -> None:
        from kipy import KiCad

        self.socket = socket
        self._kicad = KiCad(socket_path=socket, client_name=client_name, timeout_ms=timeout_ms)

    # -- session -----------------------------------------------------------

    def ping(self) -> None:
        self._kicad.ping()

    def kicad_version(self) -> str:
        v = self._kicad.get_version()
        return v.full_version or f"{v.major}.{v.minor}.{v.patch}"

    def instance_token(self) -> str | None:
        # kipy keeps the token KiCad hands out on first contact; it changes when
        # KiCad restarts. It is not public API, so read it defensively.
        client = getattr(self._kicad, "_client", None)
        token = getattr(client, "_kicad_token", None)
        return token or None

    def open_boards(self) -> list[BoardRef]:
        from kipy.errors import ApiError
        from kipy.proto.common import ApiStatusCode
        from kipy.proto.common.types import DocumentType

        try:
            docs = self._kicad.get_open_documents(DocumentType.DOCTYPE_PCB)
        except ApiError as e:
            # With no PCB editor frame open, KiCad has no handler for board
            # requests. That is "no board open", not a failure.
            if e.code == ApiStatusCode.AS_UNHANDLED:
                return []
            raise
        refs = []
        for doc in docs:
            project_dir = doc.project.path or None
            project_name = doc.project.name or None
            file_name = doc.board_filename
            if not file_name:
                if project_dir:
                    # For a few hundred ms after the PCB Editor starts, KiCad
                    # 10.0.7 lists the board with an empty board_filename but a
                    # populated project path (#34; docs/ipc-client.md, "Startup
                    # window"). Joining them would give the project directory
                    # as the board path. Treat it as "not ready" so Session.call
                    # retries it and ends in a retryable kicad_busy if it never
                    # clears.
                    raise ApiError(
                        "board document not fully loaded "
                        f"(board_filename='', project.path={project_dir!r})",
                        code=ApiStatusCode.AS_NOT_READY,
                    )
                # No filename and no project: a new board that has never been
                # saved. KiCad keeps answering this way, so it is not a startup
                # state to retry. Report the board with no path.
                refs.append(BoardRef(None, "", None, project_name, handle=doc))
                continue
            # Any other non-empty name is reported as KiCad gives it, joined to
            # the project directory when relative. Only "" was ever observed to
            # mean "not loaded yet" (#34), so no other name is retried.
            path = file_name
            if project_dir and not os.path.isabs(file_name):
                path = os.path.join(project_dir, file_name)
            refs.append(BoardRef(path, file_name, project_dir, project_name, handle=doc))
        return refs

    def close(self) -> None:
        client = getattr(self._kicad, "_client", None)
        conn = getattr(client, "_conn", None)
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.close()

    # -- board -------------------------------------------------------------

    def _board(self, ref: BoardRef) -> Any:
        from kipy.board import Board

        return Board(self._kicad._client, ref.handle)

    # File sync (#4). kicad-python 0.8 documents these three Board methods,
    # each one IPC request, with no KiCad 11 restriction:
    #   Board.get_as_string() -> SaveDocumentToString
    #   Board.save()          -> SaveDocument
    #   Board.revert()        -> RevertDocument
    # There is no "modified since last save" query in 0.8 (KiCad 11 only).

    def board_text(self, ref: BoardRef) -> str:
        return self._board(ref).get_as_string()

    def save_board(self, ref: BoardRef) -> None:
        self._board(ref).save()

    def revert_board(self, ref: BoardRef) -> None:
        self._board(ref).revert()

    def board_data(self, ref: BoardRef) -> dict[str, Any]:
        from kipy.board_types import ArcTrack
        from kipy.proto.board.board_types_pb2 import ZoneType

        board = self._board(ref)

        tb = board.get_title_block_info()
        title_block = {
            "title": tb.title,
            "date": tb.date,
            "revision": tb.revision,
            "company": tb.company,
        }

        tracks = board.get_tracks()
        arcs = sum(1 for t in tracks if isinstance(t, ArcTrack))
        zones = board.get_zones()
        nets = [n for n in board.get_nets() if n.name]

        edge, outline = _edge_outline(board)

        return {
            "title_block": title_block,
            "copper_layers": board.get_copper_layer_count(),
            "enabled_layers": len(board.get_enabled_layers()),
            "footprints": len(board.get_footprints()),
            "nets": len(nets),
            "track_segments": len(tracks) - arcs,
            "track_arcs": arcs,
            "vias": len(board.get_vias()),
            "zones": len(zones),
            "rule_areas": sum(1 for z in zones if z.type == ZoneType.ZT_RULE_AREA),
            "edge_cuts_shapes": len(edge),
            "outline_nm": outline,
        }

    # Selection (#5) and the board readers; the bodies are in backend_readers.py.

    def selection(self, ref: BoardRef, kinds: list[str] | None = None) -> dict[str, Any]:
        return _selection(self, ref, kinds)

    def board_index(self, ref: BoardRef) -> dict[str, Any]:
        return _board_index(self, ref)

    def nets(self, ref: BoardRef, include_items: bool = False) -> dict[str, Any]:
        return _nets(self, ref, include_items)

    def zones(self, ref: BoardRef) -> dict[str, Any]:
        return _zones(self, ref)

    def layers_and_stackup(self, ref: BoardRef) -> dict[str, Any]:
        return _layers_and_stackup(self, ref)

    def outline(self, ref: BoardRef) -> dict[str, Any]:
        return _outline(self, ref)

    def copper(self, ref: BoardRef) -> dict[str, Any]:
        return _copper(self, ref)

    def pads(self, ref: BoardRef, refs: list[str] | None = None) -> dict[str, Any]:
        return _pads(self, ref, refs)

    def set_selection(self, ref: BoardRef, ids: list[str]) -> None:
        _set_selection(self, ref, ids)
