"""How openkonnect decides that an editor buffer matches the file on disk.

The board dirty-state and file fingerprint helpers shared by the file-sync
tools (#4): ``save_before_external_write``, ``sync_from_disk``,
``run_kct_handoff`` and ``show_findings``.

* :func:`board_dirty_state`, the SES-01 dirty-state tri-state for one open
  board, decided from fresh reads of the editor's text and of the file;
* :class:`DirtyState`, its result, where ``unknown`` protects unsaved work;
* :func:`bind_to_file`, which ties a clean result to the file hash a caller
  read separately;
* :func:`file_state`, :func:`read_bytes`, :func:`sha256_bytes` and
  :func:`text_sha256`, the file fingerprints.

These take a :class:`openkonnect.session.Session` to reach the editor. The
connection layer does not depend on this module.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any

from . import kicad_text
from .backend import BoardRef
from .errors import ErrorCode, OpenKonnectError
from .session import Session

#: ``dirty_basis`` values. ``kicad_flag`` is KiCad 11's "modified since last
#: save" query, which kicad-python 0.8 / KiCad 10 do not offer, so this module
#: never produces it.
BASIS_KICAD_FLAG = "kicad_flag"
BASIS_CONTENT = "content_comparison"
BASIS_NONE = "none"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def text_sha256(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def file_state(path: str) -> dict[str, Any]:
    """What is on disk at ``path``: existence, SHA-256, size and mtime."""
    try:
        with open(path, "rb") as f:
            data = f.read()
        st = os.stat(path)
    except FileNotFoundError:
        return {"exists": False, "sha256": None, "size": None, "mtime_ns": None}
    return {
        "exists": True,
        "sha256": sha256_bytes(data),
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
    }


def read_bytes(path: str) -> bytes | None:
    try:
        with open(path, "rb") as f:
            return f.read()
    except FileNotFoundError:
        return None


@dataclass
class DirtyState:
    """SES-01 dirty state for one document: ``True``, ``False`` or ``"unknown"``."""

    dirty: bool | str
    basis: str
    note: str
    #: SHA-256 of the editor's native-format text when it could be read.
    editor_sha256: str | None = None
    #: How a ``False`` was proven: ``"bytes"`` or ``"canonical"`` (issue #37).
    match_basis: str | None = None
    #: SHA-256 of the exact file bytes a ``False`` was proven against. A caller
    #: that reports or checks a file hash read separately must bind the two with
    #: :func:`bind_to_file`.
    file_sha256: str | None = None

    @property
    def must_protect(self) -> bool:
        """``unknown`` counts as dirty for every tool that protects unsaved work."""
        return self.dirty is not False

    def record(self) -> dict[str, Any]:
        rec: dict[str, Any] = {
            "dirty": self.dirty,
            "dirty_basis": self.basis,
            "dirty_note": self.note,
        }
        if self.match_basis is not None:
            rec["match_basis"] = self.match_basis
        return rec


def read_board_text(session: Session, ref: BoardRef) -> str | None:
    """The editor's board as native-format text, or None if KiCad will not serve it.

    Connection-class errors (busy, timeout, instance change, unreachable)
    propagate; only "KiCad rejected / does not implement this request" is
    turned into None.
    """
    try:
        return session.call(lambda b: b.board_text(ref))
    except OpenKonnectError as err:
        if err.code in (ErrorCode.UNSUPPORTED_ON_THIS_KICAD, ErrorCode.INTERNAL):
            return None
        raise


def read_kicad_version(session: Session) -> str | None:
    """KiCad's version string, or None if KiCad will not say.

    Connection-class errors propagate, as in :func:`read_board_text`.
    """
    try:
        return session.call(lambda b: b.kicad_version())
    except OpenKonnectError as err:
        if err.code in (ErrorCode.UNSUPPORTED_ON_THIS_KICAD, ErrorCode.INTERNAL):
            return None
        raise


def editor_match(session: Session, text: str, on_disk: bytes | None) -> str | None:
    """How the editor text equals the file, both read fresh by the caller, or None.

    ``"bytes"`` if byte-identical. Otherwise ``"canonical"`` if the running
    KiCad is on the measured allow-list
    (:data:`openkonnect.kicad_text.CANONICAL_MATCH_VERSIONS`) and
    :func:`openkonnect.kicad_text.same_document` holds. Otherwise None: not
    shown equal. Only fresh content reaches this function; it never sees a
    caller-supplied hash (issue #15, #37).
    """
    if on_disk is None:
        return None
    if on_disk == text.encode("utf-8"):
        return kicad_text.MATCH_BYTES
    if kicad_text.canonical_comparison_applies(
        read_kicad_version(session)
    ) and kicad_text.same_document(text, on_disk):
        return kicad_text.MATCH_CANONICAL
    return None


def bind_to_file(ds: DirtyState, file_sha256: str | None) -> DirtyState:
    """``ds``, unless it is clean against file bytes other than ``file_sha256``.

    :func:`board_dirty_state` reads the file itself. A caller that read the file
    separately (to report its hash, or to check it against an expected hash)
    gets a ``False`` only if the match was proven on bytes with that same hash.
    If the file was written between the two reads, the match does not hold for
    the bytes the caller reports, so the state becomes ``unknown`` (issue #37).
    """
    if ds.dirty is not False or (file_sha256 is not None and ds.file_sha256 == file_sha256):
        return ds
    return DirtyState(
        "unknown",
        BASIS_CONTENT,
        "The file on disk changed while openkonnect compared it with the editor, so the "
        "match does not hold for the file it reports. It counts as dirty.",
        ds.editor_sha256,
    )


def board_dirty_state(
    session: Session,
    ref: BoardRef,
    *,
    editor_baseline_sha256: str | None = None,
) -> DirtyState:
    """Decide dirty state for an open board from documented signals only.

    KiCad 10 has no "modified since last save" query. The documented signal
    used instead is the editor's own native-format text (``SaveDocumentToString``),
    read fresh by this call, compared with the file on disk, also read fresh by
    this call:

    * The board is clean (``False``) only when the fresh editor text equals
      the fresh file: byte-identical, or, on a KiCad version where the
      serializer difference was measured (10.0.7), equal by the strict token
      comparison of :mod:`openkonnect.kicad_text` (``match_basis``
      ``"canonical"``). Then nothing in the editor is missing from the file.
      This is the only way to reach ``False``.
    * Otherwise, with ``editor_baseline_sha256`` (a caller-supplied hash of the
      editor text, e.g. ``editor_sha256`` from a save), an editor text that
      differs from it is dirty (``True``): someone edited the board in KiCad
      since. An editor text equal to it is ``unknown``: a caller-supplied hash
      cannot show the editor holds nothing the file lacks (for example after a
      save from KiCad followed by an edit and an undo). Issue #15.
    * Without a baseline, any difference is ``unknown``: it may be unsaved work,
      or only a formatting difference between KiCad's serializer and whatever
      wrote the file.
    * If the text cannot be read, the state is ``unknown`` with basis ``none``.

    A caller-supplied hash can only move the result toward ``True`` or
    ``unknown``, never to ``False``. ``False`` is never guessed.
    """
    text = read_board_text(session, ref)
    if text is None:
        return DirtyState(
            "unknown",
            BASIS_NONE,
            "KiCad did not serve the board as text, and KiCad 10 has no dirty-state query.",
        )
    h = text_sha256(text)
    on_disk = read_bytes(ref.path)
    match = editor_match(session, text, on_disk)
    if match == kicad_text.MATCH_BYTES:
        return DirtyState(
            False,
            BASIS_CONTENT,
            "The editor content is byte-identical to the saved file.",
            h,
            match,
            sha256_bytes(on_disk),
        )
    if match == kicad_text.MATCH_CANONICAL:
        return DirtyState(
            False,
            BASIS_CONTENT,
            "The editor content equals the saved file token for token. Only whitespace "
            "between tokens and the version/generator lines inside each footprint differ, "
            "which is how this KiCad version's save differs from its editor text.",
            h,
            match,
            sha256_bytes(on_disk),
        )
    if editor_baseline_sha256 is not None and h != editor_baseline_sha256:
        return DirtyState(
            True,
            BASIS_CONTENT,
            "The editor content changed since the recorded baseline (editor_sha256): "
            "someone edited the board in KiCad.",
            h,
        )
    if on_disk is None:
        return DirtyState("unknown", BASIS_NONE, "The board has no file on disk.", h)
    if editor_baseline_sha256 is not None:
        return DirtyState(
            "unknown",
            BASIS_CONTENT,
            "The editor content equals the recorded baseline (editor_sha256) but differs "
            "from the file on disk. A recorded hash cannot prove the editor holds nothing "
            "the file lacks (the file may have been saved from KiCad since, then the edit "
            "undone), so it counts as dirty.",
            h,
        )
    return DirtyState(
        "unknown",
        BASIS_NONE,
        "The editor content differs from the saved file. KiCad 10 cannot say whether that "
        "is unsaved work or only formatting, so it counts as dirty.",
        h,
    )
