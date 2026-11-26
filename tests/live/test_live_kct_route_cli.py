"""Opt-in CLI regression pinning the ``kct route`` / ``kct route-auto`` write
targets on the locked kicad-tools (issue #113).

Unlike the other ``live_kicad`` tests this one needs no running KiCad and no
open board: it runs the real ``kct`` executable directly on a fresh temporary
copy of ``tests/fixtures/selection.kicad_pcb`` (its SIG net is unrouted, so a
route changes the board). It stays behind ``--live`` because it invokes the
real CLI, so it never runs in CI. It pins the black-box write behaviour the
``run_kct_handoff`` guards rest on, so a pin bump that changes it fails here
loudly instead of silently invalidating the prose in
``docs/handoff-protocol.md`` and ``spec/capabilities.md``:

- ``kct route <board>`` with no ``-o`` leaves the input board byte-identical
  and writes ``<stem>_routed.kicad_pcb`` beside it, exiting 0;
- ``kct route-auto --net SIG <board>`` with no ``-o`` writes nothing at all,
  still exiting 0;
- ``kct route <board> -o <board> --auto-layers`` writes the board in place
  exactly, with no ``_4layer`` / ``_6layer`` files (the fixture routes at 2
  layers, so escalation never runs) and leaves a ``<stem>.kicad_prl`` sibling.

Observed against kicad-tools 0.29.0 on 2026-11-20 (issue #113); the same
behaviour was observed on 0.24.0 (#37, #63). Run with
``uv run pytest --live -m live_kicad -rs`` after ``uv sync --locked``.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from openkonnect.kct import find_kct

from ..conftest import SELECTION_BOARD

pytestmark = pytest.mark.live_kicad

COMMAND_TIMEOUT_S = 300.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(kct: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [kct, *args],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT_S,
    )


@pytest.fixture
def board(tmp_path: Path) -> Path:
    if find_kct() is None:
        pytest.skip("live_kicad: kct (kicad-tools) not found")
    path = tmp_path / "selection.kicad_pcb"
    shutil.copyfile(SELECTION_BOARD, path)
    return path


def test_route_without_output_writes_sibling_not_the_board(board: Path):
    kct = find_kct()
    assert kct is not None
    before = _sha256(board)
    r = _run(kct, "route", str(board))
    assert r.returncode == 0, r.stderr
    # No -o: the open board would stay unchanged while kct exits 0.
    assert _sha256(board) == before
    routed = board.with_name(f"{board.stem}_routed.kicad_pcb")
    assert routed.is_file()
    assert _sha256(routed) != before


def test_route_auto_without_output_writes_nothing(board: Path):
    kct = find_kct()
    assert kct is not None
    before = _sha256(board)
    names_before = {p.name for p in board.parent.iterdir()}
    r = _run(kct, "route-auto", "--net", "SIG", str(board))
    assert r.returncode == 0, r.stderr
    # No -o: nothing is written at all, not even a sibling board.
    assert _sha256(board) == before
    assert {p.name for p in board.parent.iterdir()} == names_before


def test_route_auto_layers_inplace_writes_board_exactly(board: Path):
    kct = find_kct()
    assert kct is not None
    before = _sha256(board)
    r = _run(kct, "route", str(board), "-o", str(board), "--auto-layers")
    assert r.returncode == 0, r.stderr
    # An explicit --auto-layers honors the output path exactly: the board
    # changes in place and no _4layer/_6layer file appears (the fixture
    # routes at 2 layers, so escalation never runs).
    assert _sha256(board) != before
    assert not list(board.parent.glob(f"{board.stem}_4layer*"))
    assert not list(board.parent.glob(f"{board.stem}_6layer*"))
    # New in 0.29.0: the route leaves a .kicad_prl sibling (issue #113).
    assert (board.parent / f"{board.stem}.kicad_prl").is_file()
