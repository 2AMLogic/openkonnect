"""Finding, running and checking ``kct``, the kicad-tools command-line tool.

The ``kct`` subprocess layer shared by the tools that run it:
``run_kct_handoff`` (SES-06), which runs one allow-listed write command
through the handoff protocol, and ``session_status`` (SES-01), which reports
whether ``kct`` is installed.

* :data:`KCT_ALLOW_LIST`, the board-writing commands the handoff may run, and
  :data:`BOARD_TOKEN`, the argv token replaced by the open board's path;
* :func:`validate_command`, which checks ``command`` against the allow-list
  and refuses flags that send kct's output anywhere but the open board
  (#59, #63, #65, #113), before KiCad is contacted;
* :class:`CommandResult`, :class:`Runner` and :func:`default_runner`, which
  run ``argv`` without a shell and kill the whole process tree on a timeout;
* :func:`find_kct`, the ``kct`` executable.

Nothing here talks to KiCad: the module depends on :mod:`openkonnect.errors`
only, so a tool can find or run ``kct`` without importing the write handoff
and its save and reload steps.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from . import errors as E

#: kct write commands the handoff may run, by leading argv tokens. Anything else
#: is ``invalid_argument``. Each of these rewrites a ``.kicad_pcb``.
KCT_ALLOW_LIST: tuple[tuple[str, ...], ...] = (
    ("route",),
    ("route-auto",),
    ("optimize-traces",),
    ("optimize-placement",),
    ("stitch",),
    ("zones", "add"),
    ("zones", "batch"),
    ("zones", "fill"),
    ("fix-vias",),
    ("fix-silkscreen",),
    ("fix-footprints",),
    ("fix-drc",),
    ("repair-clearance",),
    ("place-silk-refs",),
)

BOARD_TOKEN = "{board}"


@dataclass
class CommandResult:
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool = False


Runner = Callable[[list[str], float, "str | None"], CommandResult]


def _kill_tree(proc: subprocess.Popen) -> None:
    if os.name != "nt":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError):
            pass
    proc.kill()


def default_runner(argv: list[str], timeout_s: float, cwd: str | None) -> CommandResult:
    """Run ``argv`` without a shell; on timeout, kill it and its children."""
    proc = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        start_new_session=os.name != "nt",
    )
    try:
        out, err = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        out, err = proc.communicate()
        return CommandResult(None, out or "", err or "", timed_out=True)
    return CommandResult(proc.returncode, out or "", err or "")


def find_kct() -> str | None:
    found = shutil.which("kct")
    if found:
        return found
    beside = os.path.join(os.path.dirname(sys.executable), "kct")
    return beside if os.path.isfile(beside) and os.access(beside, os.X_OK) else None


def validate_command(command: Any) -> list[str]:
    """Check ``command`` against the allow-list. Returns argv without ``kct``."""
    if not isinstance(command, (list, tuple)) or not all(isinstance(a, str) for a in command):
        raise E.invalid_argument(
            "command must be a list of strings (an argv list), not a shell string.",
            got=command,
        )
    argv = list(command)
    if argv and argv[0] in ("kct", "kicad-tools"):
        argv = argv[1:]
    if not argv:
        raise E.invalid_argument("command is empty.", allowed=_allowed())
    if any("\x00" in a for a in argv):
        raise E.invalid_argument("command arguments must not contain NUL bytes.")
    if not any(tuple(argv[: len(p)]) == p for p in KCT_ALLOW_LIST):
        raise E.invalid_argument(
            f"'{' '.join(argv[:2])}' is not an allow-listed kct write command.",
            allowed=_allowed(),
        )
    if not any(BOARD_TOKEN in a for a in argv):
        raise E.invalid_argument(
            f"command must contain the token {BOARD_TOKEN}, which openkonnect replaces with "
            "the open board's path, so the command writes the file KiCad has open.",
            got=argv,
        )
    if not _check_output_flags(argv) and argv[0] in REQUIRES_OUTPUT:
        raise E.invalid_argument(
            f"kct {argv[0]} needs -o {BOARD_TOKEN}: without it, it does not write the open "
            "board (route writes <stem>_routed.kicad_pcb, route-auto saves nothing), so "
            f"the handoff would report success while KiCad's board is unchanged. Add "
            f"-o {BOARD_TOKEN} to the command.",
            got=argv,
        )
    return argv


#: Flags that make kct write somewhere other than the open board (issue #59):
#: ``--checkpoint`` writes extra state files. ``--auto-layers`` is refused too;
#: in kicad-tools 0.29.0 (observed 2026-11-20, issue #113) it is on by default,
#: and an explicit ``--auto-layers`` with ``-o {board}`` writes the open board
#: in place exactly, with no ``_4layer`` file — as 0.24.0 already did (the
#: fixture routes at 2 layers, so escalation itself never ran; that path keeps
#: its 0.24.0 observation and upstream #2809). The refusal is therefore a cheap
#: guard, not a correctness need: escalation writes were version-volatile
#: historically (pre-0.24 versions wrote ``<stem>_4layer.kicad_pcb``). Refused
#: outright, but only on the subcommands that define them, so a valid
#: abbreviation of some other flag on another subcommand (``fix-vias --c 1`` is
#: ``--copper``) is not refused.
REDIRECT_FLAGS: dict[str, frozenset[str]] = {
    "--auto-layers": frozenset({"route"}),
    "--checkpoint": frozenset({"route", "route-auto", "optimize-placement"}),
}
_OUTPUT_LONG = "--output"
#: ``--o`` is refused outright (issue #65), whatever its value and subcommand:
#: on ``route`` kct finds it ambiguous (``--output``, ``--oracle-rounds``,
#: ``--order-method``), and counting it as ``--output`` elsewhere would depend
#: on each subcommand's flag set. ``--ou`` and longer still abbreviate ``--output``.
_AMBIGUOUS_OUTPUT = "--o"

#: Subcommands that do not write the board in place when ``-o`` is absent
#: (issue #63), so a handoff must pass ``-o {board}``. In kicad-tools 0.29.0
#: (observed 2026-11-20, issue #113, on a fresh copy of
#: ``tests/fixtures/selection.kicad_pcb``) ``route`` leaves the input board
#: byte-identical and writes ``<stem>_routed.kicad_pcb`` beside it, and
#: ``route-auto`` writes nothing at all — each exiting 0, as 0.24.0 did; every
#: other allow-listed command writes the input board when ``-o`` is absent.
REQUIRES_OUTPUT: frozenset[str] = frozenset({"route", "route-auto"})


def _abbreviates(opt: str, flag: str) -> bool:
    """True if ``opt`` is ``flag`` or an argparse-style abbreviation of it."""
    return len(opt) > 2 and flag.startswith(opt)


def _check_output_flags(argv: list[str]) -> bool:
    """Refuse flags that send kct's output anywhere but ``{board}`` (issue #59).

    ``-o``/``--output`` is allowed only with the value exactly ``{board}``, in any
    spelling (``-o {board}``, ``-o{board}``, ``--output {board}``,
    ``--output={board}``, an abbreviation of ``--output`` at least ``--ou``
    long (``--o`` is refused), or ``-o`` inside a
    cluster of short flags such as ``-vo {board}`` or ``-qo{board}``).

    Returns True if an output flag (necessarily ``{board}``) was given before
    any ``--``.
    """
    subcommand = argv[0]
    seen_output = False
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--":
            return seen_output
        if arg.startswith("--"):
            opt, eq, inline = arg.partition("=")
            if opt == _AMBIGUOUS_OUTPUT:
                raise E.invalid_argument(
                    f"{_AMBIGUOUS_OUTPUT} is ambiguous (on route it also abbreviates "
                    "--oracle-rounds and --order-method), so it is not allowed in "
                    f"run_kct_handoff. Write -o {BOARD_TOKEN} or --output {BOARD_TOKEN}.",
                    got=arg,
                )
            for flag, subcommands in REDIRECT_FLAGS.items():
                if subcommand in subcommands and _abbreviates(opt, flag):
                    if flag == "--auto-layers":
                        message = (
                            "--auto-layers is refused as a conservative guard: "
                            "kicad-tools 0.29.0 honors -o {board} with it (observed "
                            "2026-11-20, #113), but older versions wrote "
                            "<stem>_4layer.kicad_pcb beside the output, leaving "
                            "KiCad's board unchanged. It is not allowed in "
                            "run_kct_handoff."
                        )
                    else:
                        message = (
                            f"{flag} makes kct write somewhere other than the open "
                            "board, so the handoff would report success while "
                            "KiCad's board is unchanged. It is not allowed in "
                            "run_kct_handoff."
                        )
                    raise E.invalid_argument(message, got=arg)
            if _abbreviates(opt, _OUTPUT_LONG):
                value = inline if eq else (argv[i + 1] if i + 1 < len(argv) else None)
                _check_output_value(arg, value)
                seen_output = True
                i += 1 if eq else 2
                continue
        elif arg.startswith("-") and len(arg) > 1 and "o" in arg[1:]:
            # Short-flag cluster containing -o: argparse reads what follows the
            # ``o`` as its value, or the next token if ``o`` is last.
            inline = arg[arg.index("o", 1) + 1 :]
            value = inline if inline else (argv[i + 1] if i + 1 < len(argv) else None)
            _check_output_value(arg, value)
            seen_output = True
            i += 1 if inline else 2
            continue
        i += 1
    return seen_output


def _check_output_value(arg: str, value: str | None) -> None:
    if value != BOARD_TOKEN:
        raise E.invalid_argument(
            f"{arg.partition('=')[0]} must be exactly {BOARD_TOKEN}: the handoff only "
            "reloads the board KiCad has open, so kct must write its result there.",
            got=value,
        )


def _allowed() -> list[str]:
    return [" ".join(p) for p in KCT_ALLOW_LIST]
