"""Guards on the shipped example scripts.

These are cheap static checks, not full runs: executing every example would
dominate the suite. They exist because three examples once crashed on Windows
while a Linux-only CI reported success.
"""

import ast
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).resolve().parents[1] / "examples").glob("*.py"))


def test_examples_directory_is_not_empty():
    assert EXAMPLES, "no example scripts found"


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_parses(path):
    """Every example must at least be valid Python."""
    ast.parse(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_printed_text_is_ascii(path):
    """Printed output must survive a legacy console codepage.

    Windows consoles default to a legacy encoding (cp1252 in western Europe,
    cp932 in Japan, ...), not UTF-8. Printing a character outside it raises
    UnicodeEncodeError and kills the script. Greek letters and subscripts are
    the usual culprits in this package, since the domain invites them.

    Only ``print`` calls are checked. Comments, docstrings and matplotlib
    labels are free to use whatever they like: matplotlib renders unicode
    happily and never sends it to the console.
    """
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()

    offenders = []
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"):
            continue
        for lineno in range(node.lineno, (node.end_lineno or node.lineno) + 1):
            if lineno > len(lines):
                continue
            line = lines[lineno - 1]
            bad = sorted({c for c in line if ord(c) > 127})
            if bad:
                chars = ", ".join(f"{c!r} (U+{ord(c):04X})" for c in bad)
                offenders.append(f"  line {lineno}: {chars}")

    assert not offenders, (
        f"{path.name} prints non-ASCII characters, which raises "
        f"UnicodeEncodeError on a legacy Windows console:\n"
        + "\n".join(offenders)
        + "\n\nUse an ASCII spelling in printed text (omega, sigma, +/-)."
    )


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_guards_its_entry_point(path):
    """Anything using parallel workers needs a __main__ guard.

    Process workers re-import the calling module on Windows and macOS, so an
    unguarded example would recurse into spawning more workers.
    """
    source = path.read_text(encoding="utf-8")
    # Only n_workers actually starts a process pool. run_chains without it is
    # sequential and needs no guard.
    if "n_workers" not in source:
        pytest.skip("does not start worker processes")
    assert '__name__ == "__main__"' in source or "__name__ == '__main__'" in source, (
        f"{path.name} starts worker processes but has no __main__ guard"
    )
