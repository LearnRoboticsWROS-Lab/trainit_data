"""Neutral fixtures (acceptance 16; D0_DESIGN.md §21.4, review m8; owner decision 2026-10-28 on guard markers).

This repository is public software: its tests and sources carry invented, customer-neutral names and
values only. The guard has two layers:

- PUBLIC (this file): the scanning logic, self-tested with SYNTHETIC markers. No protected word is written
  here -- a guard must not put into public history the words it exists to keep out.
- LOCAL (outside the public tree): the real markers, read at run time from the repository's installed hooks:
  - the customer ``PATTERN`` of ``.git/hooks/pre-push`` (the authoritative local guard);
  - ``.git/hooks/scrub_markers`` (one regular expression per line: private reference-cell / program names).

  Both are installed from ``~/fr5_ws/forensics/hooks/`` and never copied into this repository. On a fresh
  clone without them the local half is skipped; the publication gate runs the guard itself.
"""
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HOOKS = REPO / '.git' / 'hooks'
SYNTHETIC = re.compile(r'CUSTOMER_BLOCKED_TOKEN_A|PRIVATE_ROBOT_TOKEN_B|PRIVATE_PROJECT_TOKEN_C')


def _files(root=REPO):
    out = subprocess.run(['git', 'ls-files', '--cached', '--others', '--exclude-standard'], cwd=root,
                         capture_output=True, text=True, check=True)
    return [root / line for line in out.stdout.splitlines() if (root / line).is_file()]


def _hits(pattern, files, *, exempt=()):
    hits = []
    for path in files:
        if path.resolve() in exempt:
            continue
        for number, line in enumerate(path.read_text(errors='replace').splitlines(), 1):
            if pattern.search(line):
                hits.append(f'{path.name}:{number}')
    return hits


def _local_markers():
    patterns = []
    hook = HOOKS / 'pre-push'
    if hook.is_file():
        match = re.search(r"^PATTERN='([^']+)'", hook.read_text(errors='replace'), re.MULTILINE)
        if match:
            patterns.append(match.group(1))
    markers = HOOKS / 'scrub_markers'
    if markers.is_file():
        patterns += [line.strip() for line in markers.read_text().splitlines()
                     if line.strip() and not line.startswith('#')]
    return re.compile('|'.join(f'(?:{p})' for p in patterns), re.IGNORECASE) if patterns else None


def test_the_scanner_finds_markers_and_honours_exemptions(tmp_path):
    subprocess.run(['git', 'init', '-q'], cwd=tmp_path, check=True)
    (tmp_path / 'clean.py').write_text('SENSOR = "overhead_camera"\n')
    (tmp_path / 'leak.py').write_text('NOTE = "copied from customer_blocked_token_a"\n')
    (tmp_path / 'guard.py').write_text('FORBIDDEN = "PRIVATE_ROBOT_TOKEN_B"\n')
    files = _files(tmp_path)
    pattern = re.compile(SYNTHETIC.pattern, re.IGNORECASE)
    assert sorted(_hits(pattern, files)) == ['guard.py:1', 'leak.py:1']
    assert _hits(pattern, files, exempt={(tmp_path / 'guard.py').resolve()}) == ['leak.py:1']


def test_no_synthetic_marker_leaks_outside_this_file():
    assert not _hits(SYNTHETIC, _files(), exempt={Path(__file__).resolve()})


def test_no_locally_protected_marker_in_the_public_repository():
    pattern = _local_markers()
    if pattern is None:
        pytest.skip('the local guard hooks are not installed in this clone')
    hits = _hits(pattern, _files())
    assert not hits, f'{len(hits)} line(s) match a locally protected marker: {hits}'
