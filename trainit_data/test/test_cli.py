"""The ``trainit-data`` CLI (acceptance 17): the ``--json`` machine schema ``trainit-cli/1`` is stable.

The key sets below ARE the contract TSA (D0.8) will read. Changing one is a ``trainit-cli/2``.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import trainit_data
from _helpers import open_session, rgbd, stamp, sync, trigger
from trainit_data.cli import main

PKG_ROOT = Path(__file__).resolve().parents[1]

TOP = {'schema', 'command', 'ok', 'trainit_data', 'result', 'error'}
ERROR = {'code', 'message', 'context'}
STORE_SUMMARY = {'root', 'store_id', 'schemas', 'sessions'}
STORE_SESSION_ENTRY = {'session_id', 'status', 'transport', 'backend', 'captures'}
SESSION_REPORT = {'session_id', 'status', 'ok', 'captures', 'observations', 'payload_bytes', 'findings'}
STORE_REPORT = {'root', 'store_id', 'ok', 'sessions', 'findings', 'datasets'}   # datasets: D0.6, additive
FINDING = {'code', 'severity', 'artifact', 'detail'}             # + 'record' when one is concerned
RECOVERY = {'schema', 'session_id', 'recovered_wall_utc_ns', 'trainit_data', 'committed_captures', 'quarantined',
            'index'}                                              # + 'torn_tail' when there was one


def run(*args, env_root=None):
    """The CLI as a separate process, exactly as TSA will launch it (``python3 -m trainit_data``)."""
    env = dict(os.environ, PYTHONPATH=str(PKG_ROOT))
    if env_root is not None:
        env['TRAINIT_DATA_ROOT'] = str(env_root)
    out = subprocess.run([sys.executable, '-m', 'trainit_data', *args], env=env, capture_output=True, text=True)
    return out.returncode, out.stdout, out.stderr


def run_json(*args, **kw):
    code, stdout, _ = run('--json', *args, **kw)
    doc = json.loads(stdout)
    assert set(doc) == TOP and doc['schema'] == 'trainit-cli/1' and doc['trainit_data'] == trainit_data.__version__
    return code, doc


def _sealed_and_active(store):
    sealed = open_session(store)
    at = stamp()
    sealed.append_capture(trigger=trigger(at), outcome='COMPLETE', sync=sync(), samples=rgbd(at))
    sealed.close()
    active = open_session(store)
    active.append_capture(trigger=trigger(at), outcome='COMPLETE', sync=sync(), samples=rgbd(at))
    active.release()
    return sealed.session_id, active.session_id


def test_inspect_json_schema(store):
    sealed, active = _sealed_and_active(store)
    code, doc = run_json('inspect', env_root=store.root)           # the root from TRAINIT_DATA_ROOT
    assert code == 0 and doc['ok'] and doc['error'] is None and doc['command'] == 'inspect'
    assert set(doc['result']) == STORE_SUMMARY
    assert all(set(e) == STORE_SESSION_ENTRY for e in doc['result']['sessions'])
    code, doc = run_json('--root', str(store.root), 'inspect', sealed)
    assert code == 0 and set(doc['result']) == {'session'} and doc['result']['session']['status'] == 'CLOSED'
    code, doc = run_json('--root', str(store.root), 'inspect', active)
    assert set(doc['result']) == {'session', 'committed'} and doc['result']['committed']['complete'] == 1


SESSION_DETAIL = {'opened_wall_utc_ns', 'classification', 'sensors', 'outcomes', 'synthetic'}   # D0.8, --detail
CAPTURES_RESULT = {'session', 'root', 'captures', 'captures_total'}                           # D0.8, --captures


def test_inspect_detail_and_captures_d0_8(store):
    """The read-only views the TSA Data panel reads: additive keys, only when the option is given."""
    sealed, active = _sealed_and_active(store)
    root = ('--root', str(store.root))
    code, doc = run_json(*root, 'inspect', '--detail')
    assert code == 0 and set(doc['result']) == STORE_SUMMARY
    entries = {e['session_id']: e for e in doc['result']['sessions']}
    assert all(set(e) == STORE_SESSION_ENTRY | SESSION_DETAIL for e in entries.values())
    assert entries[sealed]['outcomes'] == {'complete': 1, 'partial': 0, 'failed': 0}
    assert entries[active]['outcomes'] is None and entries[active]['status'] == 'ACTIVE'
    assert entries[sealed]['synthetic'] is False and entries[sealed]['classification'] == 'INTERNAL'
    code, doc = run_json(*root, 'inspect', sealed, '--captures')
    assert code == 0 and set(doc['result']) == CAPTURES_RESULT and doc['result']['captures_total'] == 1
    capture = doc['result']['captures'][0]
    assert capture['capture_id'] == f'{sealed}/000001' and capture['outcome'] == 'COMPLETE'
    for obs in capture['observations']:
        assert (Path(doc['result']['root']) / obs['payload']['path']).is_file()
    code, doc = run_json(*root, 'inspect', active, '--captures', '--limit', '0')
    assert set(doc['result']) == CAPTURES_RESULT | {'committed'}
    assert doc['result']['captures'] == [] and doc['result']['captures_total'] == 1
    before = sorted(str(p) for p in Path(store.root).rglob('*'))
    for bad in (('inspect', '--captures'), ('inspect', sealed, '--detail'), ('inspect', sealed, '--limit', '1'),
                ('inspect', sealed, '--captures', '--limit', '-1')):
        code, doc = run_json(*root, *bad)
        assert code == 1 and doc['error']['code'] == 'INVALID_VALUE'
    assert sorted(str(p) for p in Path(store.root).rglob('*')) == before       # read-only


def test_verify_json_schema_and_exit_codes(store):
    sealed, _ = _sealed_and_active(store)
    code, doc = run_json('--root', str(store.root), 'verify', sealed)
    assert code == 0 and set(doc['result']) == SESSION_REPORT and doc['result']['findings'] == []
    code, doc = run_json('--root', str(store.root), 'verify')
    assert code == 0 and set(doc['result']) == STORE_REPORT
    assert all(set(r) == SESSION_REPORT for r in doc['result']['sessions'])
    payload = next((store.session_dir(sealed) / 'payload' / '000001').iterdir())
    payload.write_bytes(payload.read_bytes() + b'!')
    code, doc = run_json('--root', str(store.root), 'verify', sealed)
    assert code == 1 and not doc['ok'] and doc['error'] is None
    finding = doc['result']['findings'][0]
    assert set(finding) == FINDING | {'record'} and finding['code'] == 'HASH_MISMATCH'


def test_recover_json_schema(store):
    _, active = _sealed_and_active(store)
    code, doc = run_json('--root', str(store.root), 'recover', active)
    assert code == 0 and set(doc['result']) == RECOVERY and doc['result']['schema'] == 'trainit-recovery/1.0'


def test_refusals_are_structured_errors(tmp_path, store):
    code, doc = run_json('--root', str(tmp_path / 'nowhere'), 'inspect')
    assert code == 1 and not doc['ok'] and doc['result'] is None
    assert set(doc['error']) == ERROR and doc['error']['code'] == 'STORE_NOT_INITIALIZED'
    assert not (tmp_path / 'nowhere').exists()                     # the CLI never creates a store
    sealed, _ = _sealed_and_active(store)
    code, doc = run_json('--root', str(store.root), 'recover', sealed)
    assert code == 1 and doc['error']['code'] == 'SESSION_SEALED'
    code, doc = run_json('--root', str(store.root), 'inspect', 'ses-20000101T000000Z-00000000')
    assert doc['error']['code'] == 'SESSION_NOT_FOUND'


def test_human_output_is_readable(store, capsys):
    sealed, active = _sealed_and_active(store)
    assert main(['--root', str(store.root), 'verify']) == 0
    out = capsys.readouterr().out
    assert f'{sealed}  CLOSED  OK' in out and 'verify: OK' in out
    assert main(['--root', str(store.root), 'recover', active]) == 0
    out = capsys.readouterr().out
    assert 'sealed INTERRUPTED' in out and 'nothing was deleted' in out
    assert main(['--root', str(store.root), 'recover', active]) == 1
    assert 'REFUSED SESSION_SEALED' in capsys.readouterr().err


def test_usage_errors_exit_2():
    code, _, stderr = run('frobnicate')
    assert code == 2 and 'usage' in stderr


# ---- D0.6: dataset / version / labels / export / verify --dataset / verify-export ---------------------------

DATASET_REPORT = {'dataset_id', 'ok', 'versions', 'label_sets', 'findings'}
EXPORT_RESULT = {'destination', 'export_id', 'exporter', 'source', 'rights', 'summary', 'files'}
EXPORT_REPORT = {'destination', 'export_id', 'ok', 'files', 'findings'}


def test_the_d0_6_commands_end_to_end_json(store, vocab, tmp_path):
    from _d06 import color_obs, make_session, yolo_line
    from trainit_data.formats.yolo import stem_of
    sim = make_session(store, vocab, simulated=True)
    real = make_session(store, vocab, simulated=False, seed=10)
    root = ('--root', str(store.root))
    code, doc = run_json(*root, 'dataset', 'create', 'parts', '--name', 'Parts', '--object', 'part')
    assert code == 0 and doc['command'] == 'dataset create' and set(doc['result']) == {'dataset'}
    assert doc['result']['dataset']['rights'] == {'classification': 'INTERNAL'}
    code, doc = run_json(*root, 'dataset', 'create', 'parts', '--name', 'Again')
    assert code == 1 and doc['error']['code'] == 'DATASET_EXISTS'

    labels = tmp_path / 'labels'
    labels.mkdir()
    (labels / f'{stem_of(color_obs(sim, 1))}.txt').write_text(yolo_line(0, 1, 1, 4, 3))
    (labels / f'{stem_of(color_obs(real, 2))}.txt').write_text(yolo_line(1, 2, 2, 2, 2))
    code, doc = run_json(*root, 'labels', 'import-yolo', 'parts', str(labels), '--class', 'part', '--class', 'tray',
                         '--notes', 'synthetic test labels')
    assert code == 0 and set(doc['result']) == {'label_set'}
    ls_id = doc['result']['label_set']['label_set_id']
    code, doc = run_json(*root, 'labels', 'show', 'parts')
    assert code == 0 and [x['label_set_id'] for x in doc['result']['label_sets']] == [ls_id]

    code, doc = run_json(*root, 'version', 'create', 'parts', '--session', sim.session_id, '--session',
                         real.session_id, '--label-set', ls_id, '--assign', f'{sim.session_id}=train',
                         '--assign', f'{real.session_id}=val')
    assert code == 0 and doc['command'] == 'version create' and set(doc['result']) == {'version'}
    version = doc['result']['version']
    assert version['version'] == 1 and version['content']['selection']['captures'] == 6
    code, doc = run_json(*root, 'version', 'create', 'parts', '--session', sim.session_id, '--number', '1')
    assert code == 1 and doc['error']['code'] == 'VERSION_EXISTS'
    code, doc = run_json(*root, 'version', 'show', 'parts', '1')
    assert code == 0 and doc['result']['version']['content_hash'] == version['content_hash']
    code, doc = run_json(*root, 'dataset', 'show', 'parts')
    assert code == 0 and doc['result']['versions'] == [1] and doc['result']['label_sets'] == [ls_id]
    code, doc = run_json(*root, 'dataset', 'list')
    assert code == 0 and doc['result']['datasets'][0]['dataset_id'] == 'parts'

    out = tmp_path / 'export'
    code, doc = run_json(*root, 'export', 'yolo', 'parts', '1', str(out))
    assert code == 0 and doc['command'] == 'export yolo' and set(doc['result']) == {'export'}
    assert set(doc['result']['export']) == EXPORT_RESULT and doc['result']['export']['files'] == 5
    code, doc = run_json(*root, 'export', 'yolo', 'parts', '1', str(tmp_path / 'seg'), '--task', 'segment')
    assert code == 1 and doc['error']['code'] == 'EXPORT_REFUSED'
    code, doc = run_json(*root, 'verify-export', str(out))
    assert code == 0 and set(doc['result']) == EXPORT_REPORT and doc['result']['ok']
    code, doc = run_json(*root, 'verify', '--dataset', 'parts')
    assert code == 0 and set(doc['result']) == DATASET_REPORT and doc['result']['versions'] == [1]
    code, doc = run_json(*root, 'verify')
    assert code == 0 and set(doc['result']) == STORE_REPORT
    assert [set(d) for d in doc['result']['datasets']] == [DATASET_REPORT]
    code, stdout, _ = run(*root, 'version', 'show', 'parts', '1')
    assert code == 0 and 'version 1  CREATE  OBSERVATIONS' in stdout


def test_the_d0_6_commands_refuse_and_report_with_exit_1(store, vocab, tmp_path):
    from _d06 import color_obs, make_session, yolo_line
    from trainit_data.formats.yolo import stem_of
    sim = make_session(store, vocab, simulated=True)
    real = make_session(store, vocab, simulated=False, seed=10)
    root = ('--root', str(store.root))
    code, doc = run_json(*root, 'dataset', 'show', 'nope')
    assert code == 1 and doc['error']['code'] == 'DATASET_NOT_FOUND'
    run_json(*root, 'dataset', 'create', 'parts', '--name', 'Parts')
    labels = tmp_path / 'labels'
    labels.mkdir()
    (labels / f'{stem_of(color_obs(sim, 1))}.txt').write_text(yolo_line(0, 1, 1, 4, 3) + '0 0.5 0.5 0.2\n')
    code, doc = run_json(*root, 'labels', 'import-yolo', 'parts', str(labels), '--class', 'part')
    assert code == 1 and doc['error']['code'] == 'LABEL_INVALID'
    assert f'{stem_of(color_obs(sim, 1))}.txt:2' in doc['error']['message']
    (labels / f'{stem_of(color_obs(sim, 1))}.txt').write_text(yolo_line(0, 1, 1, 4, 3))
    (labels / f'{stem_of(color_obs(real, 1))}.txt').write_text(yolo_line(0, 2, 2, 2, 2))
    _, doc = run_json(*root, 'labels', 'import-yolo', 'parts', str(labels), '--class', 'part')
    first = doc['result']['label_set']['label_set_id']

    # every derivation op through the CLI
    code, doc = run_json(*root, 'version', 'create', 'parts', '--session', sim.session_id, '--label-set', first,
                         '--seed', '3', '--train', '0.5', '--val', '0.5', '--test', '0')
    assert code == 0 and doc['result']['version']['version'] == 1
    code, doc = run_json(*root, 'version', 'create', 'parts', '--op', 'ADD_SESSIONS', '--from', '1', '--session',
                         real.session_id, '--assign', f'{real.session_id}=val')
    assert code == 0 and doc['result']['version']['derivation']['op'] == 'ADD_SESSIONS'
    code, doc = run_json(*root, 'version', 'create', 'parts', '--op', 'FILTER', '--from', '2', '--remove-session',
                         real.session_id)
    assert code == 0 and doc['result']['version']['content']['selection']['captures'] == 3
    (labels / f'{stem_of(color_obs(real, 1))}.txt').unlink()
    (labels / f'{stem_of(color_obs(sim, 1))}.txt').write_text(yolo_line(0, 1, 1, 5, 3))
    _, doc = run_json(*root, 'labels', 'import-yolo', 'parts', str(labels), '--class', 'part', '--source', 'HUMAN',
                      '--supersedes', first)
    second = doc['result']['label_set']['label_set_id']
    code, doc = run_json(*root, 'version', 'create', 'parts', '--op', 'RELABEL', '--from', '3', '--label-set', second,
                         '--drop-label-set', first)
    assert code == 0 and doc['result']['version']['content']['label_sets'][0]['label_set_id'] == second
    code, doc = run_json(*root, 'version', 'create', 'parts', '--op', 'RESPLIT', '--from', '4', '--assign',
                         f'{sim.session_id}=test')
    assert code == 0 and doc['result']['version']['content']['split']['counts']['test']['groups'] == 1
    code, doc = run_json(*root, 'version', 'create', 'parts', '--op', 'RESPLIT', '--from', '5', '--session', 'x')
    assert code == 1 and doc['error']['code'] == 'DERIVATION_INVALID'
    code, doc = run_json(*root, 'version', 'create', 'parts', '--train', '1')
    assert code == 1 and doc['error']['code'] == 'INVALID_VALUE'

    busy = tmp_path / 'busy'
    busy.mkdir()
    (busy / 'keep').write_text('mine')
    code, doc = run_json(*root, 'export', 'yolo', 'parts', '5', str(busy))
    assert code == 1 and doc['error']['code'] == 'EXPORT_DESTINATION_INVALID'
    out = tmp_path / 'export'
    code, doc = run_json(*root, 'export', 'yolo', 'parts', '5', str(out), '--label-set', second)
    assert code == 0 and doc['result']['export']['summary']['splits']['test']['images'] == 1

    label = next((out / 'labels' / 'test').iterdir())
    label.write_text('0 0.1 0.1 0.1 0.1\n')
    code, doc = run_json(*root, 'verify-export', str(out))
    assert code == 1 and not doc['result']['ok'] and doc['error'] is None
    manifest = store.path('datasets/parts/versions/1/selection.jsonl')
    os.chmod(manifest, 0o644)
    manifest.write_bytes(manifest.read_bytes() + b'\n')
    code, doc = run_json(*root, 'verify', '--dataset', 'parts')
    assert code == 1 and not doc['result']['ok'] and set(doc['result']) == DATASET_REPORT
    assert all(set(f) >= FINDING for f in doc['result']['findings'])
