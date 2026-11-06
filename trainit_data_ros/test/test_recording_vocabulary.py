"""D0.8: the exported recording vocabulary is generated from the parser and says only what the parser enforces."""
import pytest
import yaml

from _ros_helpers import config, load_profile
from trainit_data.errors import DataError
from trainit_data.vocab import Vocabulary
from trainit_data_ros import __version__
from trainit_data_ros.recording_config import resolve
from trainit_data_ros.recording_vocabulary import FILE, main, render, vocabulary

VOCAB = Vocabulary.default()
EXPORTED = vocabulary(VOCAB)


def _resolve(triggers=None, sync=None, backend=None, **kw):
    doc = config()
    if triggers is not None:
        doc['triggers'] = triggers
    if sync is not None:
        doc['sync'] = sync
    if backend is not None:
        doc['source'] = {'backend': backend}
    return resolve(doc, VOCAB, load_profile=load_profile, use_sim_time=False, **kw)


def test_the_committed_file_is_generated_from_the_code():
    assert FILE.read_text() == render(VOCAB), 'run python3 -m trainit_data_ros.recording_vocabulary --write'
    assert main(['--check']) == 0
    doc = yaml.safe_load(FILE.read_text())
    assert doc == EXPORTED and doc['trainit_data_ros'] == __version__
    assert doc['schema'] == 'trainit-recording-vocabulary/1.0' and doc['recording_schema'] == 'trainit-recording/1.0'


def test_the_file_is_installed_to_share():
    setup = (FILE.parents[1] / 'setup.py').read_text()
    assert "'catalog/recording_vocabulary.yaml'" in setup


def _sample(kind):
    return {'MANUAL': {'id': 'manual', 'kind': 'MANUAL'},
            'INTERVAL': {'id': 'every', 'kind': 'INTERVAL', 'period_ms': 1, 'max_captures': 1},
            'EVENT': {'id': 'on_glue', 'kind': 'EVENT', 'process_channel': 'glue', 'edge': 'rising'},
            'APPLICATION_STATE': {'id': 'approach', 'kind': 'APPLICATION_STATE', 'task': 'pick_one',
                                  'step': 'pick', 'transition': 'enter'}}[kind]


@pytest.mark.parametrize('kind', EXPORTED['triggers']['kinds'])
def test_every_exported_trigger_kind_is_accepted(kind):
    assert _resolve(triggers=[_sample(kind)])['triggers'][0]['kind'] == kind


def test_every_exported_enum_value_is_accepted_and_nothing_else():
    fields = EXPORTED['triggers']['fields']
    for edge in fields['EVENT']['edge']['values']:
        _resolve(triggers=[dict(_sample('EVENT'), edge=edge)])
    for transition in fields['APPLICATION_STATE']['transition']['values']:
        _resolve(triggers=[dict(_sample('APPLICATION_STATE'), transition=transition)])
    for phase, transitions in EXPORTED['triggers']['pick_attempt_phases'].items():
        for transition in fields['APPLICATION_STATE']['transition']['values']:
            trigger = dict(_sample('APPLICATION_STATE'), transition=transition,
                           context={'pick_attempt': {'phase': phase}})
            if transition in transitions:
                _resolve(triggers=[trigger])
            else:
                with pytest.raises(DataError):
                    _resolve(triggers=[trigger])
    with pytest.raises(DataError):
        _resolve(triggers=[dict(_sample('EVENT'), edge='sideways')])


def test_the_exported_limits_are_the_parsers():
    with pytest.raises(DataError):                               # max_per_kind MANUAL: 1
        _resolve(triggers=[_sample('MANUAL'), dict(_sample('MANUAL'), id='again')])
    for name, spec in EXPORTED['triggers']['fields']['INTERVAL'].items():
        with pytest.raises(DataError):
            _resolve(triggers=[dict(_sample('INTERVAL'), **{name: spec['min'] - 1})])
    policy = EXPORTED['sync']['policies'][0]
    base = {'policy': policy['name'], 'policy_version': policy['version']}
    assert _resolve(sync=dict(base))['sync']['timeout_ms'] == EXPORTED['sync']['defaults']['timeout_ms']
    for bad in ({'timeout_ms': 0}, {'tolerance_ms': 0}):
        with pytest.raises(DataError):
            _resolve(sync=dict(base, **bad))
    with pytest.raises(DataError):                               # 'replay' is not a deployment backend
        _resolve(backend='holodeck')
    assert 'replay' not in EXPORTED['backends'] and set(EXPORTED['backends']) < set(VOCAB.rows('backends'))


def test_a_stale_file_is_detected(tmp_path, monkeypatch):
    import trainit_data_ros.recording_vocabulary as module
    stale = tmp_path / 'recording_vocabulary.yaml'
    stale.write_text(render(VOCAB).replace('MANUAL: 1', 'MANUAL: 2'))
    monkeypatch.setattr(module, 'FILE', stale)
    assert main(['--check']) == 1
