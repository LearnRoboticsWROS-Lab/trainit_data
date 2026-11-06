"""D0.8: the synthetic provider contract (``trainit-synthetic-provider/1.0``) and the reference provider.

The descriptor is the ONE contract between a launching tool (TSA) and a provider: every declared parameter is a
real key of the provider's configuration (absent controls are never decorative), the base configuration is valid
against it, and the provider refuses an invalid configuration BEFORE its runtime starts (no Isaac needed here).
"""
import copy
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from trainit_data.errors import DataError
from trainit_data.synthetic_provider import (config_problems, general_problems, get_path, load_descriptor,
                                             parse_descriptor, value_problem)

PKG_ROOT = Path(__file__).resolve().parents[1]
PROVIDER = PKG_ROOT.parent / 'examples' / 'isaac_replicator_bin'
DESCRIPTOR = load_descriptor(PROVIDER)
BASE = json.loads((PROVIDER / DESCRIPTOR['base_config']).read_text())


def _generator(path=PROVIDER / 'generate.py'):
    spec = importlib.util.spec_from_file_location(f'provider_generate_{abs(hash(str(path)))}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_reference_descriptor_is_complete_and_its_base_config_valid():
    assert DESCRIPTOR['id'] == 'isaac_replicator_bin' and DESCRIPTOR['backends'] == ['isaac']
    assert (PROVIDER / DESCRIPTOR['entry']).is_file() and (PROVIDER / DESCRIPTOR['base_config']).is_file()
    assert config_problems(DESCRIPTOR, BASE) == []
    defaults = {k: v['default'] for k, v in DESCRIPTOR['general'].items()}
    assert general_problems(DESCRIPTOR, **defaults) == []
    assert _generator().world_problems(BASE) == []
    assert DESCRIPTOR['outputs']['classes'] == BASE['classes']
    assert {p['level'] for p in DESCRIPTOR['parameters']} == {'simple', 'advanced'}


def test_every_declared_parameter_is_read_by_the_generator():
    """No decorative control: each key path exists in the base configuration AND its leaf is read by the code."""
    source = (PROVIDER / 'generate.py').read_text()
    for parameter in DESCRIPTOR['parameters']:
        get_path(BASE, parameter['key'])                          # KeyError = a control with no field
        leaf = parameter['key'].rsplit('.', 1)[-1]
        assert f"['{leaf}']" in source, f'{parameter["key"]} is declared but generate.py never reads it'


@pytest.mark.parametrize('key, value', [
    ('objects_per_run', [0, 3]), ('objects_per_run', [5, 4]), ('objects_per_run', [1.5, 3]),
    ('object_size_m', [0.01, 0.5]), ('lights.dome_intensity', 'bright'), ('color_range', [0.2]),
    ('camera.height_m', 10.0), ('tray.inner_m', [0.1, 0.3]), ('resolution', [1000, 1000]),
    ('rt_subframes', 0), ('settle_steps', True), ('placement.margin_m', -0.01),
])
def test_out_of_bounds_values_are_refused(key, value):
    cfg = copy.deepcopy(BASE)
    node = cfg
    *parents, leaf = key.split('.')
    for part in parents:
        node = node[part]
    node[leaf] = value
    problems = config_problems(DESCRIPTOR, cfg)
    assert len(problems) == 1 and problems[0].startswith(key)


def test_a_missing_key_and_bad_general_values_are_refused():
    cfg = copy.deepcopy(BASE)
    del cfg['lights']['key_intensity']
    assert config_problems(DESCRIPTOR, cfg) == ['lights.key_intensity: absent from the configuration']
    assert len(general_problems(DESCRIPTOR, seed=-1, runs=0, views=17)) == 3
    assert general_problems(DESCRIPTOR, seed=0, runs=1, views=1) == []


def test_float_pair_order_is_free_but_ranges_are_ordered():
    pair = {'type': 'float_pair', 'min': 0.2, 'max': 1.2}
    assert value_problem(pair, [0.5, 0.3]) is None
    assert value_problem({'type': 'float_range', 'min': 0.0, 'max': 1.0}, [0.5, 0.3])


def test_world_rules_beyond_per_field_bounds():
    gen = _generator()
    cfg = copy.deepcopy(BASE)
    cfg['tray']['inner_m'] = [1.2, 1.2]                           # valid per field, out of the camera's view
    assert config_problems(DESCRIPTOR, cfg) == [] and 'camera view' in gen.world_problems(cfg)[0]
    cfg['camera']['height_m'] = 3.0
    assert gen.world_problems(cfg) == []


@pytest.mark.parametrize('mutate', [
    lambda d: d.update(surprise=1),
    lambda d: d.update(schema='trainit-synthetic-provider/2.0'),
    lambda d: d.update(progress='other/1'),
    lambda d: d['parameters'].append(dict(d['parameters'][0])),                    # a key declared twice
    lambda d: d['parameters'][0].update(type='vector'),
    lambda d: d['parameters'][0].update(level='expert'),
    lambda d: d['parameters'][0].update(key='lights..x'),
    lambda d: d['parameters'][1].update(min=1.0, max=0.5),
    lambda d: d['general']['runs'].update(default=0),
    lambda d: d['general'].pop('views'),
    lambda d: d.update(entry='/abs/generate.py'),
    lambda d: d.update(backends=[]),
])
def test_the_descriptor_is_strict(mutate):
    document = json.loads((PROVIDER / 'provider.json').read_text())
    mutate(document)
    with pytest.raises(DataError):
        parse_descriptor(document)


def _run(entry, *args):
    return subprocess.run([sys.executable, str(entry), *args], capture_output=True, text=True, timeout=60)


def test_the_provider_refuses_an_invalid_configuration_before_isaac(tmp_path):
    cfg = copy.deepcopy(BASE)
    cfg['objects_per_run'] = [0, 99]
    path = tmp_path / 'resolved.json'
    path.write_text(json.dumps(cfg))
    out = _run(PROVIDER / 'generate.py', '--config', str(path), '--seed', '1', '--runs', '0', '--store',
               str(tmp_path / 'store'))
    assert out.returncode == 2, out.stdout + out.stderr
    lines = out.stdout.splitlines()
    assert lines[0] == 'TRAINIT_PROGRESS {"phase": "starting"}' and lines[-1] == 'TRAINIT_FAILED'
    refusal = json.loads(lines[-2].split(' ', 1)[1])
    assert refusal['phase'] == 'invalid_config' and len(refusal['problems']) == 2
    assert not (tmp_path / 'store').exists()                      # nothing written


def test_the_installed_layout_finds_its_core(tmp_path):
    """Installed under <prefix>/share/trainit_data/synthetic_providers/<id>/, the generator finds the core in
    <prefix>/lib/python3.*/site-packages (the colcon / pip install layout), and nothing else."""
    share = tmp_path / 'share' / 'trainit_data' / 'synthetic_providers' / 'isaac_replicator_bin'
    shutil.copytree(PROVIDER, share)
    site = tmp_path / 'lib' / 'python3.10' / 'site-packages'
    site.mkdir(parents=True)
    with pytest.raises(RuntimeError):
        _generator(share / 'generate.py').core_path()
    (site / 'trainit_data').symlink_to(PKG_ROOT / 'trainit_data')
    assert _generator(share / 'generate.py').core_path() == site
    assert _generator().core_path() == PKG_ROOT                   # the source checkout: the repository's core


def test_store_wins_over_the_environment(monkeypatch, tmp_path):
    """A launching tool's scratch store (a preview) is never the user's: --store beats TRAINIT_DATA_ROOT."""
    from trainit_data.store import resolve_store_root
    gen = _generator()
    monkeypatch.setenv('TRAINIT_DATA_ROOT', str(tmp_path / 'user_store'))
    args = gen.parse_args(['--seed', '1', '--runs', '1', '--store', str(tmp_path / 'scratch')])
    assert str(gen.store_root(args, resolve_store_root)) == str(tmp_path / 'scratch')
    args = gen.parse_args(['--seed', '1', '--runs', '1'])
    assert str(gen.store_root(args, resolve_store_root)) == str(tmp_path / 'user_store')
