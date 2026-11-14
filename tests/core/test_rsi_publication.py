"""Publication checks for aggregate-only experiment exports.

reuse-checked: existing importer privacy tests and z0evals artifact hashes.
"""
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_boundary_rejects_raw_fields_and_private_paths(tmp_path):
    check = load_script('check_rsi_public').check
    artifact = tmp_path / 'summary.json'
    artifact.write_text(json.dumps({'rows': [{'body': 'raw candidate'}]}))
    assert check(artifact, aggregate=True)
    artifact.write_text(json.dumps({'path': '/' + 'mnt' + '/private-data'}))
    assert check(artifact, aggregate=True)
    artifact.write_text(json.dumps({'unexpected': 'ghp_' + 'a' * 32}))
    assert check(artifact, aggregate=True)
    artifact.write_text(json.dumps({'score': 0.5, 'source_commit': 'a' * 40}))
    assert check(artifact, aggregate=True) == []


def test_every_new_python_file_needs_reuse_declaration(tmp_path):
    check = load_script('check_rsi_public').check
    source = tmp_path / 'demo.py'
    source.write_text('"""Test module."""\n')
    assert check(source)
    source.write_text('"""Test module.\n\nreuse-checked: existing code.\n"""\n')
    assert check(source) == []


def test_export_rejects_a_valid_label_on_an_incomplete_budget(tmp_path):
    export = load_script('export_rsi_results').export
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'summary.json').write_text(json.dumps(dict(
        rows=[dict(seed=0, arm='guided', status='valid', selection_calls=1, holdout_calls=0)],
        config={'seeds': [0], 'selection_budget': 28, 'rounds': 3})))
    source = tmp_path / 'source.json'
    source.write_text(json.dumps({'source_commit': 'a' * 40}))
    import pytest
    with pytest.raises(ValueError, match='matched-budget'):
        export(run, source, tmp_path / 'public')


def test_export_allowlists_aggregates_without_raw_prompt_fields(tmp_path):
    export = load_script('export_rsi_results').export
    run = tmp_path / 'run'
    run.mkdir()
    row = dict(seed=0, arm='guided', status='invalid', reason='http_429',
               selection_calls=1, holdout_calls=0, final_holdout_score=None,
               final_holdout_verdict_accuracy=None, selected_spearman=None,
               output='must not publish', body='must not publish')
    (run / 'summary.json').write_text(json.dumps(dict(rows=[row], config={'seeds': [0]},
        evidence='invalid run', spearman_scope='selected only')))
    (run / 'receipts.jsonl').write_text(json.dumps(dict(model='approved-model', purpose='selection')) + '\n')
    source = tmp_path / 'source.json'
    source.write_text(json.dumps({'source_commit': 'a' * 40}))
    first = tmp_path / 'first'
    public = export(run, source, first)
    assert public['rows'][0]['status'] == 'invalid'
    assert 'output' not in public['rows'][0]
    assert 'body' not in public['rows'][0]
    assert all(row['valid_seeds'] == 0 for row in public['aggregates'])
    assert load_script('check_rsi_public').check(first / 'summary.json', aggregate=True) == []
    second = tmp_path / 'second'
    export(run, source, second)
    assert (first / 'summary.json').read_bytes() == (second / 'summary.json').read_bytes()
    assert (first / 'rounds.csv').read_bytes() == (second / 'rounds.csv').read_bytes()
