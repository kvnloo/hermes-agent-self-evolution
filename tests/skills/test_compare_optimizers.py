"""End-to-end and boundary contracts for the three-arm experiment.

reuse-checked: CliRunner tests, EvalDataset serialization, GEPA adapter contract.
"""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from evolution.skills import compare_optimizers as experiment


SKILL = '---\nname: demo\ndescription: Review diffs\n---\n\n# Review\nReturn a verdict and specific findings.\n'


def fixtures(root):
    skill = root / 'SKILL.md'
    skill.write_text(SKILL)
    data = root / 'dataset'
    data.mkdir()
    for split in ('train', 'val', 'holdout'):
        rows = [dict(task_input=f'{split}-{index}', expected_behavior='correct review',
                     lineage=f'{split}-{index}', expected_passed=bool(index)) for index in range(2)]
        (data / f'{split}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return skill, data


class FakeClient:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def complete(self, messages, *, purpose, seed, max_tokens):
        self.calls.append(purpose)
        if purpose == 'generation':
            payload = json.loads(messages[-1]['content'])
            return json.dumps({'candidates': [payload['parent'] + f'\n{i}' for i in range(payload['count'])]})
        if purpose == 'holdout':
            assert (self.output / 'search-seal.json').exists()
            sealed = json.loads((self.output / 'search-seal.json').read_text())
            assert len(sealed['searches']) == 9
        task = messages[-1]['content']
        return json.dumps({'passed': task.endswith('1'), 'security_concerns': [],
                           'logic_errors': [], 'suggestions': [], 'summary': 'correct review'})


def test_three_seed_cli_seals_search_before_holdout_and_enforces_equal_budgets(tmp_path, monkeypatch):
    skill, data = fixtures(tmp_path)
    output = tmp_path / 'run'
    fake = FakeClient(output)
    monkeypatch.setattr(experiment, 'ProviderClient', lambda *args, **kwargs: fake)
    result = CliRunner().invoke(experiment.main, [
        '--skill-path', str(skill), '--dataset', str(data), '--output', str(output),
        '--seeds', '0,1,2', '--rounds', '1', '--top-k', '1', '--population-size', '2',
    ])
    assert result.exit_code == 0, (result.output, result.exception)
    summary = json.loads((output / 'summary.json').read_text())
    assert len(summary['rows']) == 9
    assert {row['arm'] for row in summary['rows']} == {'gepa', 'guided', 'random'}
    assert all(row['status'] == 'valid' for row in summary['rows'])
    assert {row['selection_calls'] for row in summary['rows']} == {4}
    assert {row['holdout_calls'] for row in summary['rows']} == {4}
    assert all(row['final_holdout_verdict_accuracy'] == 1 for row in summary['rows'])
    first_audit = fake.calls.index('holdout')
    assert 'generation' not in fake.calls[first_audit:]
    assert 'selection' not in fake.calls[first_audit:]
    assert SKILL == skill.read_text()
    assert not any('holdout' in p.name for p in data.iterdir() if p.name != 'holdout.jsonl')


def test_split_lineage_overlap_is_rejected(tmp_path):
    _, data = fixtures(tmp_path)
    (data / 'holdout.jsonl').write_text((data / 'train.jsonl').read_text())
    with pytest.raises(ValueError, match='overlap'):
        experiment.load_dataset(data)


def test_provider_failure_invalidates_rows_without_fabricated_scores(tmp_path):
    skill, data = fixtures(tmp_path)
    client = Mock()
    client.complete.side_effect = experiment.EvaluationFailure('http_429')
    summary = experiment.run_comparison(skill, data, tmp_path / 'invalid', (0, 1, 2),
                                        1, 1, 2, client, client)
    assert len(summary['rows']) == 9
    assert all(row['status'] == 'invalid' for row in summary['rows'])
    assert all(row['final_holdout_score'] is None for row in summary['rows'])


def test_changed_output_directory_cannot_overwrite_evidence(tmp_path):
    skill, data = fixtures(tmp_path)
    out = tmp_path / 'existing'
    out.mkdir()
    (out / 'important.txt').write_text('keep')
    with pytest.raises(FileExistsError):
        experiment.run_comparison(skill, data, out, (0,), 1, 1, 2, Mock(), Mock())
    assert (out / 'important.txt').read_text() == 'keep'


def test_provider_pacing_uses_observed_token_limit():
    assert experiment.quota_delay({'x-ratelimit-limit-tokens': '8000'}, {'total_tokens': 1000}) == 10
    assert experiment.quota_delay({}, {}) == 2


def test_empty_body_is_not_a_valid_skill_candidate(tmp_path):
    evaluator = experiment.DevelopmentEvaluator(Mock(), 0, 1, tmp_path, 'name: demo\ndescription: review',
                                                 'Review supplied code carefully.', {})
    assert not evaluator.admissible('   ')


def test_provider_backoff_is_bounded_and_receipts_redact_secrets(tmp_path, monkeypatch):
    from openai import RateLimitError
    response = SimpleNamespace(status_code=429, headers={}, request=SimpleNamespace())
    client = experiment.ProviderClient('groq', tmp_path / 'receipts.jsonl')
    client._client = Mock()
    client._client.chat.completions.create.side_effect = RateLimitError('limited', response=response, body={})
    sleeps = []
    monkeypatch.setattr(experiment.time, 'sleep', sleeps.append)
    with pytest.raises(experiment.EvaluationFailure, match='http_429'):
        client.complete([{'role': 'user', 'content': 'task'}], purpose='selection', seed=0, max_tokens=10)
    assert sleeps == [4, 8]
    assert client._client.chat.completions.create.call_count == 3
    receipt = (tmp_path / 'receipts.jsonl').read_text()
    assert receipt.count('http_429') == 3
    assert 'Authorization' not in receipt
    with pytest.raises(experiment.EvaluationFailure):
        client.complete([], purpose='selection', seed=0, max_tokens=10)
    assert client._client.chat.completions.create.call_count == 3
