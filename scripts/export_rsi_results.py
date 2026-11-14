"""Export allowlisted, aggregate-only RSI experiment evidence.

reuse-checked: compare_optimizers receipts, z0evals pinned-artifact importer.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


ROW_FIELDS = (
    'seed', 'arm', 'status', 'reason', 'selection_calls', 'holdout_calls',
    'final_holdout_score', 'final_holdout_verdict_accuracy', 'selected_spearman',
    'candidate_rejections', 'generation_calls', 'search_seconds', 'provider_seconds',
    'provider_attempts', 'prompt_tokens', 'completion_tokens',
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(run: Path, source: Path, destination: Path):
    summary = json.loads((run / 'summary.json').read_text())
    provenance = json.loads(source.read_text())
    rows = [{field: row.get(field) for field in ROW_FIELDS} for row in summary['rows']]
    config = summary['config']
    for row in summary['rows']:
        if row['status'] == 'valid':
            checkpoints = row.get('checkpoints', [])
            if (not checkpoints or row['selection_calls'] != config['selection_budget'] or
                    len(checkpoints) != config['rounds'] + 1 or
                    row['holdout_calls'] <= 0 or row['holdout_calls'] % len(checkpoints)):
                raise ValueError('A valid row violates the matched-budget publication boundary')
    checkpoints = [dict(seed=row['seed'], arm=row['arm'], **checkpoint)
                   for row in summary['rows'] if row['status'] == 'valid'
                   for checkpoint in row.get('checkpoints', [])]
    receipts = [json.loads(line) for line in (run / 'receipts.jsonl').read_text().splitlines()]
    models = {purpose: sorted({row['model'] for row in receipts if row['purpose'] == purpose})
              for purpose in ('selection', 'generation', 'holdout')}
    aggregates = []
    for arm in ('gepa', 'guided', 'random'):
        subset = [row for row in rows if row['arm'] == arm and row['status'] == 'valid']
        invalid = [row for row in rows if row['arm'] == arm and row['status'] != 'valid']
        item = dict(arm=arm, valid_seeds=len(subset), invalid_seeds=len(invalid))
        for metric in ('final_holdout_score', 'final_holdout_verdict_accuracy', 'selected_spearman',
                       'search_seconds', 'provider_seconds', 'prompt_tokens', 'completion_tokens'):
            values = [row[metric] for row in subset if row[metric] is not None]
            item[metric] = dict(n=len(values), mean=statistics.mean(values) if values else None,
                                min=min(values) if values else None, max=max(values) if values else None)
        aggregates.append(item)
    paired = []
    for control in ('gepa', 'random'):
        differences = []
        for seed in summary['config']['seeds']:
            pair = {row['arm']: row for row in rows if row['seed'] == seed and row['status'] == 'valid'}
            if 'guided' in pair and control in pair:
                differences.append(dict(seed=seed,
                    keyword_delta=pair['guided']['final_holdout_score'] - pair[control]['final_holdout_score'],
                    verdict_delta=pair['guided']['final_holdout_verdict_accuracy'] - pair[control]['final_holdout_verdict_accuracy']))
        paired.append(dict(comparison='guided-minus-' + control, differences=differences,
                           mean_keyword_delta=statistics.mean(row['keyword_delta'] for row in differences) if differences else None,
                           mean_verdict_delta=statistics.mean(row['verdict_delta'] for row in differences) if differences else None))
    raw_files = sorted(path for path in run.rglob('*') if path.is_file() and not path.name.endswith('.tmp'))
    public = dict(schema='rsi.skill-selection.v1', source=provenance, config=summary['config'],
                  models=models, rows=rows, aggregates=aggregates, paired=paired,
                  evidence=summary['evidence'], spearman_scope=summary['spearman_scope'],
                  raw_artifacts=[dict(path=str(path.relative_to(run)), sha256=sha256(path)) for path in raw_files])
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'summary.json').write_text(json.dumps(public, indent=2, allow_nan=False) + '\n')
    columns = ('seed', 'arm', 'selection_calls', 'candidate_id', 'development_score', 'holdout_score', 'verdict_accuracy')
    with (destination / 'rounds.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(checkpoints)
    return public


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export(args.run, args.source, args.output)
