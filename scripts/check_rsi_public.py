"""Check new public files for private paths and required reuse declarations.

reuse-checked: external importer privacy rules and z0evals allowlist invariants.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path


PRIVATE = re.compile(r'/(?:m[n]t|home/kv[n]|workspac[e])(?:/|\b)|ze[r]0', re.IGNORECASE)
RAW_KEYS = {'output', 'task_input', 'body', 'feedback', 'messages', 'content', 'response', 'raw_response'}


def check(path: Path, aggregate=False):
    text = path.read_text()
    errors = []
    if PRIVATE.search(text):
        errors.append(f'{path}: private path or host identifier')
    if path.suffix == '.py' and 'reuse-checked:' not in (ast.get_docstring(ast.parse(text)) or ''):
        errors.append(f'{path}: missing reuse-checked module docstring')
    if aggregate and path.suffix == '.json':
        def visit(value):
            if isinstance(value, dict):
                if RAW_KEYS & value.keys():
                    errors.append(f'{path}: raw prompt or response field')
                for item in value.values():
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)
        visit(json.loads(text))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='0a929e3aa20e15cf04dc7c28492a7d41a5139125')
    parser.add_argument('--aggregates', type=Path)
    args = parser.parse_args()
    changed = subprocess.check_output(['git', 'diff', '--name-only', '--diff-filter=ACMR', args.base, '--'], text=True).splitlines()
    errors = []
    for name in changed:
        path = Path(name)
        if path.is_file():
            errors.extend(check(path, aggregate=bool(args.aggregates and path.is_relative_to(args.aggregates))))
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'public boundary: {len(changed)} changed files checked')


if __name__ == '__main__':
    main()
