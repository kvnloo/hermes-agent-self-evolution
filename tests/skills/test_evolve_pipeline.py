"""End-to-end tests for ``evolve()`` that run without any network or LLM call.

Every other test in this repo exercises helpers in isolation, so none of them
reaches the call sites inside ``evolve()`` where the pipeline actually broke
(constraint validation on the bare body, the GEPA constructor, reading the
evolved text back, the success verdict). These tests drive ``evolve()`` end to
end against:

* a throwaway Hermes repo holding one valid SKILL.md,
* a fixed golden dataset (train/val/holdout JSONL),
* an offline LM whose answer depends only on whether a marker string appears
  in the prompt, and
* stub ``compile()`` methods that do what GEPA and MIPROv2 do to a program:
  rewrite each named predictor's ``signature.instructions`` (or attach demos).

The real ``dspy.GEPA.__init__`` still runs, so its argument validation is
exercised.
"""

import io
import json
import re
from pathlib import Path

import dspy
import pytest
from dspy.clients.engines.dummy_engine import DummyEngine
from dspy.lm15 import Message, Response, TextPart, Usage
from dspy.utils.dummies import DummyLM
from rich.console import Console

from evolution.skills import evolve_skill

MARKER = "ZEBRA-MARKER"

SKILL_MD = """---
name: demo-skill
description: Demo skill used by the offline pipeline tests.
---
# Demo skill

Use this skill when reviewing a pull request before merge.

## Procedure

1. Read the task and the linked issue.
2. Check the merge state:

```bash
gh pr view --json mergeStateStatus,reviewDecision
```

3. Report what blocks the merge, if anything.

## Lessons

- Gate the commit on the test command's exit code, never on grepping its output.
- Do not force push a shared branch; open a follow-up PR instead.
"""

EXPECTED = "answer the review question about the repository carefully and EVOLVED"


class _MarkerEngine(DummyEngine):
    def _complete_messages(self, messages):
        return Response(
            id=None,
            model="dummy",
            message=Message.assistant([TextPart(self.owner.answer_for(messages))]),
            finish_reason="stop",
            usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0),
        )


class MarkerLM(DummyLM):
    """Answers 'EVOLVED answer' iff MARKER appears anywhere in the prompt."""

    def __init__(self, *args, **kwargs):
        super().__init__([{}])
        self._engine_spec = _MarkerEngine(self)

    def answer_for(self, messages):
        text = "\n".join(m["content"] for m in messages)
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else text
        fields = {"output": "EVOLVED answer" if MARKER in text else "baseline answer"}
        if re.search(r"\bplan\b", system):
            fields = {"plan": "p", **fields}
        elif re.search(r"\breasoning\b", system):
            fields = {"reasoning": "r", **fields}
        return self._format_answer_fields(fields)

    def copy(self, **kwargs):
        return self


def _rewrite_instructions(student, fn):
    program = student.deepcopy()
    for _name, predictor in program.named_predictors():
        predictor.signature = predictor.signature.with_instructions(fn(predictor.signature.instructions))
    return program


def _attach_marker_demo(student):
    program = student.deepcopy()
    for _name, predictor in program.named_predictors():
        predictor.demos = [dspy.Example(task_input=f"demo {MARKER}", output="EVOLVED answer")]
    return program


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    skill_dir = tmp_path / "hermes" / "skills" / "demo" / "demo-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(SKILL_MD)

    calls = {"gepa": [], "mipro": []}
    state = {"mutate": lambda student: student.deepcopy()}

    def fake_gepa_compile(self, student, *, trainset, teacher=None, valset=None):
        calls["gepa"].append(self)
        return state["mutate"](student)

    class FakeMIPROv2:
        def __init__(self, *args, **kwargs):
            calls["mipro"].append(kwargs)

        def compile(self, student, *args, **kwargs):
            return state["mutate"](student)

    monkeypatch.setattr(dspy, "LM", MarkerLM)
    monkeypatch.setattr(dspy.GEPA, "compile", fake_gepa_compile)
    monkeypatch.setattr(dspy, "MIPROv2", FakeMIPROv2)
    buf = io.StringIO()
    monkeypatch.setattr(evolve_skill, "console", Console(file=buf, width=240, force_terminal=False))
    monkeypatch.chdir(tmp_path)
    # Older bases ignore --hermes-repo; the env var points them at the fake repo too.
    monkeypatch.setenv("HERMES_AGENT_REPO", str(tmp_path / "hermes"))

    def run(mutate=None, holdout_n=8):
        if mutate is not None:
            state["mutate"] = mutate
        ds = tmp_path / f"ds{holdout_n}"
        ds.mkdir(exist_ok=True)
        for split, n in (("train", 2), ("val", 2), ("holdout", holdout_n)):
            rows = [json.dumps({"task_input": f"{split} task {i}", "expected_behavior": EXPECTED}) for i in range(n)]
            (ds / f"{split}.jsonl").write_text("\n".join(rows) + "\n")
        evolve_skill.evolve(
            skill_name="demo-skill",
            iterations=2,
            eval_source="golden",
            dataset_path=str(ds),
            hermes_repo=str(tmp_path / "hermes"),
        )
        out = tmp_path / "output" / "demo-skill"
        evolved = sorted(out.glob("**/evolved*.md")) if out.exists() else []
        metrics = sorted(out.glob("*/metrics.json")) if out.exists() else []
        log = buf.getvalue()
        return {
            "failed": (out / "evolved_FAILED.md").exists(),
            "evolved_text": evolved[-1].read_text() if evolved else "",
            "metrics": json.loads(metrics[-1].read_text()) if metrics else None,
            "log": log,
            "declared_win": ("✓ Evolution improved skill" in log)
            or ("✓ Statistically significant improvement" in log),
            "calls": calls,
        }

    return run


def test_unchanged_valid_skill_passes_constraints(pipeline):
    """A valid skill 'evolved' into itself must pass every constraint (#169)."""
    r = pipeline()
    assert not r["failed"], r["log"]
    assert r["metrics"] is not None and r["metrics"]["constraints_passed"] is True


def test_gepa_is_constructed_instead_of_silent_mipro_fallback(pipeline):
    """dspy.GEPA must accept our arguments; no fallback to MIPROv2 (#10, #119)."""
    r = pipeline()
    assert r["calls"]["mipro"] == [], r["log"]
    assert len(r["calls"]["gepa"]) == 1
    assert r["calls"]["gepa"][0].reflection_lm is not None


def test_optimizer_instruction_edit_lands_in_evolved_skill(pipeline):
    """What the optimizer writes to the predictor instructions is the evolved skill (#141, #172)."""
    r = pipeline(lambda s: _rewrite_instructions(s, lambda ins: ins + f"\n- {MARKER}: cite the file you changed."))
    assert MARKER in r["evolved_text"], r["log"]
    assert "gh pr view --json mergeStateStatus" in r["evolved_text"]
    assert r["evolved_text"].count("name: demo-skill") == 1


def test_evolved_body_with_own_frontmatter_is_not_nested(pipeline):
    """If the optimizer emits a whole SKILL.md, the file keeps one frontmatter block (#119 bug 2)."""
    r = pipeline(lambda s: _rewrite_instructions(
        s, lambda ins: "---\nname: demo-skill\ndescription: rewritten\n---\n\n" + ins + f"\n- {MARKER}"
    ))
    assert MARKER in r["evolved_text"], r["log"]
    assert r["evolved_text"].count("name: demo-skill") == 1
    assert r["evolved_text"].count("\n---\n") == 1


def test_reassemble_strips_duplicate_frontmatter():
    from evolution.skills.skill_module import reassemble_skill

    out = reassemble_skill("name: a\ndescription: b", "---\nname: a\ndescription: c\n---\n\n# Body\n")
    assert out.count("name: a") == 1
    assert out.rstrip().endswith("# Body")


def test_unchanged_skill_text_is_not_reported_as_a_win(pipeline):
    """Demo-only gains leave SKILL.md byte-identical; that is not an evolved skill (#141)."""
    r = pipeline(_attach_marker_demo)
    assert r["metrics"] is not None, r["log"]
    assert r["metrics"]["improvement"] > 0  # the scores did move ...
    assert not r["declared_win"], r["log"]  # ... but nothing deployable changed


def test_small_holdout_gain_is_not_declared_a_win(pipeline):
    """+0.08 on 3 holdout examples cannot reach p < 0.05; no win (#135)."""
    r = pipeline(lambda s: _rewrite_instructions(s, lambda ins: ins + f"\n- {MARKER}"), holdout_n=3)
    assert r["metrics"] is not None, r["log"]
    assert r["metrics"]["improvement"] > 0
    assert not r["declared_win"], r["log"]


def test_consistent_gain_on_larger_holdout_is_declared_a_win(pipeline):
    """Positive control: the gates must still let a real, consistent improvement through."""
    r = pipeline(lambda s: _rewrite_instructions(s, lambda ins: ins + f"\n- {MARKER}"), holdout_n=8)
    assert r["metrics"] is not None, r["log"]
    assert r["declared_win"], r["log"]


def test_deleting_commands_and_lessons_fails_constraints(pipeline):
    """Removing the command block and the incident lessons must not pass the gate (#200)."""

    def gut(ins):
        kept = ins.split("## Lessons")[0]
        return re.sub(r"```bash.*?```\n", "", kept, flags=re.S) + f"\n- {MARKER}"

    r = pipeline(lambda s: _rewrite_instructions(s, gut))
    assert r["failed"], r["log"]
    assert re.search(r"✗ content_retention", r["log"]), r["log"]
