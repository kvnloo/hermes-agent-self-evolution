"""Contracts for budgeted text selection.

reuse-checked: tests/core/test_constraints.py and existing EvalDataset tests.
"""
import math

import pytest

from evolution.core.surrogate_optimizer import (
    Observation,
    SurrogateOptimizer,
    TextSurrogate,
    spearman,
)


def test_guided_predicts_before_evaluation_and_updates_only_real_outcomes():
    class Predictor:
        def __init__(self):
            self.seen = []
            self.prediction_histories = []

        def predict(self, texts):
            self.prediction_histories.append(tuple(self.seen))
            return [float(text) / 10 for text in texts]

        def update(self, observations):
            self.seen.extend(observations)

    predictor = Predictor()
    proposed = []

    def propose(incumbent, history, count, rng):
        proposed.append((incumbent, tuple(history)))
        return ['1', '9', '3', '8']

    evaluated = []

    def evaluate(text):
        evaluated.append(text)
        return Observation(text, float(text) / 10, 'real')

    result = SurrogateOptimizer(rounds=2, population_size=4, top_k=2, seed=7).optimize(
        '0', propose, evaluate, predictor=predictor,
    )
    assert evaluated == ['0', '9', '8', '9', '8']
    assert len(result.observations) == 5
    assert result.best.text == '9'
    assert len(predictor.prediction_histories[0]) == 1
    assert len(predictor.prediction_histories[1]) == 3
    assert predictor.seen == list(result.observations)
    assert proposed[1][0] == '9'
    assert len(proposed[1][1]) == 3
    assert [r.predicted for r in result.rounds] == [(0.1, 0.9, 0.3, 0.8)] * 2
    assert result.rounds[1].evaluations == 5


def test_random_arm_has_same_budget_and_is_seeded():
    def run(arm):
        return SurrogateOptimizer(rounds=3, population_size=4, top_k=2, seed=11).optimize(
            '0', lambda *args: ['1', '2', '3', '4'],
            lambda text: Observation(text, int(text) / 10), selection=arm,
        )

    guided = run('guided')
    random = run('random')
    assert len(guided.observations) == len(random.observations) == 7
    assert random == run('random')
    assert [r.selected for r in random.rounds] != [r.selected for r in guided.rounds]


@pytest.mark.parametrize('kwargs', [
    {'rounds': 0}, {'population_size': 0}, {'top_k': 0},
    {'population_size': 2, 'top_k': 3},
])
def test_invalid_search_budget_rejected(kwargs):
    with pytest.raises(ValueError):
        SurrogateOptimizer(**kwargs)


def test_bad_candidate_pool_cannot_silently_reduce_budget():
    with pytest.raises(ValueError, match='population'):
        SurrogateOptimizer(rounds=1, population_size=4).optimize(
            'a', lambda *args: ['b'], lambda text: Observation(text, 0.5),
        )


def test_nonfinite_or_out_of_range_observations_rejected():
    for score in (math.nan, math.inf, -0.1, 1.1):
        with pytest.raises(ValueError):
            Observation('x', score)


def test_evaluator_must_score_the_selected_text():
    with pytest.raises(ValueError, match='text'):
        SurrogateOptimizer(rounds=1).optimize(
            'a', lambda *args: ['b', 'c', 'd', 'e'],
            lambda text: Observation('different', 0.5),
        )


def test_predictor_must_return_one_finite_score_per_candidate():
    class BadPredictor:
        def update(self, observations):
            pass

        def predict(self, texts):
            return [float('nan')]

    with pytest.raises(ValueError, match='prediction'):
        SurrogateOptimizer(rounds=1).optimize(
            'a', lambda *args: ['b', 'c', 'd', 'e'],
            lambda text: Observation(text, 0.5), predictor=BadPredictor(),
        )


def test_text_surrogate_learns_from_observed_feedback():
    predictor = TextSurrogate()
    assert predictor.predict(['alpha', 'beta']) == [0.5, 0.5]
    predictor.update([Observation('alpha', 1), Observation('beta', 0)])
    a, b = predictor.predict(['alpha', 'beta'])
    assert 0 <= b < a <= 1
    assert TextSurrogate().predict(['alpha']) == [0.5]


def test_spearman_handles_ties_and_undefined_correlations():
    assert spearman([1, 1, 2], [2, 2, 3]) == pytest.approx(1)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1)
    assert spearman([1, 1], [2, 3]) is None
    assert spearman([1], [2]) is None
    with pytest.raises(ValueError):
        spearman([1], [1, 2])
