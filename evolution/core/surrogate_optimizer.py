"""Budgeted prediction-guided selection over explicit artifact text.

reuse-checked: SkillModule, EvalDataset, fitness metrics, GEPA adapter interface.
"""
from __future__ import annotations

import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Literal, Protocol, Sequence


@dataclass(frozen=True)
class Observation:
    """An external development outcome, never a prediction or holdout label."""

    text: str
    score: float
    feedback: str = ''

    def __post_init__(self):
        if not self.text.strip() or not math.isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError('An observation needs nonempty text and a finite score in [0, 1]')


class Predictor(Protocol):
    def predict(self, texts: Sequence[str]) -> list[float]: ...
    def update(self, observations: Sequence[Observation]) -> None: ...


class TextSurrogate:
    """Sparse word embeddings with cosine-kernel regression over real outcomes.

    This small nonparametric surrogate is not JEPA or a semantic language model.
    """

    def __init__(self):
        self._examples: list[tuple[dict[str, float], float]] = []

    @staticmethod
    def _embed(text: str) -> dict[str, float]:
        counts = Counter(re.findall(r'\w+', text.lower()))
        norm = math.sqrt(sum(value * value for value in counts.values()))
        return {word: value / norm for word, value in counts.items()} if norm else {}

    def update(self, observations: Sequence[Observation]) -> None:
        self._examples.extend((self._embed(item.text), item.score) for item in observations)

    def predict(self, texts: Sequence[str]) -> list[float]:
        predictions = []
        for text in texts:
            embedded = self._embed(text)
            neighbors = sorted((
                (sum(value * other.get(word, 0) for word, value in embedded.items()), score)
                for other, score in self._examples
            ), reverse=True)[:5]
            weights = [(math.exp(8 * similarity), score) for similarity, score in neighbors]
            predictions.append((0.5 + sum(weight * score for weight, score in weights)) /
                               (1 + sum(weight for weight, _ in weights)))
        return predictions


@dataclass(frozen=True)
class RoundRecord:
    round: int
    candidates: tuple[str, ...]
    predicted: tuple[float, ...]
    selected: tuple[int, ...]
    observed: tuple[Observation, ...]
    best: Observation
    evaluations: int


@dataclass(frozen=True)
class SearchResult:
    best: Observation
    observations: tuple[Observation, ...]
    rounds: tuple[RoundRecord, ...]


Proposer = Callable[[str, tuple[Observation, ...], int, random.Random], list[str]]
Evaluator = Callable[[str], Observation]


@dataclass(frozen=True)
class SurrogateOptimizer:
    """Evaluate exactly one baseline plus rounds * top_k candidate batches.

    Callbacks receive development data only. The caller owns task-call accounting,
    candidate constraints, provider failures, and post-search holdout evaluation.
    """

    rounds: int = 3
    population_size: int = 4
    top_k: int = 1
    seed: int = 0

    def __post_init__(self):
        if min(self.rounds, self.population_size, self.top_k) < 1 or self.top_k > self.population_size:
            raise ValueError('Invalid rounds, population_size, or top_k')

    def optimize(
        self,
        initial: str,
        propose: Proposer,
        evaluate: Evaluator,
        *,
        selection: Literal['guided', 'random'] = 'guided',
        predictor: Predictor | None = None,
    ) -> SearchResult:
        if selection not in ('guided', 'random'):
            raise ValueError('selection must be guided or random')
        predictor = predictor if predictor is not None else TextSurrogate()
        rng = random.Random(self.seed)

        def observe(text: str) -> Observation:
            outcome = evaluate(text)
            if outcome.text != text:
                raise ValueError('Evaluator returned an outcome for different text')
            return outcome

        best = observe(initial)
        history = [best]
        records = []
        predictor.update([best])
        for round_index in range(1, self.rounds + 1):
            candidates = propose(best.text, tuple(history), self.population_size, rng)
            if len(candidates) != self.population_size or any(not text.strip() for text in candidates):
                raise ValueError('Proposer returned an invalid population')
            predictions = predictor.predict(candidates)
            if len(predictions) != len(candidates) or any(not math.isfinite(p) for p in predictions):
                raise ValueError('Predictor returned invalid predictions')
            order = list(range(len(candidates)))
            rng.shuffle(order)
            if selection == 'guided':
                order.sort(key=lambda index: predictions[index], reverse=True)
            selected = tuple(order[:self.top_k])
            observed = tuple(observe(candidates[index]) for index in selected)
            for outcome in observed:
                if outcome.score > best.score:
                    best = outcome
            history.extend(observed)
            records.append(RoundRecord(
                round_index, tuple(candidates), tuple(predictions), selected,
                observed, best, len(history),
            ))
            predictor.update(observed)
        return SearchResult(best, tuple(history), tuple(records))


def spearman(predicted: Sequence[float], observed: Sequence[float]) -> float | None:
    """Return Pearson correlation of average ranks, or None for undefined data."""
    if len(predicted) != len(observed):
        raise ValueError('Prediction and observation counts differ')
    if len(predicted) < 2:
        return None

    def ranks(values):
        order = sorted(range(len(values)), key=values.__getitem__)
        ranked = [0.0] * len(values)
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and values[order[end]] == values[order[start]]:
                end += 1
            for index in order[start:end]:
                ranked[index] = (start + end - 1) / 2
            start = end
        return ranked

    x, y = ranks(predicted), ranks(observed)
    center = (len(x) - 1) / 2
    numerator = sum((a - center) * (b - center) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a - center) ** 2 for a in x) * sum((b - center) ** 2 for b in y))
    return numerator / denominator if denominator else None
