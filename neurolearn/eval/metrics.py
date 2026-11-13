"""Metrics (P11 §5) and the clustered bootstrap (§5.1; D-40, D-49).

Scoring is a pure function of a question record from ``truth.json`` and the reply text: no model
judges another model. The bootstrap resamples clusters of questions (a question's cluster is the
object its answer belongs to, ``synthetic_user.question_cluster``), never single questions,
because the same fact is asked at several lags, in both recall conditions and inside two-hop
questions.

Three procedures are defined here and nowhere else:

* ``paired_cluster_bootstrap``: the 95% interval of one difference between two systems on one
  seed (§5.1 "paired cluster bootstrap"). "Above baseline" means its interval excludes zero on
  every seed.
* ``not_worse_guard``: the "not worse on any category" guard: per category the difference
  averaged over seeds, a one-sided bootstrap p-value, Holm's step-down correction across the
  categories at a family-wise 5% (D-49).
* ``gk_evaluate``: the kill criterion G-K (D-41): the primary metric and the guard.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Hashable, Iterable, Iterator, Mapping, Sequence

import numpy as np

from neurolearn.eval.synthetic_user import style_check

GK_PRIMARY = ("jargon", "style", "multi")  # D-41: the categories that justify parametric memory
GK_GUARD = ("fact", "jargon", "multi", "style", "updated", "tool", "anchor")  # P11 §6
MIN_CLUSTERS = 10  # §5.1: below this a category is reported without pass/fail and left out of a guard
LEVEL = 0.95
ALPHA = 0.05
DRAWS = 10_000  # §5.1, one comparison
GUARD_DRAWS = 100_000  # §5.1, the guard
_ZERO_TOL = 1e-12  # a resampled difference within this of zero counts as zero (float rounding)

# ------------------------------------------------------------------------ scoring ---

_DIGIT_GROUP = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")


def fold(text: str) -> str:
    """Case, punctuation and whitespace folded (§5): NFKC, lower case, thousands separators
    between digits dropped, every other Unicode punctuation character replaced by a space,
    whitespace collapsed. ``"Brazik's port: 10,234."`` becomes ``"brazik s port 10234"``."""
    t = _DIGIT_GROUP.sub("", unicodedata.normalize("NFKC", text).lower())
    t = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in t)
    return " ".join(t.split())


def _contains(tokens: Sequence[str], sub: Sequence[str]) -> bool:
    n, m = len(tokens), len(sub)
    return m > 0 and any(tokens[i : i + m] == list(sub) for i in range(n - m + 1))


def exact_match(gold: str, reply: str) -> bool:
    """Normalised exact match of the gold answer *within* the reply: the folded gold is a
    contiguous run of whole tokens of the folded reply. An empty gold never matches."""
    return _contains(fold(reply).split(), fold(gold).split())


def token_f1(gold: str, reply: str) -> float:
    """Token F1 between the gold answer and the best window of the reply of the gold's length.

    Replies are free text, so F1 against the whole reply would measure brevity, not content;
    the window keeps the definition parallel to "exact match within the reply" while giving
    partial credit (``decaf mint`` for ``decaf mint tea`` scores 0.8).
    """
    g, r = fold(gold).split(), fold(reply).split()
    if not g or not r:
        return 0.0
    gc, m = Counter(g), len(g)
    best = 0.0
    for i in range(max(1, len(r) - m + 1)):
        w = r[i : i + m]
        common = sum((Counter(w) & gc).values())
        if common:
            p, q = common / len(w), common / m
            best = max(best, 2 * p * q / (p + q))
    return best


@dataclass(frozen=True)
class Score:
    correct: bool  # exact match within the reply; for `style`, the rule's string check
    f1: float | None  # None for `style` (no gold string)
    stale: bool | None  # `updated` only: the old value is in the reply and the new one is not


def score(question: Mapping[str, Any], reply: str) -> Score:
    """Score one reply to one evaluation question (§5 accuracy, stale-answer rate, style)."""
    if question["category"] == "style":
        return Score(style_check(question["check"], reply), None, None)
    ok = exact_match(question["gold"], reply)
    stale = None
    if question["category"] == "updated":
        stale = exact_match(question["old_value"], reply) and not ok
    return Score(ok, token_f1(question["gold"], reply), stale)


def degenerate(reply: str, ended: bool) -> bool:
    """Rollout health (P03 §5): empty reply, no end-of-turn within the limit, or a 4-gram (of
    folded words) that occurs more than 3 times."""
    words = fold(reply).split()
    if not words or not ended:
        return True
    grams = Counter(tuple(words[i : i + 4]) for i in range(len(words) - 3))
    return any(c > 3 for c in grams.values())


# ---------------------------------------------------------------- the bootstrap ---


@dataclass(frozen=True)
class Interval:
    """A point estimate with its bootstrap interval, and what it rests on."""

    estimate: float
    low: float
    high: float
    n: int  # questions
    clusters: int
    level: float = LEVEL

    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0

    def fmt(self, digits: int = 3, signed: bool = False) -> str:
        f = f"{{:{'+' if signed else ''}.{digits}f}}"
        return f"{f.format(self.estimate)} [{f.format(self.low)}, {f.format(self.high)}]"


def cluster_sums(values: Sequence[float], clusters: Sequence[Hashable]) -> tuple[np.ndarray, np.ndarray]:
    """Per-cluster sum of ``values`` and question count, clusters in sorted-label order."""
    if len(values) != len(clusters):
        raise ValueError("values and clusters differ in length")
    if not len(values):
        raise ValueError("no questions")
    acc: dict[Hashable, list[float]] = {}
    for v, c in zip(values, clusters):
        if not math.isfinite(float(v)):
            raise ValueError(f"non-finite value {v!r}")
        s = acc.setdefault(c, [0.0, 0.0])
        s[0] += float(v)
        s[1] += 1.0
    keys = sorted(acc, key=lambda k: (type(k).__name__, str(k)))
    return np.array([acc[k][0] for k in keys]), np.array([acc[k][1] for k in keys])


def _rng(seed: int | Sequence[int]) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed))


def _blocks(k: int, draws: int) -> Iterator[int]:
    block = max(1, min(draws, 4_000_000 // max(k, 1)))
    left = draws
    while left:
        b = min(block, left)
        yield b
        left -= b


def _weight_blocks(rng: np.random.Generator, k: int, draws: int) -> Iterator[np.ndarray]:
    """Bootstrap draws as cluster multiplicities, in memory-bounded blocks. Row ``b`` says how
    often each of the ``k`` clusters was drawn (with replacement, ``k`` draws) in draw ``b``."""
    for b in _blocks(k, draws):
        idx = rng.integers(0, k, size=(b, k)) + (np.arange(b) * k)[:, None]
        yield np.bincount(idx.ravel(), minlength=b * k).reshape(b, k).astype(np.float64)


def _sign_blocks(rng: np.random.Generator, k: int, draws: int) -> Iterator[np.ndarray]:
    """Rademacher weights (each cluster's sign flipped with probability 1/2), in blocks."""
    for b in _blocks(k, draws):
        yield (rng.integers(0, 2, size=(b, k)) * 2 - 1).astype(np.float64)


def _resampled_means(sums: np.ndarray, counts: np.ndarray, draws: int, rng: np.random.Generator) -> np.ndarray:
    return np.concatenate([(w @ sums) / (w @ counts) for w in _weight_blocks(rng, len(sums), draws)])


def _interval(estimate: float, stats: np.ndarray, n: int, k: int, level: float) -> Interval:
    lo, hi = np.quantile(stats, [(1 - level) / 2, 1 - (1 - level) / 2])
    return Interval(float(estimate), float(lo), float(hi), n, k, level)


def cluster_bootstrap_mean(values: Sequence[float], clusters: Sequence[Hashable], draws: int = DRAWS,
                           seed: int | Sequence[int] = 0, level: float = LEVEL) -> Interval:
    """Mean of ``values`` with a percentile interval from resampling whole clusters. With every
    question in its own cluster this is the ordinary per-question bootstrap."""
    sums, counts = cluster_sums(values, clusters)
    stats = _resampled_means(sums, counts, draws, _rng(seed))
    return _interval(sums.sum() / counts.sum(), stats, int(counts.sum()), len(sums), level)


def paired_cluster_bootstrap(x_sys: Sequence[float], x_base: Sequence[float], clusters: Sequence[Hashable],
                             draws: int = DRAWS, seed: int | Sequence[int] = 0, level: float = LEVEL) -> Interval:
    """Interval of ``mean(x_sys) − mean(x_base)`` over the same questions (§5.1): each draw takes
    every question of each drawn cluster for both systems."""
    if len(x_sys) != len(x_base):
        raise ValueError("the two systems must be scored on the same questions")
    d = np.asarray(x_sys, dtype=np.float64) - np.asarray(x_base, dtype=np.float64)
    return cluster_bootstrap_mean(d, clusters, draws, seed, level)


@dataclass(frozen=True)
class Paired:
    """One category on one seed: scores of the system and the baseline on the same questions."""

    x_sys: tuple[float, ...]
    x_base: tuple[float, ...]
    clusters: tuple[Hashable, ...]

    def __post_init__(self) -> None:
        if not (len(self.x_sys) == len(self.x_base) == len(self.clusters)):
            raise ValueError("x_sys, x_base and clusters must have the same length")

    @property
    def n(self) -> int:
        return len(self.clusters)

    @property
    def k(self) -> int:
        return len(set(self.clusters))

    def diff(self) -> float:
        return float(np.mean(self.x_sys) - np.mean(self.x_base))


def paired_mean_of_categories(data: Mapping[str, Paired], draws: int = DRAWS, seed: int | Sequence[int] = 0,
                              level: float = LEVEL) -> Interval:
    """Interval of the difference in the *unweighted mean of per-category accuracies* (the G-K
    primary metric, D-41), recomputed on every draw. Clusters are resampled within each category
    (stratified), so every draw contains every category. The categories' clusters must be
    disjoint; otherwise one cluster would be resampled twice independently."""
    if not data:
        raise ValueError("no categories")
    seen: dict[Hashable, str] = {}
    for cat, p in data.items():
        for c in set(p.clusters):
            if c in seen and seen[c] != cat:
                raise ValueError(f"cluster {c!r} is in both {seen[c]!r} and {cat!r}")
            seen[c] = cat
    rng = _rng(seed)
    stats = np.zeros(draws)
    for cat in sorted(data):
        p = data[cat]
        sums, counts = cluster_sums(np.asarray(p.x_sys) - np.asarray(p.x_base), p.clusters)
        stats += _resampled_means(sums, counts, draws, rng)
    stats /= len(data)
    est = float(np.mean([data[c].diff() for c in data]))
    return _interval(est, stats, sum(p.n for p in data.values()), sum(p.k for p in data.values()), level)


def p_below_zero(per_seed: Sequence[Paired], draws: int = GUARD_DRAWS, seed: int | Sequence[int] = 0
                 ) -> tuple[float, float]:
    """One-sided p-value against "not worse" (D-54): a wild cluster bootstrap with the null
    imposed. Under "no difference" each cluster's summed paired difference is as likely to have
    either sign, so each draw flips every cluster's sign at random (independently per seed) and
    recomputes the seed-averaged difference. ``p = (count + 1) / (draws + 1)`` where ``count`` is
    the number of draws at or below the observed difference. Small values mean the system is
    worse than the baseline. Returns (seed-averaged difference, p).

    The percentile version (resample clusters, count draws at or above zero) was specified
    first; with 10 to 40 clusters per seed it failed I-P11-9 (D-54)."""
    if not per_seed:
        raise ValueError("no seeds")
    rng = _rng(seed)
    observed = float(np.mean([p.diff() for p in per_seed]))
    total = np.zeros(draws)
    for p in per_seed:
        sums, counts = cluster_sums(np.asarray(p.x_sys) - np.asarray(p.x_base), p.clusters)
        n = counts.sum()
        total += np.concatenate([(e @ sums) / n for e in _sign_blocks(rng, len(sums), draws)])
    total /= len(per_seed)
    count = int(np.count_nonzero(total <= observed + _ZERO_TOL))
    return observed, (count + 1) / (draws + 1)


def p_below_zero_percentile(per_seed: Sequence[Paired], draws: int = GUARD_DRAWS, seed: int | Sequence[int] = 0
                            ) -> tuple[float, float]:
    """The first-specified p-value, kept so I-P11-9 can report its error rate (D-54): clusters
    resampled within each seed, ``p`` = share of draws whose seed-averaged difference is at or
    above zero. Not used by the guard."""
    if not per_seed:
        raise ValueError("no seeds")
    rng = _rng(seed)
    total = np.zeros(draws)
    for p in per_seed:
        sums, counts = cluster_sums(np.asarray(p.x_sys) - np.asarray(p.x_base), p.clusters)
        total += _resampled_means(sums, counts, draws, rng)
    total /= len(per_seed)
    count = int(np.count_nonzero(total >= -_ZERO_TOL))
    return float(np.mean([p.diff() for p in per_seed])), (count + 1) / (draws + 1)


def holm(p_values: Mapping[str, float], alpha: float = ALPHA) -> dict[str, tuple[float, bool]]:
    """Holm's step-down procedure. Returns, per hypothesis, (its step's threshold, rejected).
    The hypothesis with the ``i``-th smallest p-value (0-based) has threshold
    ``alpha / (m − i)``; testing stops at the first that is not rejected, and no later one is
    rejected whatever its p-value."""
    order = sorted(p_values, key=lambda k: (p_values[k], k))
    m, out, stopped = len(order), {}, False
    for i, k in enumerate(order):
        thr = alpha / (m - i)
        rejected = not stopped and p_values[k] <= thr
        stopped = stopped or not rejected
        out[k] = (thr, rejected)
    return out


@dataclass(frozen=True)
class GuardRow:
    category: str
    included: bool
    estimate: float | None  # seed-averaged difference (system − baseline)
    p_value: float | None
    threshold: float | None  # Holm threshold at this category's step
    rejected: bool  # "worse" after correction
    clusters: tuple[int, ...]  # per seed
    questions: tuple[int, ...]  # per seed
    note: str = ""


@dataclass(frozen=True)
class GuardResult:
    rows: tuple[GuardRow, ...]
    fails: bool
    alpha: float

    def row(self, category: str) -> GuardRow:
        return next(r for r in self.rows if r.category == category)


def not_worse_guard(per_seed: Sequence[Mapping[str, Paired]], categories: Sequence[str], alpha: float = ALPHA,
                    draws: int = GUARD_DRAWS, seed: int = 0, min_clusters: int = MIN_CLUSTERS,
                    method: str = "wild") -> GuardResult:
    """"Not worse on any category" (§5.1, D-49, D-54): one family of one-sided tests on the
    seed-averaged differences, Holm-corrected at family-wise ``alpha``. A category with fewer
    than ``min_clusters`` clusters on any seed, or absent on a seed, is reported and left out.
    ``method="percentile"`` uses the first-specified p-value; it exists for I-P11-9 only."""
    test = {"wild": p_below_zero, "percentile": p_below_zero_percentile}[method]
    rows: dict[str, GuardRow] = {}
    p_values: dict[str, float] = {}
    for ci, cat in enumerate(categories):
        present = [s.get(cat) for s in per_seed]
        ks = tuple(p.k if p else 0 for p in present)
        ns = tuple(p.n if p else 0 for p in present)
        if not per_seed or any(p is None for p in present):
            rows[cat] = GuardRow(cat, False, None, None, None, False, ks, ns, "no questions on some seed")
            continue
        if min(ks) < min_clusters:
            est = float(np.mean([p.diff() for p in present]))
            rows[cat] = GuardRow(cat, False, est, None, None, False, ks, ns,
                                 f"fewer than {min_clusters} clusters on some seed: reported, not tested")
            continue
        est, pv = test(present, draws, (seed, ci))
        p_values[cat] = pv
        rows[cat] = GuardRow(cat, True, est, pv, None, False, ks, ns)
    for cat, (thr, rej) in holm(p_values, alpha).items():
        r = rows[cat]
        rows[cat] = GuardRow(cat, True, r.estimate, r.p_value, thr, rej, r.clusters, r.questions, r.note)
    ordered = tuple(rows[c] for c in categories)
    return GuardResult(ordered, any(r.rejected for r in ordered), alpha)


def per_seed_worse(per_seed: Sequence[Mapping[str, Paired]], categories: Sequence[str], draws: int = DRAWS,
                   seed: int = 0, level: float = LEVEL) -> dict[tuple[int, str], Interval]:
    """Uncorrected, descriptive per-seed intervals of every category (§5.1 says to report them,
    marked as descriptive). The naive rule "worse if some interval lies below zero" is what the
    guard replaces; I-P11-9 measures how often it fails by chance."""
    out = {}
    for si, s in enumerate(per_seed):
        for ci, cat in enumerate(categories):
            if cat in s:
                p = s[cat]
                out[(si, cat)] = paired_cluster_bootstrap(p.x_sys, p.x_base, p.clusters, draws, (seed, si, ci), level)
    return out


# ------------------------------------------------------------------ G-K (D-41) ---


@dataclass(frozen=True)
class GKResult:
    primary: tuple[Interval | None, ...]  # per seed; None when a primary category is missing
    primary_passes: bool  # interval excludes zero on the positive side on every seed
    guard: GuardResult
    passes: bool
    notes: tuple[str, ...]


def gk_evaluate(per_seed: Sequence[Mapping[str, Paired]], draws: int = DRAWS, guard_draws: int = GUARD_DRAWS,
                seed: int = 0) -> GKResult:
    """G-K: ``full`` above ``a`` on the mean of the recall-on accuracies of `jargon`, `style`,
    `multi` (interval excludes zero on every seed), and not worse on any guarded category."""
    notes = []
    primary: list[Interval | None] = []
    for si, s in enumerate(per_seed):
        missing = [c for c in GK_PRIMARY if c not in s]
        if missing:
            primary.append(None)
            notes.append(f"seed index {si}: no questions in {missing}; the primary metric is undefined")
            continue
        primary.append(paired_mean_of_categories({c: s[c] for c in GK_PRIMARY}, draws, (seed, si, 99)))
    ok = bool(primary) and all(iv is not None and iv.low > 0 for iv in primary)
    guard = not_worse_guard(per_seed, GK_GUARD, draws=guard_draws, seed=seed)
    return GKResult(tuple(primary), ok, guard, ok and not guard.fails, tuple(notes))


def paired_by_category(ans_sys: Iterable[Mapping[str, Any]], ans_base: Iterable[Mapping[str, Any]],
                       recall: bool, phase: str = "after") -> dict[str, Paired]:
    """Pair two runs' answer records question by question (same ``qid``, recall condition and
    phase) and group them by category. Raises if the two runs were not asked the same questions
    (I-P11-5)."""
    def index(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
        return {r["qid"]: r for r in rows if r["recall"] == recall and r["phase"] == phase}

    a, b = index(ans_sys), index(ans_base)
    if set(a) != set(b):
        raise ValueError(f"the runs were asked different questions ({len(set(a) ^ set(b))} differ)")
    groups: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    for qid in sorted(a):
        ra, rb = a[qid], b[qid]
        if (ra["category"], ra["cluster"]) != (rb["category"], rb["cluster"]):
            raise ValueError(f"question {qid} differs between the runs")
        groups[ra["category"]].append((float(ra["correct"]), float(rb["correct"]), ra["cluster"]))
    return {cat: Paired(tuple(x for x, _, _ in v), tuple(y for _, y, _ in v), tuple(c for _, _, c in v))
            for cat, v in sorted(groups.items())}


# ---------------------------------------------------------- aggregation for reports ---


def _mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / len(xs))


def metric_rows(answers: Sequence[Mapping[str, Any]], system: str) -> list[dict[str, Any]]:
    """``metrics.jsonl`` rows from answer records: one row per (cycle, system, metric, category,
    lag, recall) with value and n (§7). ``cycle`` and ``lag`` also take the value ``"all"``.

    Metric names: ``accuracy``, ``f1``, ``truncated`` (no end-of-turn within the limit),
    ``recall_hit`` (recall on: every gold source entry was among the recalled chunks),
    ``stale`` (`updated`), ``degenerate`` (`anchor`), ``plasticity`` (lag 0, recall off, after
    consolidation); ``accuracy_pre`` and ``f1_pre`` for the same-day questions asked before
    consolidation; ``accuracy_bare`` for the bare base's reference answers. Sub-categories
    ``forgotten:once``, ``forgotten:twice`` and ``style:<rule kind>`` are added.
    """
    groups: dict[tuple[str, Any, str, Any, bool], list[Mapping[str, Any]]] = defaultdict(list)
    for r in answers:
        cats = [r["category"]]
        if r["category"] == "forgotten":
            cats.append("forgotten:twice" if r.get("stated_twice") else "forgotten:once")
        if r["category"] == "style" and r.get("rule_kind"):
            cats.append(f"style:{r['rule_kind']}")
        lag = r["lag"] if r["lag"] is not None else "-"
        for cat in cats:
            for cyc in (r["cycle"], "all"):
                for lg in (lag, "all"):
                    groups[(r["phase"], cyc, cat, lg, bool(r["recall"]))].append(r)
    suffix = {"after": "", "before": "_pre", "bare": "_bare"}
    out: list[dict[str, Any]] = []
    for (phase, cyc, cat, lag, recall), rows in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        sfx = suffix[phase]

        def add(metric: str, values: Sequence[float]) -> None:
            if values:
                out.append({"cycle": cyc, "system": system, "metric": metric, "category": cat, "lag": lag,
                            "recall": recall, "value": _mean(values), "n": len(values)})

        add("accuracy" + sfx, [float(r["correct"]) for r in rows])
        if phase == "bare":
            continue
        add("f1" + sfx, [r["f1"] for r in rows if r["f1"] is not None])
        if phase != "after":
            continue
        add("truncated", [float(not r["ended"]) for r in rows])
        if recall:
            add("recall_hit", [float(r["recall_hit"]) for r in rows if r.get("recall_hit") is not None])
        if cat == "updated":
            add("stale", [float(r["stale"]) for r in rows if r.get("stale") is not None])
        if cat == "anchor":
            add("degenerate", [float(r["degenerate"]) for r in rows])
        if lag == 0 and not recall:
            add("plasticity", [float(r["correct"]) for r in rows])
    return out


def retention_rows(answers: Sequence[Mapping[str, Any]], system: str, lags: Sequence[int],
                   categories: Sequence[str] = ("fact", "jargon")) -> list[dict[str, Any]]:
    """Retention (§5): recall-off accuracy at lag L divided by recall-off accuracy at lag 0, on
    the facts that have both questions (after consolidation). ``value`` is None when the lag-0
    accuracy of those facts is 0; ``n`` is the number of facts."""
    out = []
    for cat in categories:
        by_fact: dict[str, dict[int, float]] = defaultdict(dict)
        for r in answers:
            if r["category"] == cat and r["phase"] == "after" and not r["recall"] and r["lag"] is not None:
                by_fact[r["answer_fact_id"]][r["lag"]] = float(r["correct"])
        for lag in lags:
            if lag == 0:
                continue
            both = [v for v in by_fact.values() if 0 in v and lag in v]
            if not both:
                continue
            a0, al = _mean([v[0] for v in both]), _mean([v[lag] for v in both])
            out.append({"cycle": "all", "system": system, "metric": "retention", "category": cat, "lag": lag,
                        "recall": False, "value": (al / a0) if a0 > 0 else None, "n": len(both),
                        "acc_lag0": a0, "acc_lag": al})
    return out


def above_chance_p(correct: Sequence[float], chance: Sequence[float], clusters: Sequence[Hashable],
                   draws: int = DRAWS, seed: int | Sequence[int] = 0) -> tuple[float, float, float]:
    """For I-P11-4: one-sided cluster-bootstrap p-value that accuracy exceeds the chance rate.
    Returns (accuracy, mean chance rate, p); "at chance" means p > 0.05."""
    d = np.asarray(correct, dtype=np.float64) - np.asarray(chance, dtype=np.float64)
    sums, counts = cluster_sums(d, clusters)
    stats = _resampled_means(sums, counts, draws, _rng(seed))
    p = (int(np.count_nonzero(stats <= _ZERO_TOL)) + 1) / (draws + 1)
    return float(np.mean(correct)), float(np.mean(chance)), p
