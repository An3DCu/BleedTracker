"""The measures every queue reports in the upgrades.

Each queue reports its margin over the base rate first. Next come its precision,
the base rate and the share of the pool's lost impressions its pages hold. A page's
lost impressions are its impressions minus next month's, never below zero. The
share sits next to what a random queue of the same size would hold on average.

The per-account queue hands each held-out client slots by its share of the pool's
pages. The paired bootstrap puts an interval on the forest minus the rule inside one
deployment. Both are post-hoc.
"""

import numpy as np
import pandas as pd

from bleedtracker.metrics import TOP_K
from bleedtracker.model import SEED

BOOTSTRAP_DRAWS = 10_000


def lost_impressions(frame: pd.DataFrame) -> pd.Series:
    """Each page's impressions minus next month's, never below zero."""
    return (frame.imp - frame.imp_next).clip(lower=0)


def measures(hits: float, lost: float, items: int, pool: pd.DataFrame) -> dict:
    """The measures of a queue. They come from its hits, its lost impressions and its size."""
    precision = hits / items
    base_rate = float(pool.label_declining.mean())
    return {
        "margin_points": round(100 * (precision - base_rate), 1),
        "precision": round(precision, 4),
        "base_rate": round(base_rate, 4),
        "lost_impressions_share": round(lost / float(lost_impressions(pool).sum()), 4),
        "random_lost_impressions_share": round(items / len(pool), 4),
    }


def queue_measures(queue: pd.DataFrame, pool: pd.DataFrame) -> dict:
    """The measures of a queue of whole pages drawn from `pool`."""
    return measures(
        float(queue.label_declining.sum()),
        float(lost_impressions(queue).sum()),
        len(queue),
        pool,
    )


def tie_fair_sum(scores, values, k: int) -> float:
    """Sum of `values` over the k highest scores.

    Pages tied at the cut share the remaining slots. Each slot counts the mean value
    of the tied pages. A random pick among them gives the same on average.
    """
    s = np.asarray(scores, dtype=float)
    v = np.asarray(values, dtype=float)
    if len(s) < k:
        raise ValueError(f"A queue of {k} needs at least {k} pages, got {len(s)}.")
    cut = np.sort(s)[::-1][k - 1]
    above = s > cut
    tied = s == cut
    return float(v[above].sum() + (k - int(above.sum())) * v[tied].mean())


def slot_split(pages_per_client: pd.Series, slots: int = TOP_K) -> pd.Series:
    """Slots per client. Each gets one, then the rest go by largest remainder of the shares.

    A tie on the remainder goes to the client with more pages, then to the first one
    in the index.
    """
    if len(pages_per_client) > slots:
        raise ValueError("There are more clients than slots.")
    rest = slots - len(pages_per_client)
    quota = rest * pages_per_client / pages_per_client.sum()
    floor = np.floor(quota).astype(int)
    left = rest - int(floor.sum())
    order = pd.DataFrame(
        {"remainder": quota - floor, "pages": pages_per_client, "position": range(len(quota))}
    ).sort_values(["remainder", "pages", "position"], ascending=[False, False, True])
    extra = pd.Series(0, index=pages_per_client.index)
    extra[order.index[:left]] = 1
    return 1 + floor + extra


def per_account_measures(pool: pd.DataFrame, score: str, eligible_only: bool) -> dict:
    """The per-account queue's measures for one scorer.

    Slots follow each client's share of the whole pool's pages. Inside a client, the
    scorer's top pages fill its slots. Ties are counted tie-fair. With `eligible_only`,
    a client's pages are the ones the gates let through.
    """
    slots = slot_split(pool.groupby("client_hash_id").size())
    candidates = pool[pool.eligible == 1] if eligible_only else pool
    lost = lost_impressions(candidates)
    hits = 0.0
    lost_sum = 0.0
    for client, k in slots.items():
        mine = candidates.client_hash_id == client
        if mine.sum() < k:
            raise ValueError("A client has fewer pages to pick from than its slots.")
        hits += tie_fair_sum(candidates[score][mine], candidates.label_declining[mine], k)
        lost_sum += tie_fair_sum(candidates[score][mine], lost[mine], k)
    measured = measures(hits, lost_sum, int(slots.sum()), pool)
    measured["slots_per_client"] = sorted(int(k) for k in slots)
    return measured


def editor_rank(pool: pd.DataFrame) -> np.ndarray:
    """Each page's place in the rule's editor queue order. Pages it can't take get inf.

    The order matches the editor queue. It takes eligible pages the rule scores above
    zero, by score, then by impressions, then by row.
    """
    takes = ((pool.eligible == 1) & (pool.action_score > 0)).to_numpy()
    order = np.lexsort((np.arange(len(pool)), -pool.imp.to_numpy(), -pool.action_score.to_numpy()))
    rank = np.full(len(pool), np.inf)
    taken = order[takes[order]]
    rank[taken] = np.arange(len(taken))
    return rank


def forest_rank(pool: pd.DataFrame) -> np.ndarray:
    """Each page's place in the forest's top-50 order over the whole pool."""
    order = np.lexsort((np.arange(len(pool)), -pool.forest_score.to_numpy()))
    rank = np.empty(len(pool))
    rank[order] = np.arange(len(pool))
    return rank


def top_precision(rank: np.ndarray, labels: np.ndarray, k: int = TOP_K) -> float:
    """Precision of the k pages with the lowest rank. It uses fewer when fewer can be taken."""
    finite = np.isfinite(rank)
    r, y = rank[finite], labels[finite]
    if len(r) > k:
        keep = np.argpartition(r, k - 1)[:k]
        r, y = r[keep], y[keep]
    return float(y.mean())


def paired_bootstrap(pool: pd.DataFrame, draws: int = BOOTSTRAP_DRAWS) -> dict:
    """The forest minus the rule at 50 on pages resampled inside one deployment.

    Each draw resamples the held-out pool's pages with replacement and rebuilds both
    top 50s the way the reported precision@50 builds them. Every call starts its own
    generator with seed 42. Two versions judged on the same pool then get the same
    resamples.
    """
    labels = pool.label_declining.to_numpy(dtype=float)
    rule, forest = editor_rank(pool), forest_rank(pool)
    rng = np.random.default_rng(SEED)
    diffs = np.empty(draws)
    for i in range(draws):
        idx = rng.integers(0, len(pool), len(pool))
        diffs[i] = top_precision(forest[idx], labels[idx]) - top_precision(rule[idx], labels[idx])
    observed = top_precision(forest, labels) - top_precision(rule, labels)
    return {
        "forest_minus_rule_points": round(100 * observed, 1),
        "interval_90": [
            round(float(np.percentile(diffs, 5)) * 100, 1),
            round(float(np.percentile(diffs, 95)) * 100, 1),
        ],
    }
