import numpy as np
import pandas as pd

from bleedtracker.deployments import ANCHOR_PAIR, PAIRS, plans, run_deployments


def test_plans():
    names = [(p.name, p.train_pair, p.test_pair) for p in plans()]
    assert names == [
        ("anchor 2026-03 to 2026-04", ("2026-03", "2026-04"), ("2026-03", "2026-04")),
        ("forward 2026-02 to 2026-03", ("2026-01", "2026-02"), ("2026-02", "2026-03")),
        ("forward 2026-03 to 2026-04", ("2026-02", "2026-03"), ("2026-03", "2026-04")),
        ("forward 2026-04 to 2026-05", ("2026-03", "2026-04"), ("2026-04", "2026-05")),
        ("sealed 2026-05 to 2026-06", ("2026-04", "2026-05"), ("2026-05", "2026-06")),
    ]


def month_frame(seed: int, clients: int = 10, pages: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = clients * pages
    momentum = rng.uniform(0, 1.6, n)
    imp = rng.integers(100, 5000, n).astype(float)
    return pd.DataFrame(
        {
            "client_hash_id": np.repeat([f"c{i}" for i in range(clients)], pages),
            "content_hash_id": [f"p{seed}-{i}" for i in range(n)],
            "imp": imp,
            "log_impressions": np.log1p(imp),
            "active_days": rng.integers(20, 32, n),
            "position": rng.uniform(1, 40, n),
            "ctr": rng.uniform(0, 0.05, n),
            "momentum": momentum,
            "has_first_half_traffic": 1,
            "spike_share": rng.uniform(0, 0.3, n),
            "reason_code": "slipping_and_underclicking",
            "imp_next": imp * rng.uniform(0.3, 1.3, n),
        }
    ).assign(label_declining=lambda f: (f.imp_next < 0.8 * f.imp).astype(int))


def test_every_deployment_judges_only_held_out_clients():
    frames = {pair: month_frame(i) for i, pair in enumerate(PAIRS)}
    ladder = run_deployments(frames)
    assert len(ladder.rows) == 5
    assert not ladder.train_clients & ladder.test_clients
    for plan_name, (queue, pool) in ladder.queues.items():
        assert set(pool.client_hash_id) <= ladder.test_clients, plan_name
        assert (queue.slot_type == "editor_slot").sum() <= 50
    anchor_pages = frames[ANCHOR_PAIR].client_hash_id.isin(ladder.test_clients).sum()
    assert ladder.rows[0]["pages"] == anchor_pages
