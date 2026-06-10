import numpy as np
import pytest

from pallas.search import allocate_tips

pytestmark = pytest.mark.unit

# frontier entries: (tip_id, target_fp, d_fp_to_other_side)
FRONTIER = [(1, None, 0.02), (2, None, 0.025), (3, None, 0.05)]


def test_round_robin_order():
    idx = allocate_tips(FRONTIER, 6, {}, mode='round_robin')
    assert idx == [0, 1, 2, 0, 1, 2]


def test_adaptive_prefers_close_and_successful():
    stats = {1: [3, 3], 2: [0, 3], 3: [0, 0]}  # tip1 always succeeds
    rng = np.random.default_rng(0)
    idx = allocate_tips(FRONTIER, 2000, stats, mode='adaptive', rng=rng)
    counts = np.bincount(idx, minlength=3)
    assert counts[0] > counts[1], "successful close tip must dominate failing tip"
    assert counts[2] > 0, "unvisited tip must keep an exploration share"


def test_adaptive_single_tip():
    assert allocate_tips(FRONTIER[:1], 3, {}, mode='adaptive') == [0, 0, 0]
