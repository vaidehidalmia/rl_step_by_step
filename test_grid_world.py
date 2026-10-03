"""Tests for the deterministic GridWorld. Run with:  pytest"""

import numpy as np
import pytest

from env import N, TERMINAL_REWARDS, TERMINAL_LABELS, rc_to_i, i_to_rc, move, build_P, destination, perpendicular
from dp import q_values, value_iteration, greedy_policy, policy_evaluation, policy_iteration


@pytest.fixture
def P():
    return build_P()


# --- env: coordinates and movement -------------------------------------------

def test_coordinate_round_trip():
    for s in range(N * N):
        assert rc_to_i(*i_to_rc(s)) == s
    assert rc_to_i(0, 4) == 4
    assert i_to_rc(21) == (4, 1)


def test_move_basic():
    assert move(20, 1) == 21   # A moving right reaches S
    assert move(12, 0) == 7    # center moving up


def test_move_walls():
    assert move(20, 3) == 20   # A left: wall
    assert move(20, 2) == 20   # A down: wall
    assert move(4, 0) == 4     # top-right corner, up: wall
    assert move(4, 1) == 4     # top-right corner, right: wall
    assert move(4, 2) == 9
    assert move(4, 3) == 3


def test_terminals():
    assert TERMINAL_REWARDS[4] == 10.0
    assert TERMINAL_REWARDS[21] == 1.0
    assert 20 not in TERMINAL_REWARDS


# --- env: transition model -----------------------------------------------------

def test_build_P(P):
    assert P[20][1] == [(1.0, 21, 1.0, True)]    # A -> S: reward +1, episode ends
    assert P[20][3] == [(1.0, 20, 0.0, False)]   # A -> wall: stay
    assert P[3][1] == [(1.0, 4, 10.0, True)]     # next to L -> L
    assert P[21][3] == [(1.0, 21, 0.0, True)]    # S is terminal: can't leave
    assert P[12][0] == [(1.0, 7, 0.0, False)]    # ordinary move


def test_probabilities_sum_to_one(P):
    for s in range(len(P)):
        for a in range(len(P[s])):
            assert np.isclose(sum(prob for prob, *_ in P[s][a]), 1.0)


# --- dp: one-step lookahead ----------------------------------------------------

def test_q_values_zero_V(P):
    V = np.zeros(N * N)
    assert np.allclose(q_values(P, V, 3, 0.9), [0, 10, 0, 0])
    assert np.allclose(q_values(P, V, 20, 0.9), [0, 1, 0, 0])


def test_q_values_ignore_value_after_done(P):
    V = np.zeros(N * N)
    V[4] = 5   # fake value on L must not leak into Q
    assert np.isclose(q_values(P, V, 3, 0.9)[1], 10.0)


# --- dp: value iteration -------------------------------------------------------

def test_value_iteration_gamma_09(P):
    V, _ = value_iteration(P, 0.9)
    assert np.isclose(V[3], 10.0)
    assert np.isclose(V[2], 9.0)
    assert V[4] == 0 and V[21] == 0
    assert np.isclose(V[20], 10 * 0.9**7)


def test_greedy_policy(P):
    V9, _ = value_iteration(P, 0.9)
    pi9 = greedy_policy(P, V9, 0.9)
    assert pi9[20] == 0   # A goes up, toward L
    assert pi9[3] == 1    # next to L: right
    assert pi9[9] == 0    # below L: up

    V5, _ = value_iteration(P, 0.5)
    pi5 = greedy_policy(P, V5, 0.5)
    assert pi5[20] == 1   # A goes right, into S


# --- dp: policy evaluation and policy iteration --------------------------------

def test_policy_evaluation_always_up(P):
    V_up = policy_evaluation(P, np.zeros(N * N, dtype=int), 0.9)
    assert np.isclose(V_up[9], 10.0)
    assert np.isclose(V_up[24], 7.29)
    assert V_up[20] == 0


def test_policy_iteration_matches_value_iteration(P):
    V9, _ = value_iteration(P, 0.9)
    pi9 = greedy_policy(P, V9, 0.9)
    V_pi, pi_pi, _ = policy_iteration(P, 0.9)
    assert np.allclose(V_pi, V9)
    assert np.array_equal(pi_pi, pi9)


# --- experiments: gamma thresholds ---------------------------------------------

@pytest.mark.parametrize("gamma, expected", [
    (0.4, "S"), (0.5, "S"), (0.65, "S"), (0.71, "S"), (0.73, "L"), (0.9, "L"),
])
def test_A_destination_threshold(P, gamma, expected):
    V, _ = value_iteration(P, gamma)
    assert destination(greedy_policy(P, V, gamma), 20) == expected


@pytest.mark.parametrize("gamma, expected_cells", [
    (0.4, [10, 11, 15, 16, 17, 20, 22, 23]),
    (0.5, [15, 16, 20, 22]),
    (0.6, [15, 16, 20, 22]),
    (0.65, [20]),
    (0.73, []),
])
def test_S_territory(P, gamma, expected_cells):
    V, _ = value_iteration(P, gamma)
    pi = greedy_policy(P, V, gamma)
    s_cells = [s for s in range(N * N)
               if s not in TERMINAL_LABELS and destination(pi, s) == "S"]
    assert s_cells == expected_cells

def test_perpendicular():
    assert set(perpendicular(0)) == {1, 3}
    assert set(perpendicular(3)) == {0, 2}

def test_build_P_slip_from_A():
    P = build_P(slip=0.2)
    assert sorted(P[20][1]) == sorted([
        (0.8, 21, 1.0, True),    # intended: right, into S
        (0.1, 15, 0.0, False),   # slip up
        (0.1, 20, 0.0, False),   # slip down: wall, stay
    ])


def test_slip_probabilities_sum_to_one():
    P = build_P(slip=0.2)
    for s in range(len(P)):
        for a in range(len(P[s])):
            assert np.isclose(sum(prob for prob, *_ in P[s][a]), 1.0)