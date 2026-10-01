"""Dynamic programming for any finite MDP given as P[s][a] = [(prob, next_state, reward, done), ...].

Nothing here knows about grids: the number of states is len(P) and the number
of actions is len(P[s]), so these functions also work on other MDPs.
"""

import numpy as np


def q_values(P, V, s, gamma):
    """One-step lookahead: Q(s, a) = sum over s' of prob * (reward + gamma * V(s'))."""
    q = np.zeros(len(P[s]))
    for a in range(len(P[s])):
        for prob, next_state, reward, done in P[s][a]:
            v_next_state = V[next_state] if not done else 0
            q[a] += prob * (reward + gamma * v_next_state)
    return q


def value_iteration(P, gamma, theta=1e-8):
    """Apply the Bellman optimality update until the largest change is below theta."""
    V = np.zeros(len(P))
    sweeps = 0
    while True:
        delta = 0.0
        for s in range(len(P)):
            new_value = max(q_values(P, V, s, gamma))
            delta = max(delta, abs(new_value - V[s]))
            V[s] = new_value
        sweeps += 1
        if delta < theta:
            break
    return V, sweeps


def greedy_policy(P, V, gamma):
    """Pick the action with the highest Q-value in each state (ties -> lowest index)."""
    policy = np.zeros(len(P), dtype=int)
    for s in range(len(P)):
        policy[s] = np.argmax(q_values(P, V, s, gamma))
    return policy


def policy_evaluation(P, policy, gamma, theta=1e-8):
    """Compute V for a fixed policy with the Bellman expectation update."""
    V = np.zeros(len(P))
    while True:
        delta = 0.0
        for s in range(len(P)):
            new_value = q_values(P, V, s, gamma)[policy[s]]
            delta = max(delta, abs(new_value - V[s]))
            V[s] = new_value
        if delta < theta:
            break
    return V


def policy_iteration(P, gamma, callback=None):
    """Alternate evaluation and greedy improvement until the policy stops changing.

    callback, if given, is called with the current policy at the start of each iteration.
    """
    policy = np.zeros(len(P), dtype=int)   # start with "always action 0"
    iterations = 0
    while True:
        if callback is not None:
            callback(policy)
        V = policy_evaluation(P, policy, gamma)
        new_policy = greedy_policy(P, V, gamma)
        iterations += 1
        if np.array_equal(new_policy, policy):
            break
        policy = new_policy
    return V, policy, iterations