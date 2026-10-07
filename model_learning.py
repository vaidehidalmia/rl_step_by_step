import numpy as np
from collections import Counter, defaultdict

from env import N, TERMINAL_REWARDS, ACTIONS
from dp import value_iteration, greedy_policy, policy_iteration, policy_evaluation


NON_TERMINAL_STATES = [s for s in range(N * N) if s not in TERMINAL_REWARDS]

def step(P, s, a, rng):
    """Sample one transition from the true model: return (next_state, reward, done)."""
    outcomes = P[s][a]
    probs = [prob for prob, *_ in outcomes]
    i = rng.choice(len(outcomes), p=probs)
    _, next_state, reward, done = outcomes[i]
    return next_state, reward, done

def reset(rng):
    """Return a random non-terminal start state."""
    return int(rng.choice(NON_TERMINAL_STATES))

def collect_experience(P, n_episodes, rng, max_steps=100):
    """Run random-policy episodes; return a list of (s, a, next_state, reward, done)."""
    data = []
    for _ in range(n_episodes):
        s = reset(rng)
        for _ in range(max_steps):
            a = rng.integers(len(ACTIONS))
            next_state, reward, done = step(P, s, a, rng)
            data.append((s, a, next_state, reward, done))
            if done:
                break
            s = next_state
    return data

def build_P_hat(data, n_states, n_actions):
    """Estimate P from transitions; unseen (s, a) pairs self-loop with reward 0."""
    counts = defaultdict(Counter)
    for s, a, next_state, reward, done in data:
        counts[(s, a)][(next_state, reward, done)] += 1

    P_hat = [[None] * n_actions for _ in range(n_states)]
    for s in range(n_states):
        for a in range(n_actions):
            outcome_counts = counts[(s, a)]
            total = outcome_counts.total()
            if total == 0:
                P_hat[s][a] = [(1.0, s, 0.0, False)]
            else:
                P_hat[s][a] = [(c/total, next_state, reward, done) for (next_state, reward, done), c in outcome_counts.items()]
    return P_hat

def evaluate_learned_model(P, P_hat, gamma):
    """Plan in P_hat, grade the resulting policy in the true P."""
    V_hat, _ = value_iteration(P_hat, gamma)
    pi_hat = greedy_policy(P_hat, V_hat, gamma)
    V_pi_hat = policy_evaluation(P, pi_hat, gamma)
    V_star, _ = value_iteration(P, gamma)
    return V_hat, pi_hat, V_pi_hat, V_star