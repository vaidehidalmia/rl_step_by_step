"""Experiments: policy iteration trace, two-gamma comparison, and the full gamma sweep.

Run with:  python experiments.py
"""

import matplotlib.pyplot as plt

from env import N, TERMINAL_LABELS, build_P, destination
from dp import value_iteration, greedy_policy, policy_iteration
from viz import print_policy, plot_value_and_policy


def show_policy_iteration(P, gamma=0.9):
    """Print the policy at each round of policy iteration."""
    def show(policy):
        print_policy(policy)
        print()

    _, _, iterations = policy_iteration(P, gamma, callback=show)
    print(f"policy iteration converged in {iterations} iterations")


def compare_two_gammas(P, gammas=(0.9, 0.5)):
    """Side-by-side heatmaps for two discount factors."""
    fig, axes = plt.subplots(1, len(gammas), figsize=(5.5 * len(gammas), 5))
    for gamma, ax in zip(gammas, axes):
        V, _ = value_iteration(P, gamma)
        plot_value_and_policy(V, greedy_policy(P, V, gamma), gamma, ax)
    plt.tight_layout()
    plt.show()


def run_gamma_sweep(P, gammas=(0.4, 0.5, 0.6, 0.65, 0.7, 0.71, 0.73, 0.9)):
    """Plot the policy for each gamma and report which cells end up at S."""
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    for gamma, ax in zip(gammas, axes.flat):
        V, _ = value_iteration(P, gamma)
        pi = greedy_policy(P, V, gamma)
        plot_value_and_policy(V, pi, gamma, ax)

        s_cells = [s for s in range(N * N)
                   if s not in TERMINAL_LABELS and destination(pi, s) == "S"]
        print(f"γ = {gamma:<5} A goes to {destination(pi, 20)}   "
              f"S territory ({len(s_cells)} cells): {s_cells}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    P = build_P()
    show_policy_iteration(P)
    compare_two_gammas(P)
    run_gamma_sweep(P)