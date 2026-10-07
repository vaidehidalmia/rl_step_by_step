import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
 
from env import N, ACTIONS, build_P
from model_learning import collect_experience, build_P_hat, evaluate_learned_model
from viz import plot_value_and_policy

SLIP = 0.2
GAMMA = 0.9
A = 20 

def learn_and_evaluate(P, data, gamma=GAMMA):
    """Build P_hat from data and return (V_hat, pi_hat, V_pi_hat, V_star)."""
    P_hat = build_P_hat(data, N * N, len(ACTIONS))
    return evaluate_learned_model(P, P_hat, gamma)

def belief_vs_reality(P, n_values=(5, 20, 5000), seeds=(0, 1, 2)):
    """Print what the agent believes about A, what its plan really earns, and the best possible."""
    print(f"slip = {SLIP}, gamma = {GAMMA}, watching state {A}\n")
    for n in n_values:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            data = collect_experience(P, n, rng)
            V_hat, _, V_pi_hat, V_star = learn_and_evaluate(P, data)
            print(f"n={n:<5} seed={seed}  believes {V_hat[A]:5.2f}  "
                  f"actually {V_pi_hat[A]:5.2f}  best {V_star[A]:5.2f}  "
                  f"worst gap {np.max(V_star - V_pi_hat):5.2f}")
        print()

def learning_curve(P, n_values=(1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000),
                   n_seeds=10):
    """Plot belief and reality at A, and the worst-case gap, against the amount of data.
 
    Each seed is one agent that keeps its earlier experience and keeps collecting more,
    so the point at n=50 uses the same first 20 episodes as the point at n=20.
    """
    n_values = sorted(n_values)
    believed = np.zeros((n_seeds, len(n_values)))
    actual = np.zeros((n_seeds, len(n_values)))
    worst_gap = np.zeros((n_seeds, len(n_values)))
 
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        data, collected = [], 0
        for j, n in enumerate(n_values):
            data += collect_experience(P, n - collected, rng)   # only the new episodes
            collected = n
            V_hat, _, V_pi_hat, V_star = learn_and_evaluate(P, data)
            believed[seed, j] = V_hat[A]
            actual[seed, j] = V_pi_hat[A]
            worst_gap[seed, j] = np.max(V_star - V_pi_hat)
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
 
    for values, label in [(believed, "believes (V_hat)"), (actual, "actually earns (V_pi_hat)")]:
        mean, std = values.mean(axis=0), values.std(axis=0)
        ax1.plot(n_values, mean, marker="o", label=label)
        ax1.fill_between(n_values, mean - std, mean + std, alpha=0.2)
    ax1.axhline(V_star[A], color="black", linestyle="--", label="best possible (V*)")
    ax1.set_xscale("log")
    ax1.set_xlabel("episodes of experience")
    ax1.set_ylabel(f"value at state {A}")
    ax1.set_title("Belief vs. reality at A (mean ± 1 std over seeds)")
    ax1.legend()
 
    mean, std = worst_gap.mean(axis=0), worst_gap.std(axis=0)
    ax2.plot(n_values, mean, marker="o", color="tab:red")
    ax2.fill_between(n_values, np.maximum(mean - std, 0), mean + std,
                     color="tab:red", alpha=0.2)
    ax2.set_xscale("log")
    ax2.set_xlabel("episodes of experience")
    ax2.set_ylabel("max over states of V* − V_pi_hat")
    ax2.set_title("Worst-case loss from planning with the learned model")
 
    fig.suptitle(f"Model learning: slip = {SLIP}, γ = {GAMMA}, {n_seeds} seeds")
    plt.tight_layout()
    plt.show()
 
 
def belief_vs_reality_maps(P, n_values=(5, 20, 2000), seed=0):
    """Top row: what the agent believes (V_hat). Bottom row: what its policy really earns."""
    fig, axes = plt.subplots(2, len(n_values), figsize=(5.5 * len(n_values), 10))
    for j, n in enumerate(n_values):
        rng = np.random.default_rng(seed)
        data = collect_experience(P, n, rng)
        V_hat, pi_hat, V_pi_hat, _ = learn_and_evaluate(P, data)
 
        plot_value_and_policy(V_hat, pi_hat, GAMMA, axes[0, j])
        axes[0, j].set_title(f"{n} episodes: believed value")
        plot_value_and_policy(V_pi_hat, pi_hat, GAMMA, axes[1, j])
        axes[1, j].set_title(f"{n} episodes: true value of that policy")
 
    fig.suptitle(f"Same policy, two value functions (slip = {SLIP}, γ = {GAMMA}, seed {seed})")
    plt.tight_layout()
    plt.show()
 
 
if __name__ == "__main__":
    P = build_P(slip=SLIP)
    belief_vs_reality(P)
    learning_curve(P)
    belief_vs_reality_maps(P)
 