import matplotlib.pyplot as plt
 
from env import N, ACTION_ARROWS, TERMINAL_LABELS, rc_to_i, i_to_rc

def print_policy(policy):
    """Print the policy as a grid of arrows, with terminals shown by label."""
    for row in range(N):
        symbols = []
        for col in range(N):
            s = rc_to_i(row, col)
            # if s is terminal: append a label for it
            if s in TERMINAL_LABELS:
                symbols.append(TERMINAL_LABELS[s])
            # else: append the arrow for policy[s]
            else:
                symbols.append(ACTION_ARROWS[policy[s]])
        print(" ".join(symbols))

def plot_value_and_policy(V, policy, gamma, ax):
    """Draw V as a heatmap with the policy's arrows and values on top."""
    im = ax.imshow(V.reshape(N, N), cmap="viridis")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
 
    # Text color threshold for contrast (guard against an all-zero V)
    vmax = V.max() if V.max() > 0 else 1.0
 
    for s in range(N * N):
        row, col = i_to_rc(s)
        color = "white" if V[s] < 0.5 * vmax else "black"
 
        if s in TERMINAL_LABELS:
            ax.text(col, row, TERMINAL_LABELS[s],          # x = col, y = row
                    ha="center", va="center",
                    fontsize=20, fontweight="bold", color=color)
        else:
            ax.text(col, row - 0.12, ACTION_ARROWS[policy[s]],
                    ha="center", va="center", fontsize=18, color=color)
            ax.text(col, row + 0.32, f"{V[s]:.2f}",
                    ha="center", va="center", fontsize=8, color=color)
 
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"γ = {gamma}")