import numpy as np
import matplotlib.pyplot as plt

# row 0:  .  .  .  .  L      L = +10 (terminal), far corner
# row 1:  .  .  .  .  .
# row 2:  .  .  .  .  .
# row 3:  .  .  .  .  .
# row 4:  A  S  .  .  .      S = +1 (terminal), A = where you'll "start" watching

N = 5

def rc_to_i(row, col):
    return row * N + col

def i_to_rc(i):
    return (i // N, i % N)

for s in range(N * N):
    assert rc_to_i(*i_to_rc(s)) == s
assert rc_to_i(0, 4) == 4
assert i_to_rc(21) == (4, 1)
for row in range(N):
    print(" ".join(f"{rc_to_i(row, col):2d}" for col in range(N)))

ACTIONS = [
    (-1, 0),   # 0: up
    (0, 1),   # 1: right
    (1, 0),   # 2: down
    (0, -1),   # 3: left
]
ACTION_ARROWS = ["↑", "→", "↓", "←"]

def move(s, a):
    """Return the next state after taking action a from state s (ignoring terminals)."""
    row, col = i_to_rc(s)
    dr, dc = ACTIONS[a]
    new_row, new_col = (row + dr, col + dc) 
    if new_row < 0 or new_row >= N: new_row = row
    if new_col < 0 or new_col >= N: new_col = col
    return rc_to_i(new_row, new_col)

assert move(20, 1) == 21   # A moving right reaches S
assert move(20, 3) == 20   # A moving left hits the wall, stays
assert move(20, 2) == 20   # A moving down hits the wall, stays
assert move(12, 0) == 7    # center moving up

TERMINAL_REWARDS = {
    rc_to_i(0, 4): 10.0,   # L
    rc_to_i(4, 1): 1.0,    # S
}

assert TERMINAL_REWARDS[4] == 10.0
assert TERMINAL_REWARDS[21] == 1.0
assert 20 not in TERMINAL_REWARDS

def build_P():
    # P[s][a] = [(prob, next_state, reward, done), ...]
    P = [[None] * 4 for _ in range(N * N)]

    for s in range(N * N):
        if s in TERMINAL_REWARDS:
            for a in range(4):
                P[s][a] = [(1.0, s, 0.0, True)]
            continue

        for a in range(4):
            next_state = move(s, a)
            reward = TERMINAL_REWARDS.get(next_state, 0.0)
            done = next_state in TERMINAL_REWARDS  
            P[s][a] = [(1.0, next_state, reward, done)]   

    return P

P = build_P()
assert P[20][1] == [(1.0, 21, 1.0, True)]    # A → S: reward +1, episode ends
assert P[20][3] == [(1.0, 20, 0.0, False)]   # A → wall: stay, nothing happens
assert P[3][1]  == [(1.0, 4, 10.0, True)]    # next to L → L
assert P[21][3] == [(1.0, 21, 0.0, True)]    # S is terminal: can't leave
assert P[12][0] == [(1.0, 7, 0.0, False)]    # ordinary move


def q_values(P, V, s, gamma):
    # q_value = probability × (reward + γ × value of next state)
    q = np.zeros(4)
    for a in range(4):
        for prob, next_state, reward, done in P[s][a]:
            v_next_state = V[next_state] if not done else 0
            q[a] += prob * (reward + gamma * v_next_state)
    return q

V_test = np.zeros(N * N)
assert np.allclose(q_values(P, V_test, 3, 0.9),  [0, 10, 0, 0])   # only "right" pays
assert np.allclose(q_values(P, V_test, 20, 0.9), [0, 1, 0, 0])    # only "right" (into S) pays

V_test[4] = 5
assert np.isclose(q_values(P, V_test, 3, 0.9)[1], 10.0)

def value_iteration(P, gamma, theta=1e-8):
    V = np.zeros(N * N)
    sweeps = 0
    while True:
        delta = 0.0
        for s in range(N * N):
            # 1. compute q for this state
            q = q_values(P, V, s, gamma)
            # 2. new value = max of q
            new_value = max(q)
            # 3. update delta with how much V[s] changed
            delta = max(delta, abs(new_value - V[s]))
            # 4. store the new value
            V[s] = new_value
        sweeps += 1
        if delta < theta:
            break
    return V, sweeps

V, sweeps = value_iteration(P, 0.9)
# print(np.round(V.reshape(N, N), 2))
# print(sweeps)

assert np.isclose(V[3], 10.0)
assert np.isclose(V[2], 9.0)
assert V[4] == 0 and V[21] == 0
assert np.isclose(V[20], 10 * 0.9**7)

def greedy_policy(P, V, gamma):
    policy = np.zeros(N * N, dtype=int)
    for s in range(N * N):
        # compute q for s, store the index of the best action
        q = q_values(P, V, s, gamma)
        policy[s] = np.argmax(q)
    return policy

V9, _ = value_iteration(P, 0.9)
pi9 = greedy_policy(P, V9, 0.9)
assert pi9[20] == 0    # A goes up, toward L
assert pi9[3] == 1     # next to L: go right
assert pi9[9] == 0     # below L: go up

V5, _ = value_iteration(P, 0.5)
pi5 = greedy_policy(P, V5, 0.5)
assert pi5[20] == 1    # A goes right, into S

TERMINAL_LABELS = {rc_to_i(0, 4): "L", rc_to_i(4, 1): "S"}

def print_policy(policy):
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

# print_policy(pi9)
# print_policy(pi5)

def policy_evaluation(P, policy, gamma, theta=1e-8):
    V = np.zeros(N * N)
    while True:
        delta = 0.0
        for s in range(N * N):
            # 1. compute q for this state
            q = q_values(P, V, s, gamma)
            # 2. new value = value of the action the policy picks
            new_value = q[policy[s]]
            # 3. update delta with how much V[s] changed
            delta = max(delta, abs(new_value - V[s]))
            # 4. store the new value
            V[s] = new_value
            # same as value iteration, except: which entry of q?
        if delta < theta:
            break
    return V

V_up = policy_evaluation(P, np.zeros(N * N, dtype=int), 0.9)
assert np.isclose(V_up[9], 10.0)
assert np.isclose(V_up[24], 7.29)
assert V_up[20] == 0

def policy_iteration(P, gamma):
    policy = np.zeros(N * N, dtype=int)   # start with "always up"
    iterations = 0
    while True:
        print_policy(policy)
        print()
        V = policy_evaluation(P, policy, gamma)           # evaluate
        new_policy = greedy_policy(P, V, gamma)
        iterations += 1
        # if the policy didn't change, stop
        if np.array_equal(new_policy, policy):
            break
        policy = new_policy
    return V, policy, iterations

V_pi, pi_pi, iters = policy_iteration(P, 0.9)
assert np.allclose(V_pi, V9)
assert np.array_equal(pi_pi, pi9)
print(iters)

def plot_value_and_policy(V, policy, gamma, ax):
    # Layer 1: heatmap of the value function
    im = ax.imshow(V.reshape(N, N), cmap="viridis")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Text color threshold for contrast (guard against an all-zero V)
    vmax = V.max() if V.max() > 0 else 1.0

    # Layers 2–3: label every cell
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

    # Layer 4: clean frame and title
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"γ = {gamma}")

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
plot_value_and_policy(V9, pi9, 0.9, axes[0])
plot_value_and_policy(V5, pi5, 0.5, axes[1])
plt.tight_layout()
plt.show()

def destination(policy, s, max_steps=50):
    """Follow the policy from s and return the label of the terminal it reaches."""
    for _ in range(max_steps):
        if s in TERMINAL_LABELS:
            return TERMINAL_LABELS[s]
        s = move(s, policy[s])
    return None   # never reached a terminal (policy loops or hits a wall forever)


gammas = [0.4, 0.5, 0.6, 0.65, 0.7, 0.71, 0.73, 0.9]

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