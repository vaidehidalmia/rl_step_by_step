# row 0:  .  .  .  .  L      L = +10 (terminal), far corner
# row 1:  .  .  .  .  .
# row 2:  .  .  .  .  .
# row 3:  .  .  .  .  .
# row 4:  A  S  .  .  .      S = +1 (terminal), A = where you'll "start" watching

N = 5
 
ACTIONS = [
    (-1, 0),   # 0: up
    (0, 1),    # 1: right
    (1, 0),    # 2: down
    (0, -1),   # 3: left
]
ACTION_ARROWS = ["↑", "→", "↓", "←"]

def rc_to_i(row, col):
    return row * N + col
 
 
def i_to_rc(i):
    return (i // N, i % N)
 
 
TERMINAL_REWARDS = {
    rc_to_i(0, 4): 10.0,   # L
    rc_to_i(4, 1): 1.0,    # S
}
TERMINAL_LABELS = {rc_to_i(0, 4): "L", rc_to_i(4, 1): "S"}

def move(s, a):
    """Return the next state after taking action a from state s (ignoring terminals).
 
    Moving off the grid leaves the agent where it is.
    """
    row, col = i_to_rc(s)
    dr, dc = ACTIONS[a]
    new_row, new_col = row + dr, col + dc
    if new_row < 0 or new_row >= N:
        new_row = row
    if new_col < 0 or new_col >= N:
        new_col = col
    return rc_to_i(new_row, new_col)

def build_P(slip = 0.0):
    """Build the transition model: P[s][a] = [(prob, next_state, reward, done), ...].
 
    Terminal states self-loop with reward 0; otherwise the agent could leave
    and re-enter S to farm its reward.
    """
    P = [[None] * len(ACTIONS) for _ in range(N * N)]
 
    for s in range(N * N):
        if s in TERMINAL_REWARDS:
            for a in range(len(ACTIONS)):
                P[s][a] = [(1.0, s, 0.0, True)]
            continue
 
        
        for a in range(len(ACTIONS)):
            side1, side2 = perpendicular(a)
            outcomes = [(a, 1 - slip), (side1, slip/2), (side2, slip/2)]
            P[s][a] = []
            for action, prob in outcomes:
                if prob == 0:
                    continue
                next_state = move(s, action)
                reward = TERMINAL_REWARDS.get(next_state, 0.0)
                done = next_state in TERMINAL_REWARDS
                P[s][a].append((prob, next_state, reward, done))
 
    return P

def destination(policy, s, max_steps=50):
    """Follow the policy from s and return the label of the terminal it reaches.
 
    Returns None if no terminal is reached (the policy loops or bumps a wall forever).
    """
    for _ in range(max_steps):
        if s in TERMINAL_LABELS:
            return TERMINAL_LABELS[s]
        s = move(s, policy[s])
    return None

def perpendicular(a):
    """Return the two actions at 90 degrees to action a."""
    len_actions = len(ACTIONS)
    return ((a - 1 + len_actions) % len_actions, (a + 1) % len_actions)