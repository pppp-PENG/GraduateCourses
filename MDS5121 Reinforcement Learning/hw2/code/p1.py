import numpy as np


def solve_mountain_mdp(n, theta=1e-10):
    """
    solving the mountain climbing MDP using value iteration.

    Args:
        n: the max index of ledges (states are numbered from 0 to n).
        theta: threshold for value function.

    Returns:
        V (n+1,): v[i] is the optimal value function for ledge i.
        policy (n+1,): policy[i] is the optimal action for ledge i ('c' or 'r'). (policy[0] and policy[n] are '-')
    """
    # action space
    actions = ["c", "r"]

    # reward function
    R = np.zeros((n + 1, n + 1))
    R[:, n] = 1.0

    # initialize value function
    V = np.zeros(n + 1)
    V[0] = 0.0
    V[n] = 1.0

    # value iteration
    while True:
        delta = 0.0
        V_new = V.copy()

        for state in range(1, n):
            Q = []
            for action in actions:
                if action == "c":  # Careful Step
                    value = ((n - state) / n) * (R[state, state + 1] + V[state + 1]) + (
                        state / n
                    ) * (R[state, state - 1] + V[state - 1])
                elif action == "r":  # Risky Grapple
                    value = 0.0
                    for next_state in range(0, n + 1):
                        if next_state != state:
                            value += (1 / n) * (R[state, next_state] + V[next_state])
                Q.append(value)

            best_value = max(Q)
            V_new[state] = best_value
            delta = max(delta, abs(best_value - V[state]))

        # update value function
        V = V_new
        # check for convergence
        if delta < theta:
            break

    # initialize policy
    policy = ["-"] * (n + 1)

    for state in range(1, n):
        Q = []
        for action in actions:
            if action == "c":
                value = ((n - state) / n) * (R[state, state + 1] + V[state + 1]) + (
                    state / n
                ) * (R[state, state - 1] + V[state - 1])
            elif action == "r":
                value = 0.0
                for next_state in range(0, n + 1):
                    if next_state != state:
                        value += (1 / n) * (R[state, next_state] + V[next_state])
            Q.append(value)
        best_action_index = np.argmax(Q)
        policy[state] = actions[best_action_index]

    return V, policy


# assume n = 5
if __name__ == "__main__":
    n = 5
    V_opt, policy_opt = solve_mountain_mdp(n)

    print("\nOptimal value function V:")
    for i in range(n + 1):
        print(f"    V[{i}] = {V_opt[i]:.4f}")

    print("\nOptimal policy pi:")
    for i in range(n + 1):
        print(f"    pi_{i} = {policy_opt[i]}")
