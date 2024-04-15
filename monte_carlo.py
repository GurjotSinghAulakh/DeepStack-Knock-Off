import random


def monte_carlo_simulation(root, trials=1000):
    win_count = 0
    for _ in range(trials):
        result = run_simulation(root)
        if result == "win":
            win_count += 1
    return win_count / trials

def run_simulation(node):
    if not node.children:
        return simulate_final_state(node.state)
    # Randomly select one of the possible actions
    choice = random.choice(node.children)
    return run_simulation(choice)

def simulate_final_state(state):
    # Determine win or lose based on state
    return "win" if random.random() > 0.5 else "lose"
