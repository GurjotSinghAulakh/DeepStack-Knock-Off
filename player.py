import numpy as np
from dataclasses import dataclass, field
from poker_oracle import PokerOracle


@dataclass
class Player:
    is_human: bool
    pile: int
    name: str
    aggresiveness: float = 1
    hole_cards: list[str] = field(default_factory=list)
    active: bool = True
    has_raised: bool = False
    has_folded: bool = False

    def choose_action(self, actions, state, oracle: PokerOracle):
        rollout_win_rate = oracle.evaluate_hole_pair_win_probability(self.hole_cards, state["public_cards"], num_simulations=5000)
        actions_list = [f"{i + 1}. {action['action']} {action['amount']}" for i, action in enumerate(actions)]
        print(f"\nLegal actions for {self.name}: {actions_list}, pile: {self.pile}. Hole cards: {self.hole_cards}, community_cards: {state['public_cards']} = RWR: {rollout_win_rate:.2f}")
        if self.is_human:
            while True:
                try:
                    chosen_idx = int(input("Choose an action: ")) - 1
                    if 0 <= chosen_idx < len(actions):
                        break
                    else:
                        print("Invalid action number, please try again.")
                except ValueError:
                    print("Invalid input, please enter a number.")
        else:
            chosen_idx = self.ai_choose_action(actions, state, rollout_win_rate)

        print(f"{self.name} chose {actions[chosen_idx]}")
        return actions[chosen_idx]

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return "Human - " + self.name if self.is_human else "AI - " + self.name

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def ai_choose_action(self, actions, state, rollout_win_rate):
        # Initialize weights for each action based on the action type
        base_weights = {
            "fold": 0.2,
            "check": 0.5,
            "call": 0.5,
            "raise": 0.5,
            "allin": 0.5,
        }
        weights = np.array([base_weights.get(action["action"], 0) for action in actions])

        # print(f"Initial weights: {weights}")
        # Compute the adjustment factor
        adjustment_factor = self.aggresiveness * rollout_win_rate
        # print(f"Adjustment factor: {adjustment_factor:.2f}, aggressiveness: {self.aggresiveness:.2f}, RWR: {rollout_win_rate:.2f}")

        # Apply a sigmoid function to adjust weights based on the adjustment factor
        for i, action in enumerate(actions):
            if action["action"] in ["raise", "allin"]:
                # Apply more weight but control with sigmoid function
                weights[i] *= self.sigmoid(20 * (adjustment_factor - 0.5))  # Adjust sigmoid curve steepness and shift
            else:
                # Scale down aggressive adjustment
                weights[i] *= self.sigmoid(10 * (adjustment_factor - 0.3))

        weights /= weights.sum()
        # print(f"Adjusted weights: {weights}")

        return np.random.choice(len(actions), p=weights)
