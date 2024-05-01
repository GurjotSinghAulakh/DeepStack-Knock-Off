import numpy as np
from dataclasses import dataclass, field
from poker_oracle import PokerOracle
from enum import Enum
from subtree_manager import SubtreeManager
from state_manager import StateManager, GameState, PokerGameStage
from actions import ActionType, Action, ACTIONS, agent_action_index
from config import COMBINATIONS


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
    eval_hole_probs: bool = False
    range_size: int = COMBINATIONS

    def __post_init__(self):
        self.r1 = np.ones(self.range_size) / self.range_size
        self.r2 = np.ones(self.range_size) / self.range_size
        self.opponent_strategy = np.ones((self.range_size, len(ACTIONS))) / len(ACTIONS)

    def reset_ranges(self):
        self.r1 = np.ones(self.range_size) / self.range_size
        self.r2 = np.ones(self.range_size) / self.range_size

    def inform(self, action_taken, player):
        if player.name == self.name:
            return

        action_idx = agent_action_index(action_taken)

        updated_r2 = SubtreeManager.bayesian_range_update(self.r2, self.opponent_strategy, action_idx)
        if np.sum(updated_r2) == 0:
            return

        self.r2 = updated_r2

    def choose_action(self, actions, state, oracle=PokerOracle()):
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

    def take_action(self, state: GameState):
        legal_actions = StateManager.get_legal_actions(state)
        if not self.eval_hole_probs:
            idx = np.random.randint(len(legal_actions))
            act_type = legal_actions[idx]
        else:
            act = self.rollout_action(legal_actions, state)
            act_type = act.action_type
        if act_type == ActionType.RAISE:
            amount = 10
        else:
            amount = 0
        return Action(act_type, amount)

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return "Human - " + self.name if self.is_human else "AI - " + self.name

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def ai_choose_action(self, actions, state, rollout_win_rate):
        # Initialize weights for each action based on the action type
        base_weights = {
            ActionType.FOLD: 0.2,
            ActionType.CHECK: 0.5,
            ActionType.CALL: 0.5,
            ActionType.RAISE: 0.5,
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

    def rollout_action(self, legal_actions, game_state: GameState) -> Action:
        """
        Handles logic for using rollout based strategy
        """
        win_probability = 0
        # if game_state.stage == PokerGameStage.PRE_FLOP:
        #     win_probability = PokerOracle.hole_hand_winning_probability_cheat_sheet(
        #         self.hand, len(game_state.player_bets)
        #     )
        # else:
        win_probability = PokerOracle().evaluate_hole_pair_win_probability(self.hole_cards, game_state.public_info, num_simulations=5000)

        print(win_probability)
        if win_probability < 0.1:
            return Action(ActionType.FOLD, 0)
        elif win_probability < 0.5:
            idx = np.random.randint(len(legal_actions))
            act_type = legal_actions[idx]
            if act_type == ActionType.RAISE:
                return Action(act_type, 10)
            else:
                return Action(act_type, 0)
        elif win_probability < 0.8:
            return Action(ActionType.CALL, 0)
        else:
            return Action(ActionType.RAISE, 10)
