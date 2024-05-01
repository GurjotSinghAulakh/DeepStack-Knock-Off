from typing import Tuple
import numpy as np
from actions import ACTIONS, Action
from state_manager import GameState, PokerGameStage
from subtree_manager import SubtreeManager


class Resolver:
    def resolve(
        self,
        state: GameState,
        r1: np.ndarray,
        r2: np.ndarray,
        end_stage: PokerGameStage,
        end_depth: int,
        nbr_rollouts: int,
        show_internal_values: bool = False,
    ) -> Tuple[Action, np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs the Re-Solve algorithm to generate an optimal action, and new ranges

        returns tuple:
            action: Action
            r1: np.ndarray
            r2: np.ndarray
            strategy of new state: np.ndarray

        """
        strategy = np.ones((r1.size, len(ACTIONS)))
        strategy /= strategy.sum(axis=1, keepdims=True)
        tree = SubtreeManager(state, end_stage, end_depth, strategy)

        r1 = r1.copy()
        r2 = r2.copy()

        strategies = np.zeros((nbr_rollouts, r1.size, len(ACTIONS)))
        for t in range(nbr_rollouts):
            tree.subtree_traversal_rollout(tree.root, r1, r2)
            strategies[t] = tree.update_strategy_at_node(tree.root)

        if show_internal_values:
            print(tree.root)

        mean_strategy = strategies.mean(axis=0)

        action_probs = r1 @ mean_strategy
        action_probs /= np.sum(action_probs)

        if show_internal_values:
            print(action_probs)

        action_index = np.random.choice(len(ACTIONS), p=action_probs)

        r1 = SubtreeManager.bayesian_range_update(r1, mean_strategy, action_index)

        action = ACTIONS[action_index]
        oponent_strategy = mean_strategy

        return (action, r1, r2, oponent_strategy)
