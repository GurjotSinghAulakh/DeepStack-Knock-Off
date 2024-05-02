import numpy as np
from deepstacklib.clients.actions import ACTIONS, Action
from deepstacklib.state.state_manager import GameState, PokerGameStage
from deepstacklib.state.subtree_manager import SubtreeManager


class Resolver:
    def resolve(
        self,
        state: GameState,
        r1: np.ndarray,
        r2: np.ndarray,
        end_stage: PokerGameStage,
        end_depth: int,
        n_rollouts: int,
    ) -> tuple[Action, np.ndarray, np.ndarray, np.ndarray]:
        """
        Re-solves the game tree from the current state

        args:
        state: GameState - current game state
        r1: np.ndarray - range vector for player 1
        r2: np.ndarray - range vector for player 2
        end_stage: PokerGameStage - stage to resolve to
        end_depth: int - depth to resolve to
        n_rollouts: int - number of rollouts to perform

        returns:
        Action - action to take
        np.ndarray - updated range vector for player 1
        np.ndarray - updated range vector for player 2
        np.ndarray - strategy of new state
        """

        # Initialize the strategy as a uniform distribution over actions
        strategy = np.ones((r1.size, len(ACTIONS)))
        strategy /= strategy.sum(axis=1, keepdims=True)

        # Create a subtree from the current state
        tree = SubtreeManager(state, end_stage, end_depth, strategy)

        # Copy the range vectors
        r1 = r1.copy()
        r2 = r2.copy()

        # Initialize the strategies array
        strategies = np.zeros((n_rollouts, r1.size, len(ACTIONS)))

        # Perform rollouts and update the strategies
        for t in range(n_rollouts):
            tree.subtree_traversal_rollout(tree.root, r1, r2)
            strategies[t] = tree.update_strategy_at_node(tree.root)

        # Calculate the mean strategy
        mean_strategy = strategies.mean(axis=0)

        action_probs = r1 @ mean_strategy     # Expected value of each action, given the current strategy
        action_probs /= np.sum(action_probs)  # Normalize the probabilities

        # Choose an action based on the probabilities
        action_index = np.random.choice(len(ACTIONS), p=action_probs)

        # Update the range vector for player 1
        r1 = SubtreeManager.bayesian_range_update(r1, mean_strategy, action_index)

        # Get the chosen action and the opponent's strategy
        action = ACTIONS[action_index]
        oponent_strategy = mean_strategy

        return (action, r1, r2, oponent_strategy)
