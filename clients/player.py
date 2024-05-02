import numpy as np
from dataclasses import dataclass, field
from game.poker_oracle import PokerOracle
from state.subtree_manager import SubtreeManager
from state.state_manager import StateManager, GameState, PokerGameStage
from clients.actions import ActionType, Action, ACTIONS, agent_action_index
from utils.config import COMBINATIONS, RESOLVER_ROLLOUTS
from clients.resolver import Resolver


@dataclass
class Player:
    name: str
    hole_cards: list[str] = field(default_factory=list)
    active: bool = True
    has_raised: bool = False
    has_folded: bool = False
    range_size: int = COMBINATIONS
    agent_type: str = "random"

    def __post_init__(self):
        """
        Post initialization setup
        """
        # Initialize the player's ranges and opponent's strategy
        self.r1 = np.ones(self.range_size) / self.range_size
        self.r2 = np.ones(self.range_size) / self.range_size
        self.opponent_strategy = np.ones((self.range_size, len(ACTIONS))) / len(ACTIONS)

    def reset_ranges(self):
        """
        Reset the player's ranges of hands to uniform distribution
        """
        self.r1 = np.ones(self.range_size) / self.range_size
        self.r2 = np.ones(self.range_size) / self.range_size

    def inform(self, action_taken: Action, player: 'Player'):
        """
        Inform the player of the action taken by the opponent, updating the opponent's strategy

        Args:
        action_taken: Action - action taken by the opponent
        player: Player - the opponent

        Returns:
        None
        """
        if player.name == self.name:
            return

        action_idx = agent_action_index(action_taken)

        updated_r2 = SubtreeManager.bayesian_range_update(self.r2, self.opponent_strategy, action_idx)
        if np.sum(updated_r2) == 0:
            return

        self.r2 = updated_r2

    def choose_action(self, actions: list[Action], state: GameState, oracle=PokerOracle()) -> Action:
        """
        Choose an action based on the current state of the game

        Args:
        actions: list[Action] - list of legal actions
        state: GameState - the current game state
        oracle: PokerOracle - the oracle to use for evaluating win probabilities

        Returns:
        Action - the chosen action
        """

        # Get the indexes of the legal actions
        action_indexes = {agent_action_index(action): action.action_type for action in actions}
        print(f"\nLegal actions for {self.name}: {action_indexes}, Hole cards: {self.hole_cards}, community_cards: {state.public_info}")
        
        # Ask the user to choose an action
        while True:
            try:
                chosen_idx = int(input("Choose an action: "))
                if chosen_idx in action_indexes.keys():
                    break
                else:
                    print("Invalid action number, please try again.")
            except ValueError:
                print("Invalid input, please enter a number.")
        return action_indexes[chosen_idx]

    def take_action(self, state: GameState) -> Action:
        """
        Take an action based on the current state of the game

        Args:
        state: GameState - the current game state

        Returns:
        Action - the chosen action
        """

        # Get the legal actions for the current state
        legal_actions = StateManager.get_legal_actions(state)

        # Choose an action based on the agent type
        if self.agent_type == "random":
            idx = np.random.randint(len(legal_actions))
            act_type = legal_actions[idx]
        elif self.agent_type == "rollout":
            act = self.rollout_action(legal_actions, state)
            act_type = act.action_type
        elif self.agent_type == "resolve":
            act = self.resolve_action(legal_actions, state)
            act_type = act.action_type
        elif self.agent_type == "human":
            act = self.choose_action(legal_actions, state)
            act_type = act.action_type

        # If the action is a raise, set the amount to 10
        if act_type == ActionType.RAISE:
            amount = 10
        else:
            amount = 0
        return Action(act_type, amount)

    def __hash__(self) -> int:
        """
        Hash the player based on their name, to allow for dictionary lookups
        """
        return hash(self.name)

    def __repr__(self) -> str:
        """
        Represent the player as a string
        """
        return "Human - " + self.name if self.is_human else "AI - " + self.name

    def rollout_action(self, legal_actions: list[Action], game_state: GameState) -> Action:
        """
        Choose an action based on the current state of the game and rollouts

        Args:
        legal_actions: list[Action] - list of legal actions
        game_state: GameState - the current game state

        Returns:
        Action - the chosen action
        """

        # Get the win probability for the current state
        win_probability = 0
        if game_state.stage == PokerGameStage.PRE_FLOP:
            win_probability = PokerOracle().get_cheat_sheet_probs(self.hole_cards)
        else:
            win_probability = PokerOracle().evaluate_hole_pair_win_probability(self.hole_cards, game_state.public_info, num_simulations=5000)

        # Choose an action based on the win probability
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

    def resolve_action(self, legal_actions: list[Action], game_state: GameState) -> Action:
        """
        Choose an action based on the current state of the game and resolving the game tree

        Args:
        legal_actions: list[Action] - list of legal actions
        game_state: GameState - the current game state

        Returns:
        Action - the chosen action
        """

        # Initialize the resolver and the state
        resolver = Resolver()
        state = game_state
        r1 = self.r1
        r2 = self.r2
        end_depth = 1

        # Determine the end stage based on the current stage
        current_stage = game_state.stage
        if current_stage == PokerGameStage.PRE_FLOP:
            return self.rollout_action(legal_actions, game_state)
        elif current_stage == PokerGameStage.FLOP:
            end_stage = PokerGameStage.TURN
        elif current_stage == PokerGameStage.TURN:
            end_stage = PokerGameStage.RIVER
        else:
            end_stage = PokerGameStage.SHOWDOWN

        # Resolve the game tree and choose an action
        action, self.r1, self.r2, self.opponent_strategy = resolver.resolve(
            state, r1, r2, end_stage, end_depth, RESOLVER_ROLLOUTS
        )
        return action
