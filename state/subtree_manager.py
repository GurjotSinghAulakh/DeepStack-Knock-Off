from enum import Enum
import random

import numpy as np

from state.state_manager import PokerGameStage, GameState, StateManager, PokerGameStateType
from game.poker_oracle import PokerOracle, PAIR_INDICES
from clients.actions import Action, ACTIONS, agent_action_index
from utils.config import RESOLVER_CHILDREN_ACTION_LIMIT, RESOLVER_N_CHILD_STATES
from utils.torchutils import to_vec_in
from nn.nn_manager import NNManager


class NodeType(Enum):
    SHOWDOWN = 0
    TERMINAL = 1
    CHANCE = 2
    PLAYER = 3
    WON = 4


class NodeVisitStatus(Enum):
    UNVISITED = 0
    VISITED_THIS_ITERATION = 1
    VISITED_PREVIOUSLY = 2


class SubtreeNode:
    def __init__(
        self,
        stage: PokerGameStage,
        state: GameState,
        depth: int,
        node_type: NodeType,
        strategy: np.ndarray,
        utility_matrix: np.ndarray,
        regrets: np.ndarray,
        values: np.ndarray,
    ) -> None:
        self.stage = stage
        self.state = state
        self.depth = depth
        self.node_type = node_type
        self.strategy = strategy
        self.children: list[tuple[Action, SubtreeNode]] = []
        self.utility_matrix = utility_matrix
        self.regrets = regrets
        self.values = values
        self.visited: NodeVisitStatus = NodeVisitStatus.UNVISITED


class SubtreeManager:

    def __init__(self, state: GameState, end_stage: PokerGameStage, end_depth: int, strategy: np.ndarray):
        """
        Generates the initial subtree for a given game state

        Assumes that the game is in a state where a player should take actions

        args:
        state: GameState - The game state to generate the subtree for
        end_stage: PokerGameStage - The stage at which the tree should
        end_depth: int - The depth at which the tree should end
        strategy: np.ndarray - The current strategy for the starting node
        """
        utility_matrix = PokerOracle.calculate_utility_matrix(tuple(state.public_info))
        self.root = SubtreeNode(
            stage=state.stage,
            state=state,
            depth=0,
            node_type=NodeType.PLAYER,
            strategy=strategy,
            utility_matrix=utility_matrix,
            regrets=np.zeros((strategy.shape[0], len(ACTIONS))),
            values=np.zeros((2, strategy.shape[0]))
        )
        self.end_stage = end_stage
        self.end_depth = end_depth
        self.root_player_index = state.current_player_index

        self.generate_initial_sub_tree(self.root)

        self.nn_manager = NNManager()

    def generate_initial_sub_tree(self, node: SubtreeNode):
        """
        Generates the initial subtree from a given node

        args:
        node: SubtreeNode - node to generate the subtree for
        """
        self.generate_children(node)

    def generate_children(self, node: SubtreeNode, action_limit: int = -1):
        """
        Adds children to the given node based on its state
        relies on the StateManager to generate the legal state/action pairs

        For chance nodes there is no action, but child states based on random deals

        args:
        node: SubtreeNode - node to generate children for
        action_limit: int - limit on the number of children to generate

        returns:
        None
        """

        if node.node_type in [NodeType.SHOWDOWN, NodeType.TERMINAL, NodeType.WON]:
            return

        if node.node_type == NodeType.CHANCE and node.children != []:
            for _, child in node.children:
                child.visited = NodeVisitStatus.UNVISITED
            return

        # Limit the number of children to generate
        child_states: list[tuple[Action, GameState]] = StateManager.get_child_states(node.state, RESOLVER_N_CHILD_STATES)
        random.shuffle(child_states)

        n_actions = 0
        for action, new_state in child_states:
            # Limit child generation
            if action is not None and action_limit != -1:
                if n_actions >= action_limit:
                    break
                n_actions += 1

            depth = node.depth + 1 if node.stage == new_state.stage else 0
            node_type = NodeType.PLAYER
            utility_matrix = node.utility_matrix.copy()

            child_actions = [a for a, _ in node.children]

            if action is not None and action in child_actions:
                child = node.children[child_actions.index(action)][1]
                child.visited = NodeVisitStatus.UNVISITED
                continue

            if new_state.stage == PokerGameStage.SHOWDOWN:
                node_type = NodeType.SHOWDOWN
            elif new_state.game_state_type == PokerGameStateType.WINNER:
                node_type = NodeType.WON
            elif new_state.stage.value > self.end_stage.value or (new_state.stage == self.end_stage and depth == self.end_depth):
                node_type = NodeType.TERMINAL
                utility_matrix = PokerOracle.calculate_utility_matrix(new_state.public_info)
            elif new_state.game_state_type == PokerGameStateType.DEALER:
                node_type = NodeType.CHANCE
                utility_matrix = PokerOracle.calculate_utility_matrix(new_state.public_info)

            new_node = SubtreeNode(
                stage=new_state.stage,
                state=new_state,
                depth=depth,
                node_type=node_type,
                strategy=node.strategy,
                utility_matrix=utility_matrix,
                regrets=node.regrets.copy(),
                values=node.values.copy(),
            )
            node.children.append((action, new_node))

    def subtree_traversal_rollout(self, node: SubtreeNode, r1: np.ndarray, r2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Recursively traverses the subtree from the given node and returns up the (estimated) values for the two players

        args:
        node: SubtreeNode - node to start the traversal from
        r1: np.ndarray - range vector for player 1
        r2: np.ndarray - range vector for player 2

        returns:
        tuple[np.ndarray, np.ndarray] - the estimated values for the two players for the given node, wrt child nodes.
        """
        v1, v2 = np.zeros_like(r1), np.zeros_like(r2)
        node.visited = NodeVisitStatus.VISITED_THIS_ITERATION
        match node.node_type:
            case NodeType.SHOWDOWN:
                # If we are at a showdown node, we can calculate the utility directly
                # based on the utility matrix and the ranges. We then multiply by the pot size to get the actual utility
                v1 = node.utility_matrix @ r2.T
                v2 = -r1 @ node.utility_matrix

                v1 *= node.state.pot
                v2 *= node.state.pot

            case NodeType.WON:
                # If we are at a won node, we can calculate the utility directly
                # based on the winner index and the pot size
                if node.state.winner_index == self.root_player_index:
                    v1 = np.ones_like(v1)
                    v2 = -np.ones_like(v2)
                else:
                    v1 = -1 * np.ones_like(v1)
                    v2 = np.ones_like(v2)

                v1 *= node.state.pot
                v2 *= node.state.pot

            case NodeType.TERMINAL:
                # If we are at a terminal node, we can estumate the utility by using one of 3 neural networks
                network = self.nn_manager.get_network(node.state.stage)
                in_vector = to_vec_in(r1, r2, node.state.public_info, node.state.pot)
                v1, v2 = network.predict_values(in_vector)
            case NodeType.PLAYER:
                # If we are at a player node, we need to calculate the utility based on the children
                # We do this by recursively calling this function on the children and updating the utility
                # based on the strategy and the values of the children, eventually returning the utility for the players,
                # either directly or by estimating the utility based on the children
                ranges = [r1, r2]

                player_index = (node.state.current_player_index + self.root_player_index) % 2

                r_p = ranges[player_index]
                r_o = ranges[1 - player_index]

                node.children = []
                self.generate_children(node, action_limit=RESOLVER_CHILDREN_ACTION_LIMIT)  # Limit the number of children to generate
                for action, child in node.children:
                    if child.visited == NodeVisitStatus.VISITED_PREVIOUSLY:
                        continue

                    child.visited = NodeVisitStatus.VISITED_PREVIOUSLY
                    a = agent_action_index(action)
                    r_p_a = SubtreeManager.bayesian_range_update(r_p, node.strategy, a)
                    r_o_a = r_o

                    action_ranges = [r_p_a, r_o_a]
                    r1_a = action_ranges[player_index]
                    r2_a = action_ranges[1 - player_index]

                    v1_a, v2_a = self.subtree_traversal_rollout(child, r1_a, r2_a)
                    v1 += node.strategy[:, a] * v1_a
                    v2 += node.strategy[:, a] * v2_a

            case NodeType.CHANCE:
                # If we are at a chance node, we need to calculate the utility based on the children
                # First, we update the ranges wrt the new public cards, then we recursively call this function on the children
                # which eventually returns either the utility or estimates the utility
                self.generate_children(node)
                S = len(node.children)
                for _, child in node.children:
                    r1_e, r2_e = r1, r2
                    r1_e = SubtreeManager.update_range_from_public_cards(r1_e, node.state.public_info)
                    r2_e = SubtreeManager.update_range_from_public_cards(r2_e, node.state.public_info)

                    v1_e, v2_e = self.subtree_traversal_rollout(child, r1_e, r2_e)
                    v1 += v1_e
                    v2 += v2_e

                v1 = v1 / S
                v2 = v2 / S

        node.values = np.array([v1, v2])

        return v1, v2

    def update_strategy_at_node(self, node: SubtreeNode) -> np.ndarray:
        """
        Updates the strategy at the given node

        args:
        node: SubtreeNode - node to update the strategy for

        returns:
        np.ndarray - updated strategy
        """
        for _, child in node.children:
            if child.visited == NodeVisitStatus.VISITED_PREVIOUSLY:
                continue
            self.update_strategy_at_node(child)
        if node.node_type == NodeType.PLAYER:
            R_t = node.regrets
            player_index = (node.state.current_player_index + self.root_player_index) % 2
            for h in PAIR_INDICES:
                node_value = node.values[player_index][h]
                for action, child in node.children:
                    if child.visited != NodeVisitStatus.VISITED_THIS_ITERATION:
                        continue
                    a = agent_action_index(action)
                    child_value = child.values[player_index][h]
                    R_t[h, a] += child_value - node_value
            node.regrets = R_t
            R_plus = np.clip(R_t, 0, None)
            R_plus_sum = np.sum(R_plus, axis=1)

            divisor = R_plus_sum[:, None]

            divisor[np.where(divisor == 0)] = 1 / R_t.shape[1]

            strategy = R_plus / divisor

            if np.sum(strategy) == 0:
                strategy = node.strategy

            node.strategy = strategy

            return node.strategy

    @staticmethod
    def bayesian_range_update(range: np.ndarray, strategy: np.ndarray, action_index: int) -> np.ndarray:
        """
        Does a bayesian update of the range vector based on the strategy and action taken

        args:
        range: np.ndarray - range vector to update
        strategy: np.ndarray - strategy vector
        action_index: int - index of the action taken

        returns:
        np.ndarray - updated range vector
        """
        p_action = np.sum(strategy[:, action_index]) / np.sum(strategy) + 0.001

        res = range * strategy[:, action_index] / p_action
        return res

    @staticmethod
    def update_range_from_public_cards(r: np.ndarray, public_cards: list[str]) -> np.ndarray:
        """
        Given a range vector and a list of new public cards, updates the range vector to exclude the new public cards

        args:
        r: np.ndarray - range vector to update
        public_cards: list[str] - list of new public cards

        returns:
        np.ndarray - updated range vector
        """
        r = r.copy()
        all_hole_cards, deck = PokerOracle.all_hole_combinations(return_deck=True)
        for card1 in public_cards:
            for card2 in deck:
                if card1 == card2:
                    continue
                hole_pair_idx = PokerOracle.cards_to_range_index(all_hole_cards, card1, card2)
                r[hole_pair_idx] = 0
        return r
