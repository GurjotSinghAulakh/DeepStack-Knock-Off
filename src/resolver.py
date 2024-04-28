from lib2to3.pytree import Node
import numpy as np
import tensorflow as tf
import itertools


# Assuming 52-card deck and standard poker rules
def generate_all_hole_pairs():
    deck = list(range(52))  # Assuming each card is represented by a unique integer
    return list(itertools.combinations(deck, 2))


class Resolver:
    def __init__(self, game_manager, state_manager, poker_oracle):
        self.game_manager = game_manager
        self.state_manager = state_manager
        self.poker_oracle = poker_oracle
        self.hole_pairs = generate_all_hole_pairs()

    def resolve(self, current_state, r1, r2, end_stage, end_depth, T):
        root = self.generate_initial_subtree(current_state, end_stage, end_depth)
        strategies = []
        for t in range(T):
            v1, v2 = self.subtree_traversal_rollout(current_state, r1, r2, end_stage, end_depth)
            strategy = self.update_strategy(root)
            strategies.append(strategy)

        avg_strategy = np.mean(strategies, axis=0)
        a_star = np.random.choice(range(len(avg_strategy)), p=avg_strategy)
        r1_a_star = self.bayesian_range_update(r1, a_star, avg_strategy)
        return a_star, r1_a_star, r2

    def subtree_traversal_rollout(self, S, r1, r2, end_stage, end_depth):
        if self.showdown_state(S):
            v1 = np.dot(self.poker_oracle.utility_matrix(S), r2)
            v2 = -np.dot(r1, self.poker_oracle.utility_matrix(S))
            return v1, v2
        elif self.stage(S) == end_stage and self.depth(S) == end_depth:
            v1, v2 = self.run_neural_network(self.stage(S), S, r1, r2)
            return v1, v2
        elif self.player_state(S):
            v1, v2 = np.zeros(self.game_manager.num_actions), np.zeros(self.game_manager.num_actions)
            P = self.player(S)
            for a in self.actions(S):
                r1_a = self.bayesian_range_update(r1, a)
                v1_a, v2_a = self.subtree_traversal_rollout(self.state_manager.successor(S, a), r1_a, r2, end_stage, end_depth)
                for h in self.hole_pairs:
                    v1[h] += self.strategy(S)[h][a] * v1_a[h]
                    v2[h] += self.strategy(S)[h][a] * v2_a[h]
            return v1, v2
        else:
            v1, v2 = np.zeros(self.game_manager.num_actions), np.zeros(self.game_manager.num_actions)
            for e in self.events(S):
                v1_e, v2_e = self.subtree_traversal_rollout(e, r1, r2, end_stage, end_depth)
                for h in self.hole_pairs:
                    v1[h] += v1_e[h]
                    v2[h] += v2_e[h]
            return v1, v2

    def update_strategy(self, node):
        S = node.state
        for C in self.state_manager.successors(node):
            self.update_strategy(C)
        if self.player_state(S):
            P = self.player(S)
            for h in self.hole_pairs:
                for a in self.actions(S):
                    self.cumulative_regret(S)[h][a] += (self.value_vector(S, a, P)[h] - self.value_vector(S, P)[h])
                    self.positive_regret(S)[h][a] = max(0, self.cumulative_regret(S)[h][a])
            for h in self.hole_pairs:
                for a in self.actions(S):
                    self.strategy(S)[h][a] = self.positive_regret(S)[h][a] / np.sum([self.positive_regret(S)[h][a_prime] for a_prime in self.actions(S)])
        return self.strategy(S)

    def generate_initial_subtree(self, current_state, end_stage, end_depth):
        # Generate initial subtree recursively based on current state, end stage, and end depth
        if self.stage(current_state) == end_stage and self.depth(current_state) == end_depth:
            # Base case: reached end stage and end depth
            return Node(current_state)
        else:
            # Recursive case: generate successors and continue building subtree
            root = Node(current_state)
            for action in self.actions(current_state):
                successor_state = self.state_manager.successor(current_state, action)
                child_node = self.generate_initial_subtree(successor_state, end_stage, end_depth)
                root.add_successor(child_node, action)
            return root

    def bayesian_range_update(self, r, a, strategy=None):
        # Update range using Bayesian update
        if strategy is not None:
            # Update range using strategy
            # Here, let's assume a simple linear update based on strategy and action
            action_prob = strategy[a]
            updated_range = r * action_prob
            updated_range /= np.sum(updated_range)  # Normalize the range
            return updated_range
        else:
            # Default update without strategy
            return r

    def run_neural_network(self, stage, S, r1, r2):
        # Run neural network to produce value vectors
        # Implementation depends on the specifics of running neural networks
        input_data = self.prepare_input_data(stage, S, r1, r2)
        value_vectors = self.neural_networks[stage].predict(input_data)
        v1, v2 = value_vectors[:, 0], value_vectors[:, 1]  # Assuming output shape (num_actions, num_players)
        return v1, v2

    def showdown_state(self, S):
        # Identify if state is a showdown state
        # Assuming a simple check based on the stage of the game
        return self.stage(S) == "Showdown"

    def player_state(self, S):
        # Identify if state is a player state
        # Assuming a simple check based on the stage of the game
        return self.stage(S) == "Player"

    def stage(self, S):
        # Get the stage of a state
        # Assuming the stage is stored as an attribute of the state
        return S.stage

    def depth(self, S):
        # Get the depth of a state
        # Assuming the depth is stored as an attribute of the state
        return S.depth

    def player(self, S):
        # Get the active player in a state
        # Assuming the active player is stored as an attribute of the state
        return S.active_player

    def actions(self, S):
        # Get available actions in a state
        # Assuming the available actions are stored as an attribute of the state
        return S.available_actions

    def strategy(self, S):
        # Access strategy at a state
        # Assuming the strategy is stored as an attribute of the state
        return S.strategy

    def cumulative_regret(self, S):
        # Access cumulative regret at a state
        # Assuming the cumulative regret is stored as an attribute of the state
        return S.cumulative_regret

    def positive_regret(self, S):
        # Access positive regret at a state
        # Assuming the positive regret is stored as an attribute of the state
        return S.positive_regret

    def value_vector(self, S, action=None, player=None):
        # Access value vectors
        # Assuming value vectors are stored as an attribute of the state
        if action is None and player is None:
            return S.value_vector
        elif action is not None:
            return S.value_vector[action]
        elif player is not None:
            return S.value_vector[:, player]

    def events(self, S):
        # Access chance events at a state
        # Assuming chance events are stored as an attribute of the state
        return S.events

    def prepare_input_data(self, stage, S, r1, r2):
        # Prepare input data for neural network based on the stage, current state, and ranges
        # Implementation depends on the specifics of preparing input data
        # Here's a placeholder implementation
        input_data = [stage, S, r1, r2]  # Placeholder, replace with actual implementation
        return input_data

    def train_neural_networks(self):
        # Train neural networks based on data
        # Implementation depends on the specifics of training neural networks
        # Here's a placeholder implementation
        for stage in self.game_manager.stages:
            # Assuming you have a list of stages in your game manager
            data = self.prepare_training_data(stage)
            # Assuming you have a method prepare_training_data to generate training data
            self.neural_networks[stage].train(data)




