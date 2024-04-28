from copy import deepcopy
import random


class Node:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = state.get_legal_actions(state.players[state.current_player_index])

    def select_child(self):
        from math import log, sqrt
        log_visits = log(self.visits)
        return max(self.children, key=lambda c: c.wins / c.visits + sqrt(2 * log_visits / c.visits))

    def expand(self):
        move = self.untried_moves.pop()
        next_state = deepcopy(self.state)
        next_state.do_action(move, next_state.players[next_state.current_player_index])
        child_node = Node(next_state, self, move)
        self.children.append(child_node)
        return child_node

    def update(self, result):
        self.visits += 1
        self.wins += result

def monte_carlo_tree_search(root, iterations=100):
    for i in range(iterations):
        node = root
        state = deepcopy(root.state)

        # Selection
        while node.untried_moves == [] and node.children != []:
            node = node.select_child()

        # Expansion
        if node.untried_moves != []:
            node = node.expand()

        # Rollout
        while state.get_legal_actions(state.players[state.current_player_index]) != []:
            possible_moves = state.get_legal_actions(state.players[state.current_player_index])
            move = random.choice(possible_moves)
            state.do_action(move, state.players[state.current_player_index])

        # Backpropagation
        while node is not None:
            node.update(state.get_result(node.state.players[node.state.dealer_index]))
            node = node.parent
