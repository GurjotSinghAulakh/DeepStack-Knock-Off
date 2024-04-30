from state_manager import StateManager
from dataclasses import dataclass
from poker_oracle import PokerOracle
import numpy as np


@dataclass
class Resolver:
    state_manager: StateManager
    poker_oracle: PokerOracle

    def bootstrap_nn(self, stage):
        deck = self.poker_oracle.generate_deck()
        pot = 20
        endstage = None
        enddepth = None
        T = 1

        if stage == "river":
            pot *= 3
            card_amount = 5
        elif stage == "turn":
            pot *= 2
            card_amount = 4
        elif stage == "flop":
            card_amount = 3

        public_cards = np.random.choice(deck, card_amount, replace=False)
        range_r1_probs = np.random.rand(len(deck))
        range_r2_probs = np.random.rand(len(deck))

        public_cards_index = [deck.index(card) for card in public_cards]

        range_r1_probs[public_cards_index] = 0
        range_r2_probs[public_cards_index] = 0

        range_r1_probs /= range_r1_probs.sum()
        range_r2_probs /= range_r2_probs.sum()

        state = {} 
        state["type"] == "playeraction"
        state["active_player"] == 1
        state["r1"] = range_r1_probs
        state["r2"] = range_r2_probs
        state["current_strategy"] = lambda h, a: 1 / len(self.state_manager.get_legal_actions(state))

        subtrees = self.subtree_traversal_rollout(state, range_r1_probs, range_r2_probs, endstage, enddepth)
        # return self.state_manager.get_legal_actions(state)


    def subtree_traversal_rollout(self, state, endstage, enddepth):
        if state["type"] == "showdown":
            utility_matrix = self.poker_oracle.generate_utility_matrix(state)
            v1 = np.dot(utility_matrix, state["r2"].T)
            v2 = -np.dot(state["r1"], utility_matrix)
        elif state["stage"] == endstage and state["depth"] == enddepth:
            v1, v2 = self.run_neural_network(state["stage"], state)
        elif state["type"] == "playeraction":
            P = state["active_player"]
            vP = np.zeros_like(state["r1"])
            vO = np.zeros_like(state["r2"])

            for action in self.state_manager.get_legal_actions(state):
                if P == 1:
                    updated_range_P = self.bayesian_range_update(state["r1"], action, state["current_strategy"])
                    updated_range_O = state["r2"].copy()
                else:
                    updated_range_P = self.bayesian_range_update(state["r2"], action, state["current_strategy"])
                    updated_range_O = state["r1"].copy()

                new_state = self.state_manager.get_next_state(state, action, updated_range_P, updated_range_O)
                vP_action, vO_action = self.subtree_traversal_rollout(new_state, endstage, enddepth)

                for h in state["hole_pairs"]:
                    vP[h] += state["current_strategy"](h, action) * vP_action[h]
                    vO[h] += state["current_strategy"](h, action) * vO_action[h]
            if P == 1:
                v1, v2 = vP, vO
            else:
                v1, v2 = vO, vP
        else:
            v1 = np.zeros_like(state["r1"])
            v2 = np.zeros_like(state["r2"])

            for event in self.state_manager.get_events(state):
                new_state = self.state_manager.get_next_state(state, event)
                v1_event, v2_event = self.subtree_traversal_rollout(new_state, endstage, enddepth)

                for h in state["hole_pairs"]:
                    v1[h] += v1_event[h] / len(state["events"])
                    v2[h] += v2_event[h] / len(state["events"])

        return v1, v2

    def bayesian_range_update(self, range, action, current_strategy):
        updated_range = range.copy()
        for h in range:
            updated_range[h] *= current_strategy(h, action)
        updated_range /= updated_range.sum()
        return updated_range


    def run_neural_network(self, stage, state):
        pass


    # def generate_subtrees(self, state, stage):
    #     legal_actions = self.state_manager.get_legal_actions(state)
    #     return [self.bootstrap_nn(stage) for action in legal_actions]