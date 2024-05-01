import numpy as np
from enum import Enum
import copy
from poker_oracle import PokerOracle
# from player import ActionType, Action
from actions import ActionType, Action


class PokerGameStage(Enum):
    PRE_FLOP = 1
    FLOP = 2
    TURN = 3
    RIVER = 4
    SHOWDOWN = 5


class PokerGameStateType(Enum):
    PLAYER = 0
    DEALER = 1
    WINNER = 2


class GameState:
    def __init__(
        self,
        stage: PokerGameStage,
        current_player_index: int,
        player_bets: np.ndarray,
        player_chips: np.ndarray,
        player_checks: np.ndarray,
        player_raised: np.ndarray,
        players_in_game: np.ndarray,
        players_all_in: np.ndarray,
        pot: int,
        bet_to_match: int,
        public_info: list[str],
        deck: list[str],
        game_state_type: PokerGameStateType = PokerGameStateType.PLAYER,
        winner_index: int = -1,
        stage_bet_count: int = 0,
    ):
        self.deck = deck
        self.stage = stage
        self.player_bets = player_bets
        self.player_chips = player_chips
        self.player_checks = player_checks
        self.player_raised = player_raised
        self.players_in_game = players_in_game
        self.players_all_in = players_all_in
        self.current_player_index = current_player_index
        self.bet_to_match = bet_to_match
        self.pot = pot
        self.public_info = public_info
        self.game_state_type = game_state_type
        self.winner_index: int = winner_index
        self.stage_bet_count = stage_bet_count

    def copy(self):
        return copy.deepcopy(self)

    def increment_player_index(self):
        self.current_player_index = (self.current_player_index + 1) % len(
            self.player_bets
        )

    def reset_for_new_round(self, redistribute_chips: bool = True):
        self.pot = 0
        self.bet_to_match = 0
        self.player_bets = np.zeros(len(self.player_bets))
        self.player_checks = np.zeros(len(self.player_bets))
        self.player_raised = np.zeros(len(self.player_bets))
        self.players_in_game = np.ones(len(self.player_bets))
        self.players_all_in = np.zeros(len(self.player_bets))
        self.deck = PokerOracle.generate_deck(randomize=True)
        self.stage = PokerGameStage.PRE_FLOP
        self.public_info = []
        self.game_state_type = PokerGameStateType.PLAYER
        self.stage_bet_count = 0

        if redistribute_chips:
            self.player_chips = np.ones(len(self.player_bets)) * 1000


class StateManager:

    @staticmethod
    def get_legal_actions(state: GameState):
        actions = [ActionType.FOLD]
        max_bet = np.max(state.player_bets)
        player_bet = state.player_bets[state.current_player_index]

        diff = max(0, max_bet - player_bet)
        can_afford_call = StateManager.has_enough(state.current_player_index, diff, state)
        all_in = state.players_all_in[state.current_player_index]

        if diff == 0 or all_in:
            actions.append(ActionType.CHECK)

        if can_afford_call and not state.player_checks[state.current_player_index]:
            actions.append(ActionType.CALL)

        # Check if they can afford the bare minimum raise
        can_afford_raise = StateManager.has_enough(state.current_player_index, max(0, diff) + 1, state)

        if can_afford_raise and not state.player_raised[state.current_player_index]:
            actions.append(ActionType.RAISE)

        return actions


        player = state["player"]
        player_bets = state["player_bets"]
        has_raised = player.has_raised
        max_bet = max(player_bets.values())

        can_call = (player_bets[player] < max_bet and player.pile >= max_bet - player_bets[player])
        can_raise = player.pile >= max_bet - player_bets[player] + 10 and not has_raised
        can_check = player_bets[player] == max_bet or all(bet == player_bets[player] for bet in player_bets.values()) or player.pile == 0
        can_fold = player.pile > 0

        legal_moves = []
        if can_call:
            legal_moves.append({"action": "call", "amount": max_bet - player_bets[player]})
        if can_raise:
            legal_moves.append({"action": "raise", "amount": max_bet - player_bets[player] + 10})
        if can_check:
            legal_moves.append({"action": "check", "amount": 0})
        if can_fold:
            legal_moves.append({"action": "fold", "amount": 0})
        return legal_moves

    @staticmethod
    def bet(player_idx: int, amount: int, state: GameState):
        s = state.copy()
        s.player_chips[player_idx] -= amount
        s.player_bets[player_idx] += amount
        s.pot += amount

        if s.player_bets[player_idx] > s.bet_to_match:
            s.bet_to_match = s.player_bets[player_idx]

        return s

    @staticmethod
    def has_enough(player_idx: int, amount: int, state: GameState):
        return state.player_chips[player_idx] >= amount

    @staticmethod
    def get_new_state_for_action(state: GameState, action: Action):
        s = state.copy()
        pot_raised = False

        if action.action_type == ActionType.FOLD:
            s.players_in_game[s.current_player_index] = False
            s.player_checks[s.current_player_index] = False
            s.player_raised[s.current_player_index] = False

        elif action.action_type == ActionType.CALL:
            diff = s.bet_to_match - s.player_bets[s.current_player_index]

            if StateManager.has_enough(s.current_player_index, diff, s):
                s = StateManager.bet(s.current_player_index, diff, s)
                s.player_checks[s.current_player_index] = True

        elif action.action_type == ActionType.CHECK:
            s.player_checks[s.current_player_index] = True

        elif action.action_type == ActionType.RAISE:
            pot_raised = True
            amount = action.amount
            s.player_raised[s.current_player_index] = True

            diff = max(0, s.bet_to_match - s.player_bets[s.current_player_index])
            total = diff + amount
            if StateManager.has_enough(s.current_player_index, total, s):
                s = StateManager.bet(s.current_player_index, total, s)

        if pot_raised:
            s.player_checks = np.zeros(len(s.player_bets), dtype=bool)
            s.player_checks[s.current_player_index] = True
            s.stage_bet_count += 1
        if np.all(s.player_checks == s.players_in_game):
            s.game_state_type = PokerGameStateType.DEALER
        if np.sum(s.players_in_game) == 1:
            s.game_state_type = PokerGameStateType.WINNER
            s.winner_index = int(np.argmax(s.players_in_game))

        s.increment_player_index()
        return s

    @staticmethod
    def progress_stage(state: GameState, deck: list[str]) -> GameState:
        """
        Creates a new state from a stage transition
        """
        s = state.copy()
        s.player_checks = np.zeros(len(s.player_bets), dtype=bool)
        s.player_checks[s.current_player_index] = True
        s.game_state_type = PokerGameStateType.PLAYER
        s.stage_bet_count = 0
        if s.stage == PokerGameStage.PRE_FLOP:
            s.stage = PokerGameStage.FLOP
            s.public_info = deck[:3]
            deck = deck[3:]
        elif s.stage == PokerGameStage.FLOP:
            s.stage = PokerGameStage.TURN
            s.public_info += deck[:1]
            deck = deck[1:]
        elif s.stage == PokerGameStage.TURN:
            s.stage = PokerGameStage.RIVER
            s.public_info += deck[:1]
            deck = deck[1:]
        elif s.stage == PokerGameStage.RIVER:
            s.stage = PokerGameStage.SHOWDOWN

        s.deck = deck

        return s
