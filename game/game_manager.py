from clients.player import Player
import numpy as np
from state.state_manager import StateManager
from game.poker_oracle import PokerOracle
from state.state_manager import GameState, PokerGameStage, PokerGameStateType
from utils.config import PLAYER_CHIPS
from clients.player import ActionType


class GameManager:
    def __init__(self, players: list[Player]):
        if len(players) < 2:
            raise ValueError("A poker game must have at least 2 players")
        if len(players) > 6:
            raise ValueError("This engine only supports up to 6 players")

        self.players = players
        self.blind_amount = 5
        self.small_blind = 1
        self.big_blind = 2 % len(players)
        self.game_state = GameState(
            stage=PokerGameStage.PRE_FLOP,
            current_player_index=0,
            player_bets=np.zeros(len(self.players)),
            player_chips=np.ones(len(self.players)) * PLAYER_CHIPS,
            player_checks=np.zeros(len(self.players)),
            player_raised=np.zeros(len(self.players)),
            players_in_game=np.ones(len(self.players)),
            players_all_in=np.zeros(len(self.players), dtype=bool),
            pot=0,
            bet_to_match=0,
            public_info=[],
            deck=PokerOracle.generate_deck(randomize=True),
        )

    def start_game(self, num_games: int = 10):
        print(f"\n{'=' * 20}")
        print("Starting a new round!")
        self.game_state.reset_for_new_round(redistribute_chips=False)

        # for player in self.players:
        #     player.reset_ranges()

        self.claim_blinds()
        self.deal_cards()

        while True:
            self.display()
            if self.game_state.game_state_type == PokerGameStateType.WINNER:
                self.winner = self.players[self.game_state.winner_index]
                break

            elif self.game_state.game_state_type == PokerGameStateType.DEALER:
                print("Dealing new cards")
                self.game_state = StateManager.progress_stage(self.game_state, self.game_state.deck)
                continue

            if self.game_state.stage == PokerGameStage.SHOWDOWN:
                self.showdown()
                break

            print()

            # Checks to see how game should progress
            if not self.game_state.players_in_game[self.game_state.current_player_index]:
                # Skip folded players
                continue

            # Player action node
            player = self.players[self.game_state.current_player_index]

            # Get action from player
            action = player.take_action(self.game_state)
            self.game_state = StateManager.get_new_state_for_action(self.game_state, action)
            for p in self.players:
                p.inform(action, player)

            # Display action info
            print(f"Player: {player.name} did: {action.action_type}")
            if action.action_type == ActionType.RAISE:
                print(f"Amount: {action.amount}")

        if self.winner is not None:
            print(f"{self.winner.name} won the game!")
            print(f"Winnings: {self.game_state.pot}")
            total_index = self.players.index(self.winner)
            self.game_state.player_chips[total_index] += self.game_state.pot

        self.rotate_blinds()

        for player in self.players:
            index = self.players.index(player)
            if self.game_state.player_chips[index] <= 0:
                print(f"{player.name} is out of chips!")
                return

        if num_games == 0:
            print("Game over!")
            print(f"Players have: {self.game_state.player_chips} chips")
            return

        self.start_game(num_games - 1)

    def showdown(self) -> None:
        """
        Showdown the game
        """
        print("=== SHOWDOWN! ===")
        remaining_players = []
        for i, player in enumerate(self.players):
            if self.game_state.players_in_game[i]:
                remaining_players.append(player)

        winner = PokerOracle().compare_hands(remaining_players, self.game_state.public_info)[0]
        self.winner = winner

    def claim_blinds(self) -> None:
        """
        Claim the blinds
        """
        print("Claiming blinds")
        print(f"Small blind: {self.players[self.small_blind].name}")
        print(f"Big blind: {self.players[self.big_blind].name}")
        print()
        self.game_state = StateManager.bet(self.small_blind, self.blind_amount, self.game_state)
        self.game_state = StateManager.bet(self.big_blind, self.blind_amount * 2, self.game_state)

        self.game_state.current_player_index = (self.big_blind + 1) % len(self.players)

    def rotate_blinds(self) -> None:
        """
        Rotate the blinds to the next player
        """
        self.small_blind = (self.small_blind + 1) % len(self.players)
        self.big_blind = (self.big_blind + 1) % len(self.players)

        self.player_checks = np.zeros(len(self.players), dtype=bool)
        self.player_raised = np.zeros(len(self.players), dtype=bool)

    def deal_cards(self) -> None:
        """
        Deal cards to players
        """
        for player in self.players:
            player.hole_cards = self.game_state.deck[:2]
            self.game_state.deck = self.game_state.deck[2:]

    def display(self) -> None:
        """
        Visualize CLI
        """
        print(f"Bet to match: {self.game_state.bet_to_match}")
        print(f"Pot: { self.game_state.pot}")
        print(f"Public cards: {self.game_state.public_info}")
        for i, player in enumerate(self.players):
            print(f"{player.name}: (bets: {self.game_state.player_bets[i]}), chips: {self.game_state.player_chips[i]}")
        print()
