from player import Player
from dataclasses import dataclass, field
import numpy as np
from state_manager import StateManager
from config import MAX_STAGE_RAISES
from poker_oracle import PokerOracle
from itertools import cycle
# stages = set(["preflop", "flop", "turn", "river"])


@dataclass
class GameManager:
    players: list[Player]
    lost_players: list[Player] = field(init=False)
    player_count: int = field(init=False)
    current_pile: int = field(init=False)
    current_deck: list[str] = field(init=False)
    player_bets: dict[Player, int] = field(init=False)
    folded_pot: int = field(init=False)
    cycle_players_big: cycle = field(init=False)
    cycle_players_small: cycle = field(init=False)
    oracle: PokerOracle = PokerOracle()
    state_manager: StateManager = StateManager()

    def __post_init__(self):
        self.player_count = len(self.players)
        self.lost_players = []
        self.current_pile = 0
        self.folded_pot = 0
        self.cycle_players_big = cycle(self.players)
        self.cycle_players_small = cycle(self.players)

    def play_game(self):
        while self.player_count > 1:
            self.current_deck = self.oracle.generate_deck()
            np.random.shuffle(self.current_deck)
            self.public_cards = None
            self.play_stage("preflop")
            self.play_stage("flop")
            self.play_stage("turn")
            self.play_stage("river")
            self.end_hand()
        print(f"{self.get_active_players()[0].name} has won the game")
        print(f"{', '.join([p.name for p in self.lost_players])} has lost the game")

    def get_active_non_folded_players(self):
        return [player for player in self.players if player.active and not player.has_folded]

    def get_active_players(self):
        return [player for player in self.players if player.active]

    def reset_player_folds(self):
        for player in self.get_active_players():
            player.has_folded = False

    def get_blinds(self):
        small_blind = next(self.cycle_players_small)
        big_blind = next(self.cycle_players_big)
        while not small_blind.active:
            small_blind = next(self.cycle_players_small)
            big_blind = next(self.cycle_players_big)
        if small_blind is big_blind:
            small_blind = next(self.cycle_players_small)
        print(f"Big blind (10): {big_blind.name}, Small blind(5): {small_blind.name}\n")
        return big_blind, small_blind

    def deal_hole_cards(self):
        for player in self.get_active_players():
            print(f"Dealing hole cards to {player}")
            print(self.current_deck[:2])
            player.hole_cards = self.current_deck[:2]
            self.current_deck = self.current_deck[2:]

    def play_stage(self, stage):
        print(f"\n\n\n----------Starting {stage} stage----------")
        if stage == "preflop":
            self.reset_player_folds()
            self.folded_pot = 0
            self.deal_hole_cards()
            self.reset_player_bets()
            self.reset_pile()

            big_blind, small_blind = self.get_blinds()
            self.update_pile(5, small_blind)
            self.update_pile(10, big_blind)

        elif stage == "flop":
            self.public_cards = self.current_deck[:3]
            self.current_deck = self.current_deck[3:]
            print(f"Community cards: {self.public_cards}")
        elif stage == "turn":
            self.public_cards.append(self.current_deck.pop(0))
            print(f"Community cards: {self.public_cards}")
        elif stage == "river":
            self.public_cards.append(self.current_deck.pop(0))
            print(f"Community cards: {self.public_cards}")

        if len(self.get_active_non_folded_players()) == 1:
            return

        for player in self.get_active_non_folded_players():
            player.has_raised = False

        while True:
            for player in self.get_active_players():
                if player.has_folded or not player.active:
                    continue
                state = {"player": player, "player_bets": self.player_bets, "stage": stage, "public_cards": self.public_cards}
                legal_actions = self.state_manager.get_legal_actions(state)
                player_action = player.choose_action(legal_actions, state, self.oracle)
                if player_action["action"] == "fold":
                    self.folded_pot += self.player_bets[player]
                    player.has_folded = True
                    if len(self.get_active_non_folded_players()) == 1:
                        return
                elif player_action["action"] == "raise":
                    player.has_raised = True

                self.update_pile(player_action["amount"], player)

            # if every player has the same bet, or if an active player has gone all in, end the betting round
            max_player_bet = 0
            for player in self.get_active_non_folded_players():
                if self.player_bets[player] > max_player_bet:
                    max_player_bet = self.player_bets[player]

            max_or_equal = []
            for player in self.get_active_non_folded_players():
                if self.player_bets[player] == max_player_bet or player.pile == 0:
                    max_or_equal.append(player)

            if len(max_or_equal) == len(self.get_active_non_folded_players()):
                return

    def update_pile(self, amount, player):
        if amount >= player.pile:
            amount = player.pile
        self.current_pile += amount
        self.player_bets[player] += amount
        player.pile -= amount

    def reset_pile(self):
        self.current_pile = 0

    def reset_player_bets(self):
        self.player_bets = {player: 0 for player in self.get_active_players()}

    def end_hand(self):
        winners = self.oracle.compare_hands(self.get_active_non_folded_players(), self.public_cards)

        all_bets = [self.player_bets[player] for player in self.get_active_non_folded_players()]
        min_allin = min(all_bets)
        main_pot = sum(min(min_allin, bet) for bet in all_bets) + self.folded_pot
        excess_bets = {player: max(0, self.player_bets[player] - min_allin) for player in self.get_active_non_folded_players()}

        # Return excess bets
        for player, excess in excess_bets.items():
            player.pile += excess
            self.current_pile -= excess  # Adjust the current pile to remove the returned bets

        # Distribute main pot
        if len(winners) == 1:
            winners[0].pile += main_pot
        else:
            for winner in winners:
                winner.pile += main_pot // len(winners)

        self.reset_pile()
        print(f"{[winner.name for winner in winners]} wins the hand")

        # Check for players who have gone bust
        newly_lost_players = [player for player in self.get_active_players() if player.pile == 0]
        for player in newly_lost_players:
            player.active = False
        self.player_count = len(self.get_active_players())

        self.lost_players.extend(newly_lost_players)

        print(f"Player piles: {[(player.name, player.pile) for player in self.players]}")
