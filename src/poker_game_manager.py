# Poker Game Manager
import random
from poker_oracle import PokerOracle
from poker_state_manager import PokerStateManager
from src.resolver import Resolver

class PokerGameManager:
    def __init__(self, players):
        self.players = players
        self.state_manager = PokerStateManager()
        self.poker_oracle = PokerOracle()
        self.game_manager = PokerGameManager(players)
        self.resolver = Resolver(game_manager, state_manager, poker_oracle)
        self.deck = self.generate_deck()
        self.pot = 0
        self.max_raises = 3  # Maximum number of raises allowed in a stage
        self.raise_count = 0  # Counter for the number of raises in the current stage


    def rotate_dealer(self):
        self.players.append(self.players.pop(0))  # Rotate the list of players

    def generate_deck(self):
        # Generate a standard 52-card deck
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']
        deck = [(rank, suit) for rank in ranks for suit in suits]
        return deck

    def deal_cards(self):
        random.shuffle(self.deck)  # Shuffle the deck
        
        # Deal private cards to each player
        for player in self.players:
            player.private_cards = [self.deck.pop(), self.deck.pop()]
        
        # Deal public cards for the flop, turn, and river stages
        self.state_manager.set_flop([self.deck.pop() for _ in range(3)])
        self.state_manager.set_turn(self.deck.pop())
        self.state_manager.set_river(self.deck.pop())

    def handle_player_action(self, player, action):
        if action == "Fold":
            self.players.remove(player)
        elif action == "Check":
            pass  # Do nothing
        elif action == "Call":
            # Player matches the current bet
            player.bet_chips(self.state_manager.current_bet - player.current_bet)
            self.manage_pot()
        elif action.startswith("Raise"):
            # Player raises the bet
            raise_amount = int(action.split()[1])
            player.bet_chips(raise_amount)
            self.manage_pot()
            self.raise_count += 1

    def manage_pot(self):
        total_bet = sum(player.current_bet for player in self.players)
        self.pot += total_bet
        for player in self.players:
            player.current_bet = 0

    def determine_winner(self):
        # Determine the best hand for each player
        player_hands = {}
        for player in self.players:
            hole_cards = player.private_cards
            public_cards = self.state_manager.public_cards
            player_hands[player] = self.oracle.evaluate_hand(hole_cards, public_cards)

        # Find the winner(s) with the best hand
        winners = []
        max_hand_value = max(player_hands.values())
        for player, hand_value in player_hands.items():
            if hand_value == max_hand_value:
                winners.append(player)

        # Distribute the pot to the winner(s)
        pot_per_winner = self.pot // len(winners)
        for winner in winners:
            winner.win_chips(pot_per_winner)

    def enforce_restrictions(self):
        if self.raise_count >= self.max_raises:
            # Limit the number of raises in a stage
            # Reset raise count for the next stage
            self.raise_count = 0
    
    def run_poker_game(self):
        # Main method to run a single poker game
        self.rotate_dealer()  # Rotate the dealer position
        self.deal_cards()  # Deal cards to players
        self.enforce_restrictions()  # Enforce restrictions at the start of the game

        # Loop through stages (pre-flop, flop, turn, river)
        for stage in ["Pre-flop", "Flop", "Turn", "River"]:
            # Print current stage
            print("Current Stage:", stage)

            # Check if only one player left, declare winner
            if len(self.players) == 1:
                print("Only one player remaining. Player", self.players[0], "is the winner!")
                return

            # Handle player actions for each player in the current stage
            for player in self.players:
                action = player.choose_action(stage)
                self.handle_player_action(player, action)

            # Manage pot and enforce restrictions at the end of each stage
            self.manage_pot()
            self.enforce_restrictions()

            # If it's the river stage, determine the winner and declare
            if stage == "River":
                self.determine_winner()
                print("Game over. Winners declared and pot distributed.")
                return

    def run_poker_series(self, num_games):
        # Method to run a series of poker games
        for game_num in range(num_games):
            print("Game", game_num + 1, "of", num_games)
            self.run_poker_game()
            print("\n")

    def add_player(self, player):
        # Method to add a player to the game
        self.players.append(player)
        print("Player", player, "added to the game.")

    def remove_player(self, player):
        # Method to remove a player from the game
        if player in self.players:
            self.players.remove(player)
            print("Player", player, "removed from the game.")
        else:
            print("Player", player, "not found in the game.")

# Example usage:

# Define or initialize game_manager, state_manager, and poker_oracle here
state_manager = PokerStateManager()
poker_oracle = PokerOracle()

# Initialize players and poker game manager
players = ["Player1", "Player2", "Player3"]
game_manager = PokerGameManager(players)

# Add or remove players as needed
game_manager.add_player("Player4")
game_manager.remove_player("Player2")

# Run a single poker game or a series of games
game_manager.run_poker_game()
# game_manager.run_poker_series(5)  # Run 5 poker games in a series



