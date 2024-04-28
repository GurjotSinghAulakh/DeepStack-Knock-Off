import random
from collections import Counter


class Card:
    SUITS = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
    RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank
    
    @property
    def value(self):
        return Card.RANKS.index(self.rank) + 2  # value from 2 to 14, Ace is high

    def __repr__(self):
        return f"{self.rank} of {self.suit}"

class Deck:
    def __init__(self):
        self.cards = [Card(suit, rank) for suit in Card.SUITS for rank in Card.RANKS]
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self):
        try:
            return self.cards.pop()
        except IndexError:
            return None  # No more cards in the deck

class Player:

    def __init__(self, name):
        self.name = name
        self.hand = []
        self.chips = 100000
        self.in_play = True
        self.current_bet = 0

    def add_card(self, card):
        self.hand.append(card)

    def bet(self, amount):
        if amount <= self.chips:
            self.chips -= amount
            return amount
        else:
            raise ValueError("Not enough chips")

    def fold(self):
        self.in_play = False

class TexasHoldem:
    def __init__(self, players):
        self.players = []
        self.players = players
        self.pot = 0
        self.deck = Deck()
        self.community_cards = []
        self.dealer_position = 0 
        self.current_bet = 0
        self.round_finished = False

    def export_state(self):
        # Add your logic here to export the current state of the game
        state = {
            "players": [player.name for player in self.players],
            "pot": self.pot,
            "community_cards": [str(card) for card in self.community_cards],
            "dealer_position": self.dealer_position,
            "current_bet": self.current_bet,
            "round_finished": self.round_finished
        }
        return state

    def deal_hole_cards(self):
        for _ in range(2):
            for player in self.players:
                if player.in_play:
                    player.add_card(self.deck.deal())

    def rotate_dealer(self):
        self.dealer_position = (self.dealer_position + 1) % len(self.players)

    def handle_betting_round(self):
        for player in self.players:
            if player.in_play:
                actions = self.legal_actions(player)
                chosen_action = random.choice(actions)  # For simplicity, choosing random action
                self.process_action(player, chosen_action)

    def add_player(self, player):
        self.players.append(player)

    def start_round(self):
        self.rotate_dealer()
        self.deck = Deck()  
        self.community_cards = []
        self.community_cards.clear()
        self.deal_cards(2)
        self.betting_round()
        self.flop()
        self.betting_round()
        self.turn()
        self.betting_round()
        self.river()
        self.betting_round()
        self.determine_winner()
        for player in self.players:
            player.in_play = True
            player.hand = []
            player.current_bet = 0
            player.add_card(self.deck.deal())
            player.add_card(self.deck.deal())
        self.deal_hole_cards()
        self.handle_betting_round()

    def round_over(self):
        # Example condition for ending a round
        active_players = [p for p in self.players if p.in_play]
        if len(active_players) <= 1 or self.all_bets_settled():
            self.round_finished = True
        return self.round_finished

    def all_bets_settled(self):
        # Assuming a method to check if all bets are settled for this round
        return all(player.current_bet == self.current_bet for player in self.players if player.in_play)

    def legal_actions(self, player):
        actions = ["fold"]
        if player.current_bet < self.current_bet:
            if player.chips > (self.current_bet - player.current_bet):
                actions.append("call")
            else:
                actions.append("all-in")
        else:
            actions.append("check")
        if player.chips > self.current_bet:
            actions.append("raise")
        return actions
    
    def process_action(self, player, action):
        if action == "fold":
            player.fold()
        elif action == "call":
            bet_amount = self.current_bet - player.current_bet
            self.pot += player.bet(bet_amount)
        elif action == "check":
            pass  # No action needed
        elif action == "raise":
            raise_amount = player.chips // 2  # Example logic
            bet_amount = self.current_bet - player.current_bet + raise_amount
            self.pot += player.bet(bet_amount)
            self.current_bet = player.current_bet + raise_amount

    def deal_cards(self, count):
        for _ in range(count):
            for player in self.players:
                card = self.deck.deal()
                if card:
                    player.add_card(card)

    def flop(self):
        self.deal_community_cards(3)

    def turn(self):
        self.deal_community_cards(1)

    def river(self):
        self.deal_community_cards(1)

    def deal_community_cards(self, count):
        for _ in range(count):
            card = self.deck.deal()
            if card:
                self.community_cards.append(card)

    def betting_round(self):
        for player in self.players:
            if player.in_play:
                try:
                    bet = player.bet(10)  # Simplified fixed betting for now
                    self.pot += bet
                except ValueError as e:
                    print(str(e))
                    player.fold()
                    print(f"{player.name} folds.")

    def determine_winner(self):
        best_score = None
        winning_player = None

        for player in self.players:
            combined_hand = player.hand + self.community_cards
            hand_score = self.evaluate_hand(combined_hand)
            if not best_score or hand_score > best_score:
                best_score = hand_score
                winning_player = player

        print(f"{winning_player.name} wins the round with a hand rank of {best_score[0]} and score {best_score[1]}.")
        winning_player.chips += self.pot
        self.pot = 0  # Reset the pot for the next round

    def evaluate_hand(self, cards):
        ranks = [card.value for card in cards]
        suits = [card.suit for card in cards]
        rank_counts = Counter(ranks).most_common()
        suit_counts = Counter(suits).most_common()

        # Sort cards by rank and use reverse order (highest first)
        sorted_cards = sorted(cards, key=lambda card: (card.value), reverse=True)

        is_flush = suit_counts[0][1] >= 5
        is_straight = self.is_straight(sorted_cards)

        # Flush and straight flush checks
        if is_flush:
            if is_straight:
                sorted_flush_cards = [card for card in sorted_cards if card.suit == suit_counts[0][0]]
                straight_flush = self.is_straight(sorted_flush_cards)
                if straight_flush:
                    return (9, straight_flush[0])  # Highest card in the straight flush
            return (6, sorted_cards[0].value)  # Highest card in flush

        # Four of a kind, full house, three of a kind, two pairs, one pair
        if rank_counts[0][1] == 4:
            return (8, rank_counts[0][0])  # Four of a kind
        elif rank_counts[0][1] == 3:
            if rank_counts[1][1] == 2:
                return (7, rank_counts[0][0])  # Full house
            return (4, rank_counts[0][0])  # Three of a kind
        elif rank_counts[0][1] == 2:
            if rank_counts[1][1] == 2:
                return (3, rank_counts[0][0])  # Two pairs
            return (2, rank_counts[0][0])  # One pair

        if is_straight:
            return (5, is_straight[0])  # Highest card in the straight

        # High card
        return (1, sorted_cards[0].value)

    def is_straight(self, sorted_cards):
        """Check for a straight in the sorted list of cards."""
        unique_cards = sorted(set([card.value for card in sorted_cards]), reverse=True)
        for i in range(len(unique_cards) - 4):
            if unique_cards[i] - unique_cards[i + 4] == 4:
                return (unique_cards[i],)  # Return highest value in the straight as a tuple
        # Special case for the low Ace
        if set([14, 5, 4, 3, 2]).issubset(unique_cards):
            return (5,)  # Highest card in the 5-high straight (5-4-3-2-A) as a tuple
        return None


# Usage
# players = [Player("Alice"), Player("Bob"), Player("Charlie")]
# game = TexasHoldem(players)
# game.start_round()

