import itertools
import random

from game_manager import Deck

class PokerOracle:
    @staticmethod
    def rank_hand(cards):
        """
        Ranks a hand of cards using standard poker rules.
        Returns a tuple representing the type of hand and the high cards for tie-breaking.
        """
        if not cards:
            return None

        values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                  '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14 }
        suits = [card.suit for card in cards]
        ranks = sorted([values[card.rank] for card in cards], reverse=True)
        rank_counts = {rank: ranks.count(rank) for rank in ranks}
        rank_sorted = sorted(rank_counts.items(), key=lambda x: (-x[1], -x[0]))

        is_flush = len(set(suits)) == 1
        is_straight = len(set(ranks)) == 5 and ranks[0] - ranks[4] == 4

        if ranks == [14, 5, 4, 3, 2]:  # Special case: 5-high straight (A-2-3-4-5)
            is_straight = True
            ranks = [5, 4, 3, 2, 1]

        # High Card: (1, [High, ...])
        rank_value = (1, ranks)
        if is_straight and is_flush:
            return (9, [ranks[0]])  # Straight flush
        elif rank_sorted[0][1] == 4:
            return (8, [rank_sorted[0][0], rank_sorted[1][0]])  # Four of a kind
        elif rank_sorted[0][1] == 3 and rank_sorted[1][1] == 2:
            return (7, [rank_sorted[0][0], rank_sorted[1][0]])  # Full house
        elif is_flush:
            return (6, ranks)  # Flush
        elif is_straight:
            return (5, [ranks[0]])  # Straight
        elif rank_sorted[0][1] == 3:
            return (4, [rank_sorted[0][0]] + [r for r, _ in rank_sorted if r != rank_sorted[0][0]])  # Three of a kind
        elif rank_sorted[0][1] == 2 and rank_sorted[1][1] == 2:
            return (3, [rank_sorted[0][0], rank_sorted[1][0], rank_sorted[2][0]])  # Two pair
        elif rank_sorted[0][1] == 2:
            return (2, [rank_sorted[0][0]] + [r for r, _ in rank_sorted if r != rank_sorted[0][0]])  # One pair
        return rank_value

    @staticmethod
    def calculate_probabilities(player_cards, community_cards, remaining_deck):
        """
        Calculate the probabilities of winning for the player's hand against hypothetical opponents.
        """
        trials = 1000
        wins = 0
        remaining_deck = [card for card in remaining_deck if card not in player_cards and card not in community_cards]

        for _ in range(trials):
            random.shuffle(remaining_deck)
            opponent_cards = remaining_deck[:2]
            remaining_community = remaining_deck[2:5] if len(community_cards) < 5 else []
            final_community = community_cards + remaining_community

            player_hand = player_cards + final_community
            opponent_hand = opponent_cards + final_community

            player_rank = PokerOracle.rank_hand(player_hand)
            opponent_rank = PokerOracle.rank_hand(opponent_hand)

            if player_rank > opponent_rank:
                wins += 1

        return wins / trials

# Example usage:
# deck = Deck()
# player_cards = [deck.deal() for _ in range(2)]
# print(player_cards)
# community_cards = [deck.deal() for _ in range(3)]
# print(community_cards)
# oracle = PokerOracle()
# print("Hand rank:", oracle.rank_hand(player_cards + community_cards))
# print("Winning probability:", oracle.calculate_probabilities(player_cards, community_cards, deck.cards))
