from collections import Counter
from player import Player
import itertools
import random


class PokerHandType:
    HIGH_CARD = 1
    ONE_PAIR = 2
    TWO_PAIR = 3
    THREE_OF_A_KIND = 4
    STRAIGHT = 5
    FLUSH = 6
    FULL_HOUSE = 7
    FOUR_OF_A_KIND = 8
    STRAIGHT_FLUSH = 9
    ROYAL_FLUSH = 10


class PokerOracle:
    VALUES = '23456789TJQKA'
    VALUE_SCORES = {v: i for i, v in enumerate(VALUES, start=2)}
    SUITS = '♠♥♦♣'
    HAND_RANKINGS = {
        'High Card': PokerHandType.HIGH_CARD,
        'One Pair': PokerHandType.ONE_PAIR,
        'Two Pair': PokerHandType.TWO_PAIR,
        'Three of a Kind': PokerHandType.THREE_OF_A_KIND,
        'Straight': PokerHandType.STRAIGHT,
        'Flush': PokerHandType.FLUSH,
        'Full House': PokerHandType.FULL_HOUSE,
        'Four of a Kind': PokerHandType.FOUR_OF_A_KIND,
        'Straight Flush': PokerHandType.STRAIGHT_FLUSH,
        'Royal Flush': PokerHandType.ROYAL_FLUSH
    }

    def classify_hand(self, hand):
        hand = sorted(hand, key=lambda card: self.VALUE_SCORES[card[0]], reverse=True)
        values = [card[0] for card in hand]
        suits = [card[1] for card in hand]

        value_counts = Counter(values)
        most_common_values = value_counts.most_common()

        is_flush = len(set(suits)) == 1
        is_straight = len(set(self.VALUE_SCORES[value] for value in values)) == 5 and self.VALUE_SCORES[values[0]] - self.VALUE_SCORES[values[-1]] == 4

        sorted_ranks = sorted((self.VALUE_SCORES[value] for value in values), reverse=True)

        # Check for Royal Flush
        if is_flush and is_straight and values[0] == 'A':
            return 'Royal Flush', sorted_ranks
        # Check for Straight Flush
        if is_flush and is_straight:
            return 'Straight Flush', sorted_ranks
        # Check for Four of a Kind
        if most_common_values[0][1] == 4:
            return 'Four of a Kind', sorted_ranks
        # Check for Full House
        if most_common_values[0][1] == 3 and most_common_values[1][1] == 2:
            return 'Full House', sorted_ranks
        # Check for Flush
        if is_flush:
            return 'Flush', sorted_ranks
        # Check for Straight
        if is_straight:
            return 'Straight', sorted_ranks
        # Check for Three of a Kind
        if most_common_values[0][1] == 3:
            return 'Three of a Kind', sorted_ranks
        # Check for Two Pair
        if most_common_values[0][1] == 2 and most_common_values[1][1] == 2:
            return 'Two Pair', sorted_ranks
        # Check for One Pair
        if most_common_values[0][1] == 2:
            return 'One Pair', sorted_ranks
        # If none of the above, it's a High Card
        return 'High Card', sorted_ranks

    def compare_hands(self, players, public_cards):
        player_rank = {player: self.classify_hand(player.hole_cards + public_cards) for player in players}
        player_hand_ranks = {player: self.HAND_RANKINGS[rank] for player, (rank, _) in player_rank.items()}
        print(f"\nplayer hand strengths: {player_rank}")
        # print(f"\nplayer hand ranks: {player_hand_ranks}")

        if len(players) == 1:
            return players

        highest_score = max(player_hand_ranks.values())
        highest_ranks = [player for player, rank in player_hand_ranks.items() if rank == highest_score]
        if highest_ranks == 1:
            return highest_ranks

        tied_player_cards = {
            player: sorted(cards, reverse=True) for player, (_, cards) in player_rank.items() if player in highest_ranks
        }

        # Find the best set of cards among the tied players
        winning_players = [highest_ranks[0]]  # Start by assuming the first player is the best
        for player in highest_ranks[1:]:
            for card1, card2 in zip(tied_player_cards[winning_players[0]], tied_player_cards[player]):
                if card2 > card1:
                    winning_players = [player]  # New winner found
                    break
                elif card2 < card1:
                    break
            else:
                # If all cards are the same, it's a tie
                if tied_player_cards[player] == tied_player_cards[winning_players[0]]:
                    winning_players.append(player)

        return winning_players

    def generate_utility_matrix(self, public_cards):
        utility_matrix = {}
        possible_hands = itertools.combinations(self.generate_deck(), 2)

        for hand in possible_hands:
            utility_matrix[hand] = {}
            for opponent_hand in possible_hands:
                if not (set(hand) & set(opponent_hand)):  # Ensure hands are not overlapping
                    result = self.simulate_hand_vs_random_opponent(hand, opponent_hand, public_cards)
                    utility_matrix[hand][opponent_hand] = result

        return utility_matrix

    def evaluate_hole_pair_win_probability(self, hole_cards, public_cards=None, num_simulations=1000):
        public_cards = public_cards or []
        wins = 0

        for _ in range(num_simulations):
            deck = self.generate_deck()
            random.shuffle(deck)
            known_cards = set(hole_cards + public_cards)
            deck = [card for card in deck if card not in known_cards]

            opponent_hand = deck[:2]
            remaining_community = deck[2:2 + (5 - len(public_cards))]
            final_community = public_cards + remaining_community

            player_classification = self.classify_hand(hole_cards + final_community)
            opponent_classification = self.classify_hand(opponent_hand + final_community)

            player_hand_rank, player_ranks = player_classification
            opponent_hand_rank, opponent_ranks = opponent_classification

            if self.HAND_RANKINGS[player_hand_rank] > self.HAND_RANKINGS[opponent_hand_rank]:
                wins += 1
            elif self.HAND_RANKINGS[player_hand_rank] == self.HAND_RANKINGS[opponent_hand_rank]:
                if player_ranks > opponent_ranks:
                    wins += 1
                elif player_ranks == opponent_ranks:
                    wins += 0.5  # Considering a split pot as a half-win

        return wins / num_simulations

    def generate_deck(self):
        return [r + s for r in self.VALUES for s in self.SUITS]

    # The following methods are helpers for the above functionalities.
    def hand_strength(self, hand):
        """
        Assign a score to the hand so it can be compared with others.
        The highest bits are reserved for the hand type, followed by the card ranks.
        """
        rank = self.classify_hand(hand)
        score = self.HAND_RANKINGS[rank] << 20  # Reserve highest bits for hand type

        sorted_values = sorted((self.VALUE_SCORES[card[0]] for card in hand), reverse=True)
        for i, value in enumerate(sorted_values):
            score += (value << (4 * (5 - i)))  # Shift bits for each card value

        return score

    def simulate_hand_vs_random_opponent(self, hole_cards, public_cards, simulations=1000):
        wins = 0

        # Generate a list of the remaining cards
        remaining_deck = [card for card in self.generate_deck() if card not in hole_cards + public_cards]

        for _ in range(simulations):
            # Shuffle the remaining cards and deal them out
            random.shuffle(remaining_deck)
            opponent_hole = remaining_deck[:2]
            remaining_flop = remaining_deck[2:5] if len(public_cards) == 0 else []
            turn_card = remaining_deck[5] if len(public_cards) <= 3 else []
            river_card = remaining_deck[6] if len(public_cards) <= 4 else []

            full_community = public_cards + remaining_flop + [turn_card] + [river_card]
            full_community = [card for card in full_community if card]  # Filter out empty draws

            # Evaluate hands at the end of the simulation
            player_score = self.hand_strength(hole_cards + full_community)
            opponent_score = self.hand_strength(opponent_hole + full_community)

            if player_score > opponent_score:
                wins += 1

        # Return the win ratio
        return wins / simulations


if __name__ == "__main__":
    oracle = PokerOracle()
    deck = oracle.generate_deck()
    random.shuffle(deck)

    hole_cards = deck[:2]
    public_cards = deck[2:5]
    win_probability = oracle.evaluate_hole_pair_win_probability(hole_cards, public_cards)
    print(f"Win probability for {hole_cards} with {public_cards}: {win_probability}")
