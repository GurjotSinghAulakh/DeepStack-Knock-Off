from collections import Counter
import itertools
import pandas as pd
from itertools import combinations
import random
from config import BASE_CARDS, COMBINATIONS, SUITS
import numpy as np


PAIR_INDICES = [i for i in range(COMBINATIONS)]


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
    VALUE_SCORES = {v: i for i, v in enumerate(BASE_CARDS, start=2)}
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

    def evaluate_hole_pair_win_probability(self, hole_cards, public_cards=None, num_simulations=1000, cheet_sheet=False):
        public_cards = public_cards or []
        wins = 0

        for _ in range(num_simulations):
            deck = PokerOracle.generate_deck(cheat_sheet=cheet_sheet)
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

    def generate_utility_matrix(self, public_cards):
        utility_matrix = {}
        possible_hands = itertools.combinations(PokerOracle.generate_deck(), 2)

        for hand in possible_hands:
            utility_matrix[hand] = {}
            for opponent_hand in possible_hands:
                if not (set(hand) & set(opponent_hand)):  # Ensure hands are not overlapping
                    result = self.simulate_hand_vs_random_opponent(hand, opponent_hand, public_cards)
                    utility_matrix[hand][opponent_hand] = result

        return utility_matrix

    @staticmethod
    def compare_two_hands(hand1, hand2, public_cards, oracle):
        hand1_rank, hand1_score = oracle.classify_hand(hand1 + public_cards)
        hand2_rank, hand2_score = oracle.classify_hand(hand2 + public_cards)
        h1_hand_ranking = oracle.HAND_RANKINGS[hand1_rank]
        h2_hand_ranking = oracle.HAND_RANKINGS[hand2_rank]

        if h1_hand_ranking > h2_hand_ranking:
            return 1
        elif h1_hand_ranking < h2_hand_ranking:
            return -1

        if max(hand1_score) > max(hand2_score):
            return 1
        elif max(hand1_score) < max(hand2_score):
            return -1

        for i in range(len(hand1 + public_cards)):
            if hand1_score[i] > hand2_score[i]:
                return 1
            elif hand1_score[i] < hand2_score[i]:
                return -1
        return 0

    @staticmethod
    def calculate_utility_matrix(public_cards: list[str]):
        """
        Calculates the utility matrix for the given ranges

        a value of 1 at (i, j) means that hole card i wins over hole card j
        """
        oracle = PokerOracle()
        all_hole_cards = PokerOracle.all_hole_combinations()
        num_combinations = len(all_hole_cards)
        utility_matrix = np.zeros((num_combinations, num_combinations))

        for i in range(num_combinations):
            hand1 = PokerOracle.range_index_to_cards(all_hole_cards, i)
            if any(card in public_cards for card in hand1):
                continue  # Skip if any card in hand1 is in public cards

            for j in range(num_combinations):

                hand2 = PokerOracle.range_index_to_cards(all_hole_cards, j)
                if any(card in public_cards for card in hand2) or any(card in hand2 for card in hand1):
                    continue  # Skip if any card in hand2 is in public cards or hand1

                result = PokerOracle.compare_two_hands(hand1, hand2, public_cards, oracle)
                if result == 1:
                    utility_matrix[i, j] = 1
                    utility_matrix[j, i] = -1
                elif result == -1:
                    utility_matrix[j, i] = 1
                    utility_matrix[i, j] = -1
                elif result == 0:
                    utility_matrix[i, j] = 0
                    utility_matrix[j, i] = 0

        assert np.all(utility_matrix == -utility_matrix.T), "Utility matrix is not symmetric"
        assert np.sum(utility_matrix) == 0, "Utility matrix is not zero-sum"
        return utility_matrix

    @staticmethod
    def calculate_utility_matrix_bak(public_cards: list[str]):
        """
        Calculates the utility matrix for the given ranges

        a value of 1 at (i, j) means that hole card i wins over hole card j
        """
        oracle = PokerOracle()
        hand_strenghts = np.zeros(COMBINATIONS)
        for i in range(COMBINATIONS):
            all_hole_cards = PokerOracle.all_hole_combinations()
            current_hand = PokerOracle.range_index_to_cards(all_hole_cards, i)

            if any(card in public_cards for card in current_hand):
                continue
            # for card in current_hand:
            #     if card in public_cards:
            #         duplicate_cards = True

            # if duplicate_cards:
            #     continue

            hand_strenghts[i] = PokerOracle.evaluate_hand(current_hand + public_cards)

        m = np.sign(-np.subtract.outer(hand_strenghts, hand_strenghts))
        return m

    @staticmethod
    def generate_deck(cheat_sheet=False, randomize=False):
        if not cheat_sheet:
            r = [r + s for r in BASE_CARDS for s in SUITS]
        else:
            r = [r + s for r in BASE_CARDS for s in SUITS[0]]

        if randomize:
            random.shuffle(r)
        return r

    @staticmethod
    def all_hole_combinations(return_deck=False):
        deck = PokerOracle.generate_deck()
        all_permutations = combinations(deck, 2)
        if return_deck:
            return list(all_permutations), deck
        return list(all_permutations)

    @staticmethod
    def cards_to_range_index(all_hole_cards, card1, card2):
        return all_hole_cards.index((card1, card2))

    @staticmethod
    def range_index_to_cards(all_hole_cards, index):
        return all_hole_cards[index]


    def generate_cheat_sheet(self):
        cheat_sheet = {}
        deck = PokerOracle.generate_deck()
        possible_hands = itertools.combinations(deck, 2)
        stages = [3]
        print("3=276")
        for idx, possible_hand in enumerate(possible_hands):
            print(idx)
            cheat_sheet[possible_hand] = {}
            cheat_sheet[possible_hand]["winchances"] = {}
            cheat_sheet[possible_hand]["public_cards"] = {}
            for stage in stages:
                winchances = []
                for public_cards in itertools.combinations(deck, stage):
                    winchance = self.evaluate_hole_pair_win_probability(list(possible_hand), list(public_cards), num_simulations=10)
                    winchances.append(winchance)
                    cheat_sheet[possible_hand]["public_cards"][public_cards] = winchance

                cheat_sheet[possible_hand]["winchances"][stage] = (sum(winchances) / len(winchances))

        return cheat_sheet

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
        remaining_deck = [card for card in PokerOracle.generate_deck() if card not in hole_cards + public_cards]

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
    deck = PokerOracle.generate_deck()
    random.shuffle(deck)

    hole_cards = deck[:2]
    public_cards = deck[2:5]
    public_cards = []
    util = oracle.generate_cheat_sheet()
    print(util)
    df = pd.DataFrame(util)
    df.to_csv('cheat_sheet.csv', index=False)
    # print(PokerOracle.all_permuatations(hole_cards, public_cards, deck))
    # win_probability = oracle.evaluate_hole_pair_win_probability(hole_cards, public_cards, num_simulations=10000)
    # print(f"Win probability for {hole_cards} with {public_cards}: {win_probability}")
