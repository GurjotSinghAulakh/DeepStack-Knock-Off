from collections import Counter
import pandas as pd
from itertools import combinations
import random
from utils.config import BASE_CARDS, COMBINATIONS, SUITS
import numpy as np
import os


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

    def classify_hand(self, hand: list[str]) -> tuple[str, list[int]]:
        """
        Classify the given hand into a poker hand type.

        Args:
        hand: list[str] - List of cards in the hand

        Returns:
        tuple[str, list[int]] - Tuple containing the hand type and the sorted card ranks
        """

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

    def compare_hands(self, players, public_cards: list[str]) -> list:
        """
        Compare the hands of the given players and return the winner(s).

        Args:
        players: list[Player] - List of players to compare
        public_cards: list[str] - List of public cards on the table

        Returns:
        list[Player] - List of winning player(s)
        """
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

    def evaluate_hole_pair_win_probability(self, hole_cards: list[str], public_cards=None, num_simulations=1000, cheat_sheet=False) -> float:
        """
        Evaluate the win probability of the given hole cards via simulation.

        Args:
        hole_cards: list[str] - List of hole cards
        public_cards: list[str] - List of public cards on the table
        num_simulations: int - Number of simulations to run
        cheat_sheet: bool - Whether to use a cheat sheet for the simulation

        Returns:
        float - Win probability of the given hole cards and public cards
        """
        public_cards = public_cards or []
        wins = 0

        for _ in range(num_simulations):
            deck = PokerOracle.generate_deck(cheat_sheet=cheat_sheet, randomize=True)
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

    @staticmethod
    def compare_two_hands(hand1: list[str], hand2: list[str], public_cards: list[str], oracle) -> int:
        """
        Compare two hands and return 1 if hand1 wins, -1 if hand2 wins, and 0 if it's a tie.

        Args:
        hand1: list[str] - List of cards in hand 1
        hand2: list[str] - List of cards in hand 2
        public_cards: list[str] - List of public cards on the table
        oracle: PokerOracle - Instance of the PokerOracle class

        Returns:
        int - 1 if hand1 wins, -1 if hand2 wins, 0 if it's a tie
        """
        hand1 = tuple(hand1)
        hand2 = tuple(hand2)
        public_cards = tuple(public_cards)
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
    def calculate_utility_matrix(public_cards: list[str]) -> np.ndarray:
        """
        Generate a symmetric, zero-sum utility matrix for the given public cards. The matrix is of size (n, n) where
        n is the number of possible hole card combinations. A value of 1 at (i, j) means that hole card i wins over hole
        card j.

        Args:
        public_cards: list[str] - List of public cards on the table

        Returns:
        np.ndarray - Utility matrix for the given public cards
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
    def generate_deck(cheat_sheet=False, randomize=False) -> list[str]:
        """
        Generate a deck of cards.

        Args:
        cheat_sheet: bool - Whether to generate a cheat sheet deck
        randomize: bool - Whether to shuffle the deck

        Returns:
        list[str] - List of cards in the deck
        """
        if not cheat_sheet:
            r = [r + s for r in BASE_CARDS for s in SUITS]
        else:
            r = [r + s for r in BASE_CARDS for s in SUITS[0]]

        if randomize:
            random.shuffle(r)
        return r

    @staticmethod
    def all_hole_combinations(return_deck=False) -> list[tuple[str, str]]:
        """
        Get all possible hole card combinations.

        Args:
        return_deck: bool - Whether to return the deck along with the hole card combinations

        Returns:
        list[tuple[str, str]] - List of all possible hole card combinations
        """
        deck = PokerOracle.generate_deck()
        all_permutations = combinations(deck, 2)
        if return_deck:
            return list(all_permutations), deck
        return list(all_permutations)

    @staticmethod
    def cards_to_range_index(all_hole_cards: list[tuple[str, str]], card1: str, card2: str) -> int:
        """
        Given a pair of hole cards, return the index of the pair in the range.

        Args:
        all_hole_cards: list[tuple[str, str]] - List of all possible hole card combinations
        card1: str - First hole card
        card2: str - Second hole card

        Returns:
        int - Index of the hole card pair in the range
        """
        try:
            return all_hole_cards.index((card1, card2))
        except Exception:
            return all_hole_cards.index((card2, card1))

    @staticmethod
    def range_index_to_cards(all_hole_cards: list[tuple[str, str]], index: int) -> tuple[str, str]:
        """
        Given an index in the range, return the pair of hole cards.

        Args:
        all_hole_cards: list[tuple[str, str]] - List of all possible hole card combinations
        index: int - Index of the hole card pair in the range

        Returns:
        tuple[str, str] - Pair of hole cards
        """
        try:
            return all_hole_cards[index]
        except Exception:
            return all_hole_cards[index]

    def get_cheat_sheet(self, n_simulations: int = 1000, cheat_path: str = './data/cheating/cheat_sheet.csv') -> pd.DataFrame:
        """
        Get or generate a cheat sheet for the win probabilities of all possible hole card combinations.

        Args:
        n_simulations: int - Number of simulations to run for each hole card combination
        cheat_path: str - Path to the cheat sheet CSV file

        Returns:
        pd.DataFrame - Cheat sheet for the win probabilities of all possible hole card combinations
        """
        if os.path.exists(cheat_path):
            return pd.read_csv(cheat_path)

        data = []

        all_hole_combinations = PokerOracle.all_hole_combinations()
        for hole_pair in all_hole_combinations:
            win_probability = self.evaluate_hole_pair_win_probability(list(hole_pair), num_simulations=n_simulations)
            data.append({'hole_card1': hole_pair[0], 'hole_card2': hole_pair[1], 'win_probability': win_probability})

        df = pd.DataFrame(data)
        df.to_csv(cheat_path, index=False)
        return df

    def get_cheat_sheet_probs(self, hole_pair: tuple[str, str]) -> float:
        """
        Get the win probability of the given hole card pair from the cheat sheet.

        Args:
        hole_pair: tuple[str, str] - Pair of hole cards

        Returns:
        float - Win probability of the given hole card pair
        """
        df = self.get_cheat_sheet()
        try:
            return df[(df['hole_card1'] == hole_pair[0]) & (df['hole_card2'] == hole_pair[1])]['win_probability'].values[0]
        except IndexError:
            return df[(df['hole_card1'] == hole_pair[1]) & (df['hole_card2'] == hole_pair[0])]['win_probability'].values[0]


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
