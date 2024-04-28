from game_manager import Player, TexasHoldem, Card, Deck
import random

class AIPlayer(Player):
    def __init__(self, name):
        super().__init__(name)

    def decide_action(self, game):
        hand_strength = self.evaluate_hand(game.community_cards)
        # Basic decision logic based on hand strength
        if hand_strength > 5:
            bet = self.bet(50)  # Strong hand, more aggressive bet
        elif hand_strength > 2:
            bet = self.bet(20)  # Medium hand, cautious bet
        else:
            bet = self.bet(10)  # Weak hand, minimal bet
        
        game.pot += bet
        print(f"{self.name} bets {bet} with hand strength {hand_strength}")

    def evaluate_hand(self, community_cards):
        # Simplified hand strength evaluator (placeholder for real logic)
        score = 0
        all_cards = self.hand + community_cards
        rank_counts = {rank: 0 for rank in Card.RANKS}
        for card in all_cards:
            rank_counts[card.rank] += 1
        
        # Simple scoring: pairs, three of a kind, etc.
        for count in rank_counts.values():
            if count == 2:
                score += 1
            elif count == 3:
                score += 3
            elif count == 4:
                score += 6
        
        return score

# if __name__ == "__main__":
#     player1 = Player("Alice")
#     player2 = Player("Bob")
#     players = [player1, player2]
#     game = TexasHoldem(players)
#     game.add_player(AIPlayer("AI Alice"))
#     game.start_round()