import random
from game_manager import Deck


class GameState:
    def __init__(self, players, dealer_index=0):
        self.players = players
        self.dealer_index = dealer_index
        self.community_cards = []
        self.current_bets = {player.name: 0 for player in players}
        self.pot = 0
        self.current_player_index = (dealer_index + 1) % len(players)
        self.deck = Deck()
        self.deck.shuffle()

    def get_legal_actions(self, player):
        actions = ['fold']
        player_bet = self.current_bets[player.name]
        max_bet = max(self.current_bets.values())

        if player_bet < max_bet:
            if player.chips >= (max_bet - player_bet):
                actions.append('call')
            if player.chips > (max_bet - player_bet):
                actions.append('raise')
        else:
            actions.append('check')

        return actions

    def do_action(self, action, player, raise_amount=0):
        if action == 'fold':
            player.in_play = False
        elif action == 'call':
            bet = max(self.current_bets.values()) - self.current_bets[player.name]
            self.pot += player.bet(bet)
            self.current_bets[player.name] += bet
        elif action == 'check':
            pass  # nothing happens
        elif action == 'raise':
            bet = max(self.current_bets.values()) - self.current_bets[player.name] + raise_amount
            self.pot += player.bet(bet)
            self.current_bets[player.name] += bet
        self.current_player_index = (self.current_player_index + 1) % len(self.players)

    def is_winner(self, player):
        # Placeholder for winner determination logic
        return True

    def get_result(self, player):
        # This function should be a proper implementation to determine the game result
        return 1 if self.is_winner(player) else 0

    def simulate(self):
        # A simulation step where we randomly choose actions for each player
        while any(p.in_play for p in self.players) and len(self.community_cards) < 5:
            player = self.players[self.current_player_index]
            legal_actions = self.get_legal_actions(player)
            action = random.choice(legal_actions)
            self.do_action(action, player)
