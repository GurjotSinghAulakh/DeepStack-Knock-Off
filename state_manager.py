
class StateManager:

    def get_legal_actions(self, state):
        player = state["player"]
        player_bets = state["player_bets"]
        has_raised = player.has_raised
        max_bet = max(player_bets.values())

        can_call = (player_bets[player] < max_bet and player.pile >= max_bet - player_bets[player])
        can_raise = player.pile >= max_bet - player_bets[player] + 10 and not has_raised
        can_check = player_bets[player] == max_bet or all(bet == player_bets[player] for bet in player_bets.values()) or player.pile == 0
        can_fold = player.pile > 0

        legal_moves = []
        if can_call:
            legal_moves.append({"action": "call", "amount": max_bet - player_bets[player]})
        if can_raise:
            legal_moves.append({"action": "raise", "amount": max_bet - player_bets[player] + 10})
        if can_check:
            legal_moves.append({"action": "check", "amount": 0})
        if can_fold:
            legal_moves.append({"action": "fold", "amount": 0})
        return legal_moves
    
    def get_next_state(self, state, action):
        # state needs:
        # players, current_player, public_cards, player_bets, current_pile, stage, 
        # check if chance or player action
        if action["action"] == "deal_river":


            pass