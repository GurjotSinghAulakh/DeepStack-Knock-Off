from config import MAX_STAGE_RAISES

class StateManager:

    @staticmethod
    def get_legal_actions(state):
        player = state["player"]
        player_bets = state["player_bets"]
        stage_raises = state["stage_raises"]
        max_bet = max(player_bets.values())

        can_call = (player_bets[player] < max_bet and player.pile >= max_bet - player_bets[player])
        can_raise = player.pile >= max_bet - player_bets[player] + 10 and stage_raises < MAX_STAGE_RAISES
        can_check = player_bets[player] == max_bet or all(bet == player_bets[player] for bet in player_bets.values()) or player.pile == 0
        can_fold = player.pile > 0

        legal_moves = []
        if can_call:
            legal_moves.append({"action": "call", "amount": max_bet - player_bets[player]})
        if can_raise:
            legal_moves.append({"action": "raise", "amount": max_bet - player_bets[player] + 10})
        if can_check:
            legal_moves.append({"action": "check", "amount": 0})
        
        # if not legal_moves:
        #     legal_moves.append({"action": "", "amount": 0})

        if can_fold:
            legal_moves = [{"action": "fold", "amount": 0}]
        return legal_moves
