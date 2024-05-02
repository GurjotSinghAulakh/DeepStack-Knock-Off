from game.game_manager import GameManager
from clients.player import Player


def main():
    player1 = Player(name="Player 1", agent_type="resolve")
    player2 = Player(name="Player 2", agent_type="rollout")
    # player3 = Player(is_human=False, name="Player 3", agent_type="random")
    game = GameManager(players=[player1, player2])
    game.start_game(num_games=150)


if __name__ == "__main__":
    main()
