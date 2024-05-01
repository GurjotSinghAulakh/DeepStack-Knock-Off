# from game_manager import GameManager
from gm import GameManager
from player import Player


def main():
    player1 = Player(is_human=False, pile=1000, name="Player 1", aggresiveness=1)
    # player2 = Player(is_human=False, pile=10000, name="Player 2", aggresiveness=0.1)
    player2 = Player(is_human=False, pile=1000, name="Player 2", eval_hole_probs=True)
    # player3 = Player(is_human=False, pile=100, name="Player 3")
    game = GameManager(players=[player1, player2])
    game.start_game(num_games=150)


if __name__ == "__main__":
    main()
