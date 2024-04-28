from game_manager import GameManager
from player import Player


def main():
    player1 = Player(is_human=False, pile=100, name="Player 1")
    player2 = Player(is_human=True, pile=100, name="Player 2")
    # player3 = Player(is_human=False, pile=100, name="Player 3")
    game = GameManager(players=[player1, player2])
    game.play_game()


if __name__ == "__main__":
    main()
