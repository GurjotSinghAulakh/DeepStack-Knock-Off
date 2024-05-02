from deepstacklib.game.game_manager import GameManager
from deepstacklib.clients.player import Player
import matplotlib.pyplot as plt
from deepstacklib.utils.config import NUM_GAMES
from deepstacklib.game.poker_oracle import PokerOracle


def main():
    player1 = Player(name="Player 1", agent_type="resolve")
    player2 = Player(name="Player 2", agent_type="resolve")
    # player3 = Player(name="Player 3", agent_type="rollout")
    # player4 = Player(name="Player 4", agent_type="rollout")
    # player3 = Player(name="Player 3", agent_type="random")
    game = GameManager(players=[player1, player2, ])
    game.start_game(NUM_GAMES)

    # Plot the number of wins for each player
    names = list(game.wins.keys())
    wins = list(game.wins.values())
    plt.bar(names, wins)
    plt.xlabel('Player')
    plt.ylabel('Number of Wins')
    plt.title('Number of Wins for Each Player')
    plt.show()


def get_cheat_sheet():
    PokerOracle().get_cheat_sheet()


if __name__ == "__main__":
    main()
