# In main.py
from game_manager import Player, TexasHoldem
from game_state import GameState
from nn import create_model, train_model, prepare_training_data, collect_data
from nn import game_state_to_features
from mcts import Node, monte_carlo_tree_search

def main():
    data = collect_data(100000)
    features, labels = prepare_training_data(data)
    
    input_dim = len(features[0])
    model = create_model(input_dim)
    train_model(model, features, labels)

    players = [Player("Alice", 1000), Player("Bob", 1000)]
    
    game = TexasHoldem(players)
    game.start_round()

    initial_state = GameState(players)
    root = Node(initial_state)
    monte_carlo_tree_search(root, 500)

    while not game.round_over():
        state_features = game_state_to_features(game.export_state())
        prediction = model.predict([state_features])[0]
        game.play_turn_based_on_prediction(prediction)
        game.play_turn()

if __name__ == "__main__":
    main()


