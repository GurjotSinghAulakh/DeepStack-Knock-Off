import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from game_manager import TexasHoldem, Player

def collect_data(num_games):
    data = []
    for _ in range(num_games):
        game = TexasHoldem([Player("Alice"), Player("Bob")])
        game.start_round()
        while not game.round_over():
            game.play_turn()
        # Collect game state and result here
        data.append((game.export_state(), game.determine_winner()))
    return data

def game_state_to_features(game_state):
    # Transform game state into a flat list of features
    features = []
    # Feature extraction
    features.extend([card.value for card in game_state.community_cards])
    for player in game_state.players:
        features.append(player.chips)
        features.extend([card.value for card in player.hand])
    return features

def create_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, game_data, game_labels, epochs=10):
    model.fit(game_data, game_labels, epochs=epochs, batch_size=32)

def prepare_training_data(data):
    features = [game_state_to_features(state) for state, _ in data]
    labels = [1 if winner == "Alice" else 0 for _, winner in data]  # binary classification for simplicity
    return features, labels
