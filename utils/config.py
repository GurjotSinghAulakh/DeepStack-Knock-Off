import math


MAX_STAGE_RAISES = 1    # Maximum number of raises per stage, currently not enforced, state manager keeps track of this now
BASE_CARDS = "9TJQKA"   # Cards used in the game, 9, 10, Jack, Queen, King, Ace. Lower cards are not used for simplicity
SUITS = "♠♥♦♣"          # Suits used in the game, spades, hearts, diamonds, clubs
NUM_CARDS = len(BASE_CARDS) * len(SUITS)      # Total number of cards in the game, 6
COMBINATIONS = math.comb(NUM_CARDS, 2)  # 276 - Number of possible combinations of 2 cards
SUITED_CARDS = [card + suit for card in BASE_CARDS for suit in SUITS]  # All possible suited cards
AVG_POT_SIZE = 20                    # Average pot size, used for generating random pots and normalization
PLAYER_CHIPS = 1000                  # Starting chips for each player
RESOLVER_ROLLOUTS = 100              # Number of rollouts to perform in the resolver
RESOLVER_CHILDREN_ACTION_LIMIT = 10  # Number of children to consider in the resolver
RESOLVER_N_CHILD_STATES = 5          # Number of child states to consider in the resolver
