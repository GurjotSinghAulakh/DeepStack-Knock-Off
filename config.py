import math

MAX_STAGE_RAISES = 1
BASE_CARDS = "9TJQKA"
SUITS = "♠♥♦♣"
NUM_CARDS = len(BASE_CARDS) * len(SUITS)
COMBINATIONS = math.comb(NUM_CARDS, 2)
SUITED_CARDS = [card + suit for card in BASE_CARDS for suit in SUITS]
