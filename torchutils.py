import torch
import numpy as np
from poker_oracle import PokerOracle


def to_vec_in(r1: np.ndarray, r2: np.ndarray, public_cards: list[str], pot: int) -> torch.Tensor:
    r1_t = torch.Tensor(r1).reshape(1, -1)
    r2_t = torch.Tensor(r2).reshape(1, -1)
    deck = PokerOracle.generate_deck()
    public_cards_t = torch.Tensor([deck.index(card) for card in public_cards]).reshape(1, -1)
    pot_t = torch.Tensor([pot]).reshape(1, -1)

    return torch.cat([r1_t, r2_t, public_cards_t, pot_t], dim=1)
