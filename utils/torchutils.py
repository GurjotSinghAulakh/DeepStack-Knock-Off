import torch
import numpy as np
from game.poker_oracle import PokerOracle


def to_vec_in(r1: np.ndarray, r2: np.ndarray, public_cards: list[str], pot: int) -> torch.Tensor:
    """
    public info to input tensor

    args:
    r1: np.ndarray - range vector for player 1
    r2: np.ndarray - range vector for player 2
    public_cards: list[str] - list of public cards
    pot: int - current pot size

    returns:
    torch.Tensor - input tensor for the neural network
    """
    r1_t = torch.Tensor(r1).reshape(1, -1)
    r2_t = torch.Tensor(r2).reshape(1, -1)
    deck = PokerOracle.generate_deck()
    public_cards_t = torch.Tensor([deck.index(card) for card in public_cards]).reshape(1, -1)
    pot_t = torch.Tensor([pot]).reshape(1, -1)

    return torch.cat([r1_t, r2_t, public_cards_t, pot_t], dim=1)


def from_vec_out(v1: torch.Tensor, v2: torch.Tensor, dot_sum: torch.Tensor = None) -> torch.Tensor:
    """
    cat output tensors to single tensor

    args:
    v1: torch.Tensor - output tensor for player 1
    v2: torch.Tensor - output tensor for player 2
    dot_sum: torch.Tensor - sum of dot products (LEGACY)

    returns:
    torch.Tensor - concatenated output tensor
    """
    # return torch.cat([v1, v2, dot_sum], dim=1)
    return torch.cat([v1, v2], dim=1)
