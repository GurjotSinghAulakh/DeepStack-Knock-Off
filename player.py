import numpy as np
from dataclasses import dataclass, field


@dataclass
class Player:
    is_human: bool
    pile: int
    name: str
    hole_cards: list[str] = field(default_factory=list)
    active: bool = True

    def choose_action(self, actions):
        actions_list = [f"{i + 1}. {action['action']} {action['amount']}" for i, action in enumerate(actions)]
        print(f"\nLegal actions for {self.name}: {actions_list}, pile: {self.pile}")
        if self.is_human:
            chosen_idx = int(input("Choose an action: ")) - 1
        else:
            #remove fold
            actions = [action for action in actions if action['action'] != "fold"]
            chosen_idx = np.random.choice(len(actions))
        print(f"{self.name} chose {actions[chosen_idx]}")
        return actions[chosen_idx]

    def __hash__(self) -> int:
        return hash(self.name)
    
    def __repr__(self) -> str:
        return "Human - " + self.name if self.is_human else "AI - " + self.name