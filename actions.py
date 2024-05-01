from enum import Enum


class Action:
    def __init__(self, action, amount=0) -> None:
        self.action_type = action
        self.amount = amount

    def __eq__(self, action2) -> bool:
        return self.action_type == action2.action_type and self.amount == action2.amount


class ActionType(Enum):
    FOLD = 0
    CALL = 1
    CHECK = 2
    RAISE = 3


ACTIONS = [
    Action(ActionType.FOLD),
    Action(ActionType.CALL),
    Action(ActionType.CHECK),
    Action(ActionType.RAISE, 10),
]


def agent_action_index(action: Action) -> int:
    return action.action_type.value
