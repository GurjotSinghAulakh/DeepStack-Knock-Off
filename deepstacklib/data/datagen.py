from deepstacklib.state.subtree_manager import SubtreeManager
from deepstacklib.utils.config import COMBINATIONS, AVG_POT_SIZE, PLAYER_CHIPS
from deepstacklib.state.state_manager import GameState, PokerGameStage
from deepstacklib.game.poker_oracle import PokerOracle
from deepstacklib.clients.actions import ACTIONS
from deepstacklib.utils.torchutils import to_vec_in, from_vec_out
import pytorch_lightning as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import multiprocessing as mp
import os
import pickle as pkl


def generate_random_ranges(public_cards: list[str], range_size: int = COMBINATIONS) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates two random ranges for the players

    args:
    public_cards: list[str] - list of public cards
    range_size: int - size of the range vectors

    returns:
    tuple[np.ndarray, np.ndarray] - range vectors for player 1 and player 2
    """
    r1 = np.random.random(range_size)
    r1 = r1 / np.sum(r1)
    r2 = np.random.random(range_size)
    r2 = r2 / np.sum(r2)

    r1 = SubtreeManager.update_range_from_public_cards(r1, public_cards)
    r2 = SubtreeManager.update_range_from_public_cards(r2, public_cards)

    return (r1, r2)


def generate_initial_situation_from_public_cards(public_cards: list[str]) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Generate a random initial situation from the given public cards

    args:
    public_cards: list[str] - list of public cards

    returns:
    tuple[np.ndarray, np.ndarray, int] - range vectors for player 1 and player 2, pot size
    """
    r1, r2 = generate_random_ranges(public_cards)
    pot = np.random.randint(0, AVG_POT_SIZE * 2)

    return (r1, r2, pot)


def get_calculated_values_for_situation(stage: PokerGameStage, r1: np.ndarray, r2: np.ndarray, pot: int, public_cards: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the estimated values for a given situation by emulating the game tree and exploring the subtrees

    args:
    stage: PokerGameStage - current stage of the game
    r1: np.ndarray - range vector for player 1
    r2: np.ndarray - range vector for player 2
    pot: int - current pot size
    public_cards: list[str] - list of public cards

    returns:
    tuple[np.ndarray, np.ndarray] - estimated values for player 1 and player 2
    """
    deck = PokerOracle.generate_deck(randomize=True)
    for card in public_cards:
        deck.remove(card)

    game_state = GameState(
        stage=stage,
        current_player_index=0,
        player_bets=np.array([pot / 2, pot / 2]),
        player_chips=np.ones(2) * PLAYER_CHIPS,
        player_checks=np.zeros(2),
        player_raised=np.zeros(2),
        players_in_game=np.ones(2),
        players_all_in=np.zeros(2, dtype=bool),
        pot=pot,
        bet_to_match=pot // 2,
        public_info=public_cards,
        deck=deck,
    )

    end_stage = stage
    end_depth = 1

    if stage == PokerGameStage.PRE_FLOP:
        end_stage = PokerGameStage.FLOP
    elif stage == PokerGameStage.FLOP:
        end_stage = PokerGameStage.TURN
    elif stage == PokerGameStage.TURN:
        end_stage = PokerGameStage.RIVER
    elif stage == PokerGameStage.RIVER:
        end_depth = 20

    strategy = np.ones((r1.size, len(ACTIONS))) / r1.size

    tree = SubtreeManager(game_state, end_stage, end_depth, strategy)

    v1, v2 = tree.subtree_traversal_rollout(tree.root, r1, r2)

    return (v1, v2)


def get_random_example(arg: tuple[PokerGameStage, int]) -> torch.Tensor:
    """
    Generate a random example for the given stage and number of public cards, returns a tensor of the example

    args:
    arg: tuple[PokerGameStage, int] - stage and number of public cards

    returns:
    torch.Tensor - tensor of the example data
    """
    stage, nbr_public_cards = arg

    deck = PokerOracle.generate_deck(randomize=True)
    public_cards = deck[:nbr_public_cards]
    deck = deck[nbr_public_cards:]

    r1, r2, pot = generate_initial_situation_from_public_cards(public_cards)

    v1, v2 = get_calculated_values_for_situation(stage, r1, r2, pot, public_cards)

    output = from_vec_out(
        torch.tensor(v1).reshape(1, -1),
        torch.tensor(v2).reshape(1, -1),
        torch.zeros(1).reshape(1, -1),
    )

    example = torch.cat((to_vec_in(r1, r2, public_cards, pot), output), dim=1)

    return example


def generate_data(start: int, end: int, stage: PokerGameStage, n_pub: int) -> list[torch.Tensor]:
    """
    Generate data for the given range, this function is used for multiprocessing

    args:
    start: int - start index
    end: int - end index
    stage: PokerGameStage - current stage of the game
    n_pub: int - number of public cards

    returns:
    list[torch.Tensor] - list of generated examples for this subset
    """
    local_data = []
    counter = 0
    for _ in range(start, end):
        if counter % 10 == 0:
            print(counter)
        counter += 1
        local_data.append(get_random_example((stage, n_pub)))
    return local_data


def get_data_mp(stage: PokerGameStage, n_pub: int, total_items: int, n_processes: int) -> list[torch.Tensor]:
    """
    Get data for the given stage and number of public cards using multiprocessing

    args:
    stage: PokerGameStage - stage of the game to generate data for
    n_pub: int - number of public cards
    total_items: int - total number of examples to generate
    n_processes: int - number of processes to use

    returns:
    list[torch.Tensor] - list of generated examples
    """

    pool = mp.Pool(processes=n_processes)

    # Calculate the chunk size for each process
    chunk_size = total_items // n_processes
    ranges = [(i * chunk_size, (i + 1) * chunk_size) for i in range(n_processes)]
    ranges[-1] = (ranges[-1][0], total_items)  # Ensure the last range covers the remainder

    # Map the generate_data function to the data ranges
    results = [pool.apply_async(generate_data, args=(r[0], r[1], stage, n_pub)) for r in ranges]

    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()

    # Collect all results from the processes
    data = []
    for result in results:
        data.extend(result.get())

    return data


class PokerDataModule(pl.LightningDataModule):
    def __init__(self, stage: PokerGameStage, batch_size: int, data_size: int = 1000):
        super().__init__()
        self.stage = stage
        self.data_dir = f"deepstacklib/data/{stage.name}"
        self.batch_size = batch_size
        self.data_size = data_size
        self.workers = 8

    def setup(self, stage: str = None) -> None:
        """
        DataModule setup. This function is called once before the training starts internally by PyTorch Lightning
        """

        if self.stage == PokerGameStage.RIVER:
            offset = 5
            mean = 62
            std = 32
        elif self.stage == PokerGameStage.TURN:
            offset = 4
            mean = 47
            std = 25
        elif self.stage == PokerGameStage.FLOP:
            offset = 3
            mean = 32
            std = 17

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        if not os.path.exists(f"deepstacklib/{self.data_dir}/data_tensor.pt"):
            print(f"Generating {self.data_size} data points for {self.stage.name} with 8 cores. This may take a while...")
            data = get_data_mp(stage=self.stage, n_pub=offset, total_items=self.data_size, n_processes=8)
            tensordata = torch.cat(data)
            torch.save(tensordata, f"deepstacklib/{self.data_dir}/data_tensor.pt")
            print(f"Saved data as deepstacklib/{self.data_dir}/data_tensor.pt")

        all = torch.load(f"deepstacklib/{self.data_dir}/data_tensor.pt")
        all = all.type(torch.float32)

        pot_index = 276 * 2 + offset
        all[:, pot_index] = self.normalize(all[:, pot_index], std, mean)

        val_fraction = 0.2

        val_size = int(val_fraction * len(all))
        train_size = len(all) - val_size

        self.train_data, self.val_data = random_split(all, [train_size, val_size])

    def normalize(self, data, std, mean):
        return (data - mean) / std

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.workers, persistent_workers=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.workers, persistent_workers=True, shuffle=True)


if __name__ == "__main__":
    n_pub = 5
    total_items = 3000
    n_p = 8
    stage = PokerGameStage.RIVER
    data = get_data_mp(stage=stage, n_pub=n_pub, total_items=total_items, n_processes=n_p)
    random_hash = np.random.randint(0, 999)
    pkl.dump(data, open(f"datastage1{random_hash}.pkl", "wb"))
    # data = pkl.load(open("datastage.pkl", "rb"))
    tensordata = torch.cat(data)
    test_frac = 0.2
    test_num = int(test_frac * total_items)
    train_dataset, test_dataset = tensordata.split([total_items - test_num, test_num])
    torch.save(train_dataset, f"./deepstacklib/data/{stage.name}/data_tensor{random_hash}.pt")
    print(f"Saved data as data_tensor{random_hash}.pt")
