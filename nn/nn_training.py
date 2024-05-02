from lightning import Trainer
from state.state_manager import PokerGameStage
from nn.nn_model import DeepstackNN
from utils.config import COMBINATIONS
from data.datagen import PokerDataModule


class NNTrainer:
    def __init__(self):
        self.training_order = [PokerGameStage.RIVER, PokerGameStage.TURN, PokerGameStage.FLOP]

    def train_network(self, stage: PokerGameStage, max_epochs: int = 100, data_size: int = 100) -> None:
        """
        Trains a neural network for a specific stage of the poker game

        Args:
        stage: PokerGameStage - the stage to train the network for
        max_epochs: int - the maximum number of epochs to train for
        data_size: int - the size of the dataset to use

        Returns:
        None
        """
        nbr_public_cards = 0
        if stage == PokerGameStage.FLOP:
            nbr_public_cards = 3
        elif stage == PokerGameStage.TURN:
            nbr_public_cards = 4
        elif stage == PokerGameStage.RIVER:
            nbr_public_cards = 5

        network = DeepstackNN(COMBINATIONS, nbr_public_cards)
        batch_size = 10
        data = PokerDataModule(stage, batch_size, data_size)
        trainer = Trainer(max_epochs=max_epochs, default_root_dir=f"lightning_logs/{stage.name}", devices=-1)

        trainer.fit(network, data)

    def train_all_networks(self, max_ephochs: int = 100, data_size: int = 100) -> None:
        """
        Trains all neural networks in the training order

        Args:
        max_epochs: int - the maximum number of epochs to train for
        data_size: int - the size of the dataset to use

        Returns:
        None
        """
        for stage in self.training_order:
            print(f"Training network: {stage.name}")
            self.train_network(stage, max_epochs=max_ephochs, data_size=data_size)
