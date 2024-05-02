from state.state_manager import PokerGameStage
from nn.nn_model import DeepstackNN
from glob import glob
import os
from utils.config import COMBINATIONS
import torch


class NNManager:
    def __init__(self):
        self.river_network = self.load_network(PokerGameStage.RIVER)
        self.turn_network = self.load_network(PokerGameStage.TURN)
        self.flop_network = self.load_network(PokerGameStage.FLOP)

    def get_network(self, stage: PokerGameStage) -> DeepstackNN:
        """
        Get the network for a specific stage

        Args:
        stage: PokerGameStage - the stage to get the network for

        Returns:
        DeepstackNN - the network for the stage
        """
        if stage == PokerGameStage.FLOP:
            network = self.flop_network
        elif stage == PokerGameStage.TURN:
            network = self.turn_network
        elif stage == PokerGameStage.RIVER:
            network = self.river_network

        return network

    def load_network(self, stage: PokerGameStage, version: int = -1) -> DeepstackNN:
        """
        Load a network for a specific stage

        Args:
        stage: PokerGameStage - the stage to load the network for
        version: int - the version of the network to load

        Returns:
        DeepstackNN - the loaded network
        """
        n_pub = 0
        if stage == PokerGameStage.FLOP:
            n_pub = 3
        elif stage == PokerGameStage.TURN:
            n_pub = 4
        elif stage == PokerGameStage.RIVER:
            n_pub = 5

        network = DeepstackNN(COMBINATIONS, n_pub)

        try:
            stage_dir = f"./nn/lightning_logs/{stage.name}/lightning_logs/"
            folders = glob(stage_dir + "version_*")
            if len(folders) == 0:
                raise Exception("No folders for network")

            sorted_dirs = sorted(folders, key=os.path.getmtime)
            dir = sorted_dirs[version]

            f = glob(f"{dir}/checkpoints/*.ckpt")[0]
            network = DeepstackNN.load_from_checkpoint(f, range_size=COMBINATIONS, public_info_size=n_pub)
        except Exception as e:
            print(f"Failed to load network for stage {stage.name}, version {version}, Error: {e}")
            pass

        device = "mps" if torch.cuda.is_available() else "cpu"
        network.to(device)
        return network
