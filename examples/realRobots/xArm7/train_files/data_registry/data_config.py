"""xArm7 real-robot teleoperation — data config and mixtures.

Adapted from examples/simBenchmarks/Robotwin/train_files/data_registry/data_config.py (AgilexData50Config).
The dataset is the LeRobot v2.1 conversion of the xArm7 episodes (cameras left / wrist, 8-D state and
action = 7 joint angles + gripper); its meta/modality.json is the one next to this directory.
"""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


# ---------------------------------------------------------------------------
# DataConfig — xArm7 (action_indices=16)
# ---------------------------------------------------------------------------
class XArm7DataConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = ["video.left", "video.wrist"]
    state_keys = ["state.joints", "state.gripper"]
    action_keys = ["action.joints", "action.gripper"]
    # Per-key dims for PolicyNormProcessor (7-DOF arm + binary gripper = 8-D total)
    action_key_dims = {"action.joints": 7, "action.gripper": 1}
    state_key_dims = {"state.joints": 7, "state.gripper": 1}
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    # chunk length: has to equal framework.action_model.action_horizon in the yaml. QwenOFT trains on the
    # LAST action_horizon steps of the chunk it is given, without a check, so a longer chunk here is a silent error.
    action_indices = list(range(16))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                # state.gripper is the measured opening (continuous, 1 = open), not a command
                normalization_modes={"state.joints": "min_max", "state.gripper": "min_max"},
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                # action.gripper is the command: 1 = open, 0 = closed
                normalization_modes={"action.joints": "min_max", "action.gripper": "binary"},
            ),
        ])


# ---------------------------------------------------------------------------
# Registry exports
# ---------------------------------------------------------------------------
ROBOT_TYPE_CONFIG_MAP = {
    "xarm7": XArm7DataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    # embodiment_tag lives as a classvar on the DataConfig; kept empty like the RobotWin registry.
}

# ---------------------------------------------------------------------------
# Mixtures: name -> [(dataset directory under data_root_dir, weight, robot type)]
# ---------------------------------------------------------------------------
DATASET_NAMED_MIXTURES = {
    "xarm7_t6_10": [("xarm7_t6_10", 1.0, "xarm7")],
}
