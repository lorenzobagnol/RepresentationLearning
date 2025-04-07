from dataclasses import dataclass, asdict
from typing import Any
from utils.inputdata import InputData


@dataclass
class BaseConfig:
    def to_dict(self):
        """Convert dataclass to dictionary for flexibility."""
        return asdict(self)

@dataclass
class SOMConfig(BaseConfig):
    """Configuration for SOM."""
    M: int
    N: int
    INPUT_DATA: InputData

@dataclass
class WandBConfig(BaseConfig):
    """Configuration for WeightsAndBiases."""
    PROJECT: str

@dataclass
class LifeLongConfig(BaseConfig):
    """Configuration for lifelong learning."""
    SIGMA: float
    TARGET_RADIUS: float
    ALPHA: float
    BETA: float
    BATCH_SIZE: int
    EPOCHS_PER_SUBSET: int
    SUBSET_SIZE: int
    DISJOINT_TRAINING: bool
    LR_GLOBAL_BASELINE: float
    SIGMA_BASELINE: float
    LEARNING_RATE: float
    MODE: str

@dataclass
class PytorchBatchConfig(BaseConfig):
    """PyTorch-specific batch training configuration."""
    SIGMA: float
    TARGET_RADIUS: float
    EPOCHS: int
    BATCH_SIZE: int
    LEARNING_RATE: float
    BETA: float
    MODE: str

@dataclass
class VARS(BaseConfig):
    def __init__(self, **kargs: Any):
        for key, value in kargs.items():
            setattr(self, key, value)

@dataclass
class Config:
    """Master configuration class that can hold all the sub-configurations."""
    SEED: int
    weights_and_biases_config: WandBConfig
    som_config: SOMConfig 
    LifeLong_config: LifeLongConfig 
    pytorch_batch_config: PytorchBatchConfig 
    variables: VARS

