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
class SimpleBatchConfig(BaseConfig):
    """Simple batch training configuration."""
    SIGMA: float
    EPOCHS: int
    BATCH_SIZE: int
    BETA: float

@dataclass
class PytorchBatchConfig(BaseConfig):
    """PyTorch-specific batch training configuration."""
    SIGMA: float
    EPOCHS: int
    BATCH_SIZE: int
    LEARNING_RATE: float
    BETA: float
    MODE: str

@dataclass
class OnlineConfig(BaseConfig):
    """Online training configuration."""
    SIGMA: float
    EPOCHS: int

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
    simple_batch_config: SimpleBatchConfig 
    pytorch_batch_config: PytorchBatchConfig 
    online_config: OnlineConfig 
    variables: VARS

