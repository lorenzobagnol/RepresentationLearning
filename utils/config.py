from dataclasses import dataclass, asdict


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
    SIGMA: float

@dataclass
class LifeLongConfig(BaseConfig):
    """Configuration for lifelong learning."""
    ALPHA: float
    BETA: float
    BATCH_SIZE: int
    EPOCHS_PER_SUBSET: int
    SUBSET_SIZE: int
    DISJOINT_TRAINING: bool
    LR_GLOBAL_BASELINE: float
    SIGMA_BASELINE: float
    LEARNING_RATE: float

@dataclass
class SimpleBatchConfig(BaseConfig):
    """Simple batch training configuration."""
    EPOCHS: int
    BATCH_SIZE: int

@dataclass
class PytorchBatchConfig(BaseConfig):
    """PyTorch-specific batch training configuration."""
    EPOCHS: int
    BATCH_SIZE: int
    LEARNING_RATE: float

@dataclass
class OnlineConfig(BaseConfig):
    """Online training configuration."""
    EPOCHS: int

@dataclass
class Config:
    """Master configuration class that can hold all the sub-configurations."""
    SEED: int
    som_config: SOMConfig 
    lifelong_config: LifeLongConfig 
    simple_batch_config: SimpleBatchConfig 
    pytorch_batch_config: PytorchBatchConfig 
    online_config: OnlineConfig 
