from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    input_csv_path: str = "data/lc_loan.csv"
    output_dir: str = "artifacts"
    cleaned_filename: str = "lc_loan_cleaned.parquet"
    chunk_size: int = 100_000
    low_memory: bool = True
    save_intermediate_csv: bool = False


@dataclass
class ModelingConfig:
    test_size: float = 0.3
    random_state: int = 42
    positive_class_weight: Optional[float] = 5.0


@dataclass
class Config:
    data: DataConfig
    modeling: ModelingConfig = field(default_factory=ModelingConfig)


