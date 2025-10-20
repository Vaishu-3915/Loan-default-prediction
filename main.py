import os
import pandas as pd
from config import Config, DataConfig, ModelingConfig
from utils import ensure_dir
from data_cleaning import compute_high_null_columns, clean_and_write_stream
from feature_engineering import transform
from model_training import split_data, scale_numeric, train_logistic, train_random_forest, evaluate


def run_pipeline(input_csv_path: str = None, cleaned_csv_path: str = None, artifacts_dir: str = "artifacts") -> None:
    cfg = Config(
        data=DataConfig(input_csv_path=input_csv_path or "data/lc_loan.csv", output_dir=artifacts_dir, save_intermediate_csv=True),
        modeling=ModelingConfig(),
    )

    ensure_dir(cfg.data.output_dir)

    # Use provided cleaned dataset or clean from scratch
    if cleaned_csv_path:
        df_fe = pd.read_csv(cleaned_csv_path)
        print(f"Using provided cleaned dataset: {cleaned_csv_path}")
        print(f"Dataset shape: {df_fe.shape}")
    else:
        # First pass to detect high-null columns
        high_null_cols = compute_high_null_columns(
            pd.read_csv(cfg.data.input_csv_path, chunksize=cfg.data.chunk_size, low_memory=cfg.data.low_memory)
        )

        # Clean stream and write cleaned CSV
        cleaned_output_path = clean_and_write_stream(cfg.data.input_csv_path, cfg.data, high_null_cols)

        # Load cleaned data (Parquet or CSV)
        if cleaned_output_path.endswith('.parquet'):
            df = pd.read_parquet(cleaned_output_path)
        else:
            df = pd.read_csv(cleaned_output_path)

        # Feature engineering
        df_fe = transform(df)

    # Train/valid split
    X_train, X_valid, y_train, y_valid = split_data(df_fe, cfg.modeling)

    # Scale numeric for linear models
    X_train_s, X_valid_s = scale_numeric(X_train, X_valid)

    # Train models
    lr = train_logistic(X_train_s, y_train, cfg.modeling.positive_class_weight)
    rf = train_random_forest(X_train, y_train)

    # Evaluate
    lr_metrics = evaluate(lr, X_valid_s, y_valid)
    rf_metrics = evaluate(rf, X_valid, y_valid)

    # Save metrics
    metrics_path = os.path.join(cfg.data.output_dir, 'metrics.txt')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write('Logistic Regression\n')
        f.write(str(lr_metrics) + '\n\n')
        f.write('Random Forest\n')
        f.write(str(rf_metrics) + '\n')

    print(f"Pipeline complete. Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    # Using the cleaned dataset from the notebook
    run_pipeline(cleaned_csv_path="data/lc_loan_cleaned.csv")


