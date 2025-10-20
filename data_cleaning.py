from typing import List, Iterator
import os
import pandas as pd
from config import DataConfig
from utils import ensure_dir


HIGH_NULL_DROP_THRESHOLD = 0.7


POST_ISSUANCE_COLUMNS: List[str] = [
    'funded_amnt', 'funded_amnt_inv', 'pymnt_plan', 'out_prncp',
    'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp',
    'policy_code', 'total_rec_int', 'total_rec_late_fee', 'recoveries',
    'collection_recovery_fee', 'last_pymnt_d', 'next_pymnt_d', 'last_pymnt_amnt',
    'last_credit_pull_d', 'zip_code', 'member_id', 'id', 'url'
]


TARGET_FILTER_EXCLUDE = {'Current', 'Issued', 'In Grace Period'}


def compute_high_null_columns(df_iter: Iterator[pd.DataFrame], threshold: float = HIGH_NULL_DROP_THRESHOLD) -> List[str]:
    total_rows = 0
    null_counts = None
    for chunk in df_iter:
        total_rows += len(chunk)
        chunk_null = chunk.isnull().sum()
        if null_counts is None:
            null_counts = chunk_null
        else:
            null_counts = null_counts.add(chunk_null, fill_value=0)
    if null_counts is None or total_rows == 0:
        return []
    null_ratio = null_counts / float(total_rows)
    return null_ratio[null_ratio > threshold].index.astype(str).tolist()


def filter_and_basic_drop(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'loan_status' in df.columns:
        df = df[~df['loan_status'].isin(TARGET_FILTER_EXCLUDE)]
    drop_cols = [c for c in POST_ISSUANCE_COLUMNS if c in df.columns]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
    return df


def clean_and_write_stream(
    src_path: str,
    cfg: DataConfig,
    high_null_cols: List[str],
) -> str:
    ensure_dir(cfg.output_dir)
    
    # Collect all cleaned chunks in memory for final parquet write
    cleaned_chunks = []
    
    for chunk in pd.read_csv(src_path, chunksize=cfg.chunk_size, low_memory=cfg.low_memory):
        if high_null_cols:
            keep_cols = [c for c in chunk.columns if c not in high_null_cols]
            chunk = chunk[keep_cols]
        
        chunk = filter_and_basic_drop(chunk)
        if len(chunk) == 0:
            continue
            
        # Derive target column
        if 'loan_status' in chunk.columns:
            chunk['loan_default'] = 1
            mask_non_default = chunk['loan_status'].isin([
                'Fully Paid', 'Does not meet the credit policy. Status:Fully Paid'
            ])
            chunk.loc[mask_non_default, 'loan_default'] = 0
            # Drop after deriving target
            chunk.drop(columns=['loan_status'], inplace=True)
        
        cleaned_chunks.append(chunk)

    # Combine all chunks and write as parquet
    if cleaned_chunks:
        final_df = pd.concat(cleaned_chunks, ignore_index=True)
        
        # Ensure output filename has .parquet extension
        if not cfg.cleaned_filename.endswith('.parquet'):
            base_name = os.path.splitext(cfg.cleaned_filename)[0]
            output_path = os.path.join(cfg.output_dir, f"{base_name}.parquet")
        else:
            output_path = os.path.join(cfg.output_dir, cfg.cleaned_filename)
            
        final_df.to_parquet(output_path, index=False, compression='snappy')
        
        # Also save CSV if requested
        if cfg.save_intermediate_csv:
            csv_path = os.path.splitext(output_path)[0] + '.csv'
            final_df.to_csv(csv_path, index=False)
            
        return output_path
    else:
        raise ValueError("No valid data chunks found after cleaning")


