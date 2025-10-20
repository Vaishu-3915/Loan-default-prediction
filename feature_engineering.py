import pandas as pd
from sklearn.preprocessing import OneHotEncoder


EMP_LENGTH_MAP = {
    '< 1 year': 0,
    '1 year': 1,
    '2 years': 2,
    '3 years': 3,
    '4 years': 4,
    '5 years': 5,
    '6 years': 6,
    '7 years': 7,
    '8 years': 8,
    '9 years': 9,
    '10+ years': 10,
}


ONE_HOT_COLS = ['purpose', 'term', 'verification_status', 'home_ownership']
NUMERIC_COLS = [
    'loan_amnt', 'emp_length', 'int_rate', 'annual_inc', 'dti', 'delinq_2yrs',
    'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util',
    'total_acc', 'collections_12_mths_ex_med', 'acc_now_delinq', 'credit_length'
]


def add_credit_length(df: pd.DataFrame) -> pd.DataFrame:
    if 'issue_d' in df.columns and 'earliest_cr_line' in df.columns:
        issue_year = df['issue_d'].astype(str).str.split('-').str[-1].str.extract(r'(\d{2,4})').astype(float)
        earliest_year = df['earliest_cr_line'].astype(str).str.split('-').str[-1].str.extract(r'(\d{2,4})').astype(float)
        df['credit_length'] = (issue_year.values - earliest_year.values).astype(float)
        df.drop(columns=['issue_d', 'earliest_cr_line'], inplace=True, errors='ignore')
    return df


def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Drop columns that shouldn't be in the cleaned data (from original notebook)
    drop_cols = [
        'addr_state', 'initial_list_status', 'mths_since_last_delinq',
        'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim',
        'emp_title', 'title', 'application_type'
    ]
    for col in drop_cols:
        if col in df.columns:
            df.drop(columns=col, inplace=True)
    
    # Drop rows with missing values (as in original notebook)
    df.dropna(inplace=True)
    
    # Map employment length
    if 'emp_length' in df.columns:
        df['emp_length'] = df['emp_length'].map(EMP_LENGTH_MAP)

    # Add credit length
    df = add_credit_length(df)

    # One-hot encode categorical columns
    present_ohe = [c for c in ONE_HOT_COLS if c in df.columns]
    if present_ohe:
        enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        enc_arr = enc.fit_transform(df[present_ohe])
        enc_df = pd.DataFrame(enc_arr, columns=enc.get_feature_names_out(present_ohe), index=df.index)
        df = pd.concat([df.drop(columns=present_ohe), enc_df], axis=1)

    # Remove known highly correlated or redundant features if present
    redundant_cols = ['installment', 'grade', 'sub_grade', 'term_ 36 months', 'home_ownership_RENT']
    for col in redundant_cols:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # Ensure target at end if present
    if 'loan_default' in df.columns:
        target = df['loan_default']
        df = df.drop(columns=['loan_default'])
        df['loan_default'] = target

    return df


