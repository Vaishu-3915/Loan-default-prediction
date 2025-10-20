from typing import Tuple, List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from config import ModelingConfig


NUMERIC_COLUMNS: List[str] = [
    'loan_amnt', 'emp_length', 'int_rate', 'annual_inc', 'dti', 'delinq_2yrs',
    'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util',
    'total_acc', 'collections_12_mths_ex_med', 'acc_now_delinq', 'credit_length',
]


def split_data(df: pd.DataFrame, cfg: ModelingConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=['loan_default'])
    y = df['loan_default']
    return train_test_split(X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y)


def scale_numeric(X_train: pd.DataFrame, X_valid: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    present_numeric = [c for c in NUMERIC_COLUMNS if c in X_train.columns]
    scaler = StandardScaler()
    X_train_scaled_num = pd.DataFrame(
        scaler.fit_transform(X_train[present_numeric]),
        columns=present_numeric,
        index=X_train.index,
    )
    X_valid_scaled_num = pd.DataFrame(
        scaler.transform(X_valid[present_numeric]),
        columns=present_numeric,
        index=X_valid.index,
    )
    X_train_scaled = pd.concat([X_train.drop(columns=present_numeric), X_train_scaled_num], axis=1)
    X_valid_scaled = pd.concat([X_valid.drop(columns=present_numeric), X_valid_scaled_num], axis=1)
    return X_train_scaled, X_valid_scaled


def train_logistic(X_train: pd.DataFrame, y_train: pd.Series, class_weight_pos: float | None) -> LogisticRegression:
    cw = None if class_weight_pos is None else {0: 1.0, 1: class_weight_pos}
    model = LogisticRegression(max_iter=1000, solver='liblinear', class_weight=cw, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        bootstrap=True,
        class_weight='balanced',
        max_features='sqrt',
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_valid: pd.DataFrame, y_valid: pd.Series) -> dict:
    y_pred = model.predict(X_valid)
    report = classification_report(y_valid, y_pred, output_dict=True)
    if hasattr(model, 'predict_proba'):
        auc = roc_auc_score(y_valid, model.predict_proba(X_valid)[:, 1])
    else:
        auc = roc_auc_score(y_valid, y_pred)
    return {"classification_report": report, "roc_auc": auc}


