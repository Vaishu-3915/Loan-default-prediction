import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_target_distribution(df: pd.DataFrame) -> None:
    if 'loan_default' not in df.columns:
        return
    sns.countplot(x='loan_default', data=df)
    plt.title('Target distribution')
    plt.tight_layout()
    plt.show()


def plot_numeric_histograms(df: pd.DataFrame, numeric_cols: list[str]) -> None:
    for col in numeric_cols:
        if col in df.columns:
            sns.histplot(df[col], bins=30)
            plt.title(f'{col} distribution')
            plt.tight_layout()
            plt.show()


