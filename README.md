# Loan Default Prediction - Refactored Pipeline

This project builds and evaluates machine learning models to predict whether a borrower will default on a loan, using LendingClub historical loan data. It includes end-to-end steps from data preparation and feature engineering to training and evaluation, implemented as reproducible Python scripts.

## Problem Setting

Loans are a lucrative product in the banking sector due to their potential for high revenue, but they also entail a proportionate level of risk. Despite banks' stringent assessments of an individual's loan repayment capacity, there are instances when they still experience failures. Hence, it becomes imperative to have a robust technique to select only low-risk applicants before lending loans. 

The credit score is a metric used to assess the creditworthiness of a customer. The credit score considers various factors such as payment history, credit utilization, length of credit history, types of credit accounts, and recent credit inquiries. In the past, banks have hired highly trained credit analysts to manually calculate the credit score of customers. However, with the advancement of technology, credit score calculations have transitioned to automated processes that utilize statistical models to filter eligible customers with low credit risks. These automated models also leverage historical data to provide more accurate predictions of creditworthiness. 

This project aims to assess various machine learning techniques to predict loan defaults. This will help approve the loans of low-risk customers only.

## Problem Definition

Predicting whether an applicant will default or not is a binary classification problem. In this project, we will build various supervised classification models using logistic regression, decision trees, random forest, and boosting that classify each record in the dataset into 'defaulter' or 'non-defaulter'. We then compare the classification models to select the best-performing one. The model primarily tries to answer the following questions:

(i) What is the level of risk associated with the borrower?
(ii) Considering the borrower's risk level, will they repay the loan or not, and what could be the best ML model to predict it? This is in fact the primary objective (prediction) of the model
(iii) Along the way, we will try to answer questions or identify patterns related to the distribution of loan purpose, how loan purpose and the loan amount are related, the distribution of interest rate, and the relationship between loan default and home ownership/employment through exploratory data analysis.

## Project Structure

```
├── main.py                      # Main entry point
├── config.py                    # Configuration settings
├── utils.py                     # Utility functions
├── data_loading.py              # Data ingestion utilities
├── data_cleaning.py             # Data preprocessing and cleaning
├── feature_engineering.py       # Feature transformation and encoding
├── model_training.py            # Model building and evaluation
├── exploratory_data_analysis.py # EDA and visualizations
├── requirements.txt             # Python dependencies
└── artifacts/                   # Output directory (created automatically)
    ├── lc_loan_cleaned.parquet  # Cleaned dataset (Parquet format)
    └── metrics.txt              # Model performance metrics
```

## Key Features

1. **Memory Efficient Processing**: Chunked data reading (100K rows) prevents memory overflow on large datasets
2. **Data Leakage Prevention**: Systematic removal of post-issuance columns that contain future information
3. **Automated Feature Engineering**: One-hot encoding, credit length calculation, and correlation removal
4. **Class Imbalance Handling**: Balanced class weights and sampling strategies for better default prediction
5. **Production-Ready Architecture**: Modular design with configuration management and error handling

## Data Source

This project uses the **LendingClub Loan Data** dataset:
- **Source**: LendingClub (peer-to-peer lending platform)
- **Dataset**: Historical loan data with borrower information and loan outcomes
- **Size**: ~400K loan records with 75+ features
- **Time Period**: 2007-2015
- **Availability**: Public dataset available through [Kaggle](https://www.kaggle.com/datasets/husainsb/lendingclub-issued-loans) or LendingClub website
- **Note**: Due to size, data files are not included in this repository

LendingClub is one of the largest and most well-known online peer-to-peer lending platforms that connects borrowers with investors in the US. Through LendingClub, individuals or businesses in need of loans can borrow money directly from investors. The platform assesses the creditworthiness of loan applicants and helps investors in decision-making.

## Usage

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download the dataset**:
   - Obtain LendingClub loan data (e.g., from Kaggle or LendingClub website)
   - Place `lc_loan.csv` in the `data/` folder

3. **Run the pipeline**:
   ```bash
   python main.py
   ```

   Or use your own cleaned dataset:
   ```bash
   python -c "from main import run_pipeline; run_pipeline(cleaned_csv_path='data/lc_loan_cleaned.csv')"
   ```

## Configuration

Edit `config.py` to adjust:
- Data paths and output directory
- Chunk size for processing
- Model parameters
- Class weights for imbalanced data

## Data Flow

1. **Loading**: Read CSV in chunks to detect high-null columns
2. **Cleaning**: Remove high-null columns, filter target, derive binary target
3. **Feature Engineering**: Encode categoricals, create credit length, remove redundant features
4. **Modeling**: Train Logistic Regression and Random Forest with proper scaling
5. **Evaluation**: Generate classification reports and ROC-AUC scores

## Output

The pipeline generates the following outputs in the `artifacts/` folder:
- `lc_loan_cleaned.parquet`: Cleaned dataset (Parquet format for efficiency)
- `metrics.txt`: Model performance metrics (classification reports and ROC-AUC scores)
- Optional CSV output if `save_intermediate_csv=True` in config

**Note**: The `artifacts/` folder is excluded from version control due to file size.

## Memory Considerations

The pipeline uses chunked processing for data cleaning but loads the full cleaned dataset into memory for modeling. For very large datasets (>1GB), consider:
- Increasing chunk size in config
- Using streaming feature engineering
- Implementing model training in batches
