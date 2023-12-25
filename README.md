# Problem Setting:
Loans are a lucrative product in the banking sector due to their potential for high revenue, but they also entail a proportionate level of risk. Despite banks' stringent assessments of an individual's loan repayment capacity, there are instances when they still experience failures. Hence, it becomes imperative to have a robust technique to select only low-risk applicants before lending loans. The credit score is a metric used to assess the creditworthiness of a customer. The credit score considered various factors such as payment history, credit utilization, length of credit history, types of credit accounts, and recent credit inquiries. In the past, banks have hired highly trained credit analysts to manually calculate the credit score of customers. However, with the advancement of technology, credit score calculations have transitioned to automated processes that utilize statistical models to filter eligible customers with low credit risks. These automated models also leverage historical data to provide more accurate predictions of creditworthiness. This project aims to assess various machine learning techniques to predict loan defaults. This will help approve the loans of low-risk customers only.

# Problem Definition:
Predicting whether an applicant will default or not is a binary classification problem. In this project, we will build various supervised classification models using logistic regression, decision trees, random forest, and boosting that classify each record in the dataset into ‘defaulter’ or ‘non-defaulter’. We then compare the classification models to select the best-performing one. The model primarily tries to answer the following questions.
(i) What is the level of risk associated with the borrower?
(ii) Considering the borrower's risk level, will they repay the loan or not, and what could be the best ML model to predict it? This is in fact the primary objective(prediction) of the model
(iii) Along the way, we will try to answer questions or identify patterns related to the distribution of loan purpose, how loan purpose and the loan amount are related, the distribution of interest rate, and the relationship between loan default and home ownership/employment through exploratory data analysis.

# Data Sources:
Lending Club Issued Loans: this dataset is from Kaggle. It contains the lending club’s complete loan data issued from 2007-2015.
LendingClub is one of the largest and most well-known online peer-to-peer lending platforms that connects borrowers with investors in the US. Through LendingClub individuals or businesses in need of loans can borrow money directly from investors. The platform assesses the creditworthiness of loan applicants and helps investors in decision-making.
Dataset link: https://www.kaggle.com/datasets/husainsb/lendingclub-issued-loans (we will be using the  “lc_loan.csv” file)


