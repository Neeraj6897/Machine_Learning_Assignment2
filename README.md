# Machine Learning Assignment 2: Bank Marketing Classification

## a. Problem Statement

Banks contact many customers during their marketing campaigns, but only some of them subscribe to a term deposit. The aim of this assignment is to use the information collected during these campaigns and predict whether a customer will subscribe (`yes`) or not (`no`).

This is a binary classification problem. I trained the five classification models given in the assignment on the same dataset and compared them using Accuracy, AUC, Precision, Recall, F1 Score and MCC.

## b. Dataset Description

- **Dataset Name:** Bank Marketing Dataset
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- **Local Dataset File:** `data/bank-additional-full.csv`
- **Number of Instances:** 41,188
- **Number of Features:** 20 input features + 1 target variable
- **Target Variable:** `y` (whether the client subscribed to a term deposit: yes/no)
- **Feature Types:** The dataset contains both categorical features, such as job, marital status and education, and numerical features, such as age, duration and number of campaign contacts.
- **Classification Type:** Binary Classification

**Key Features:**
| Feature | Description |
|---------|-------------|
| age | Age of the client |
| job | Type of job (admin, technician, services, etc.) |
| marital | Marital status |
| education | Education level |
| default | Has credit in default? |
| housing | Has housing loan? |
| loan | Has personal loan? |
| contact | Communication type |
| month | Last contact month |
| day_of_week | Last contact day |
| duration | Last contact duration in seconds |
| campaign | Number of contacts during campaign |
| pdays | Days since last contact from previous campaign |
| previous | Number of contacts before this campaign |
| poutcome | Outcome of previous campaign |
| emp.var.rate | Employment variation rate |
| cons.price.idx | Consumer price index |
| cons.conf.idx | Consumer confidence index |
| euribor3m | Euribor 3 month rate |
| nr.employed | Number of employees |

## c. GitHub Repository Link

GitHub Repository Link: *[https://github.com/Neeraj6897/Machine_Learning_Assignment2](https://github.com/Neeraj6897/Machine_Learning_Assignment2)*

## d. Models Used

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9139 | 0.9370 | 0.7002 | 0.4127 | 0.5193 | 0.4956 |
| Decision Tree | 0.8956 | 0.7535 | 0.5343 | 0.5700 | 0.5516 | 0.4929 |
| KNN | 0.9053 | 0.8617 | 0.6267 | 0.3944 | 0.4841 | 0.4491 |
| Naive Bayes | 0.8536 | 0.8606 | 0.4024 | 0.6175 | 0.4872 | 0.4189 |
| Random Forest (Ensemble) | 0.9205 | 0.9491 | 0.6898 | 0.5345 | 0.6023 | 0.5645 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | This model gave an accuracy of 0.9139 and an AUC of 0.9370. Its precision was good, but the recall was only 0.4127. This means that when the model predicts a subscription it is often correct, but it still misses many customers who actually subscribed. |
| Decision Tree | The Decision Tree found more positive cases than Logistic Regression and KNN, as seen from its recall of 0.5700. However, its accuracy and precision were lower, and it had the lowest AUC among the five models. |
| KNN | KNN achieved an accuracy of 0.9053, but its recall of 0.3944 was the lowest. It predicted the majority class reasonably well but did not identify enough of the customers who subscribed. Scaling was used because KNN depends on the distance between data points. |
| Naive Bayes | Naive Bayes had the lowest accuracy and precision. At the same time, it had the highest recall of 0.6175, so it identified more actual subscribers than the other models. The low precision shows that it also produced more false-positive predictions. |
| Random Forest (Ensemble) | Random Forest produced the highest Accuracy, AUC, F1 Score and MCC. Its recall was lower than Naive Bayes and Decision Tree, but it gave a better balance between precision and recall. |
| **Overall Winner for this dataset** | I selected **Random Forest (Ensemble)** as the overall winner because it gave the best Accuracy, AUC, F1 Score and MCC on the test data. Based on the complete set of metrics, it was the most balanced model for this dataset. |

## Live Streamlit App Link

Streamlit App Link: https://2025ad05063-machine-learning-assignment2.streamlit.app
