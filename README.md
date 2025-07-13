# DataScience
# DevelopersHub Data Science Internship Tasks

This repository contains my completed tasks for the **Data Science & Analytics Internship** at DevelopersHub Corporation.

## ✅ Completed Tasks

### 🔹 Task 1: Exploring and Visualizing a Simple Dataset
- Dataset: Iris Dataset
- Objective: Understand and visualize the dataset.
- Tools: pandas, seaborn, matplotlib

## Steps Covered
- Data loading from CSV
- Summary statistics and structure
- Scatter plot: Sepal vs Petal length
- Histograms and boxplots for feature distribution and outliers

## Tools Used
- Python
- Pandas
- Matplotlib



---

### 🔹 Task 2: Credit Risk Prediction
- Dataset: Loan Prediction Dataset (Kaggle)
- Objective: Predict loan default likelihood.
- Techniques: Data cleaning, Logistic Regression, Decision Tree

---

## 🗃️ Dataset
- **Source**: [Loan Prediction Dataset – Kaggle](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
- **File Used**: `LoadPredTrain.csv` and 'LoanPrediction.csv'
- Contains features like:
  - Gender, Married, Education, Self_Employed
  - ApplicantIncome, CoapplicantIncome, LoanAmount
  - Credit_History, Property_Area, etc.

---

## 🧪 Steps Performed

### ✅ Data Preprocessing
- Filled missing numeric values using **mean**.
- Filled missing categorical values using **mode**.
- Encoded categorical features using **LabelEncoder**.
- Engineered new features:
  - `Total_Income` = Applicant + Coapplicant Income
  - `Income_to_Loan_Ratio` = Total_Income / LoanAmount
- Dropped noisy and redundant features (like `Loan_ID`, `ApplicantIncome`, `LoanAmount`).

---

### 🔍 Models Trained

#### 1. **Decision Tree Classifier**
- Used `entropy` as the splitting criterion.
- Tuned parameters: `max_depth=5`, `min_samples_split=4`, `min_samples_leaf=2`.
- Achieved accuracy: **~73%**

#### 2. **Logistic Regression**
- Trained with `StandardScaler` for normalization.
- Used `L2` regularization (`penalty='l2'`, `C=0.8`).
- Achieved accuracy: **~75%**
- Delivered better recall and precision balance compared to Decision Tree.

---

## 📊 Evaluation Metrics

Both models were evaluated using:
- **Accuracy Score**
- **Confusion Matrix**
- **Classification Report** (precision, recall, F1-score)

---

## 📈 Insights & Learnings

- Logistic Regression outperformed Decision Tree slightly on both accuracy and generalization.
- Feature engineering (e.g., `Income_to_Loan_Ratio`) helped improve prediction quality.
- Most predictive features included **Credit_History**, **Education**, and **Total_Income**.

---

## ✅ Status
✔ Completed – Both models implemented, evaluated, and optimized.


---

# Task 3: Customer Churn Prediction

## 🎯 Objective
Predict which bank customers are likely to leave the bank using classification techniques and analyze what influences their decision.

## 🗂 Dataset
- Source: Churn Modelling Dataset (`Churn_Modelling.csv`)
- Target: `Exited` (1 = churned, 0 = stayed)

## 🛠️ Process
- Removed irrelevant columns (`RowNumber`, `CustomerId`, `Surname`)
- Encoded categorical features (`Gender` with Label Encoding, `Geography` with One-Hot Encoding)
- Scaled numeric features using `StandardScaler`
- Trained a `RandomForestClassifier` with tuned hyperparameters
- Evaluated using accuracy, confusion matrix, and classification report

## 🔍 Feature Insights
- **Age**, **Balance**, and **IsActiveMember** were the top indicators of churn.
- Customers from certain regions (e.g., **Germany**) were more likely to churn.
- Inactive members with high balances showed the highest churn risk.

## ✅ Status
✔ Completed – Model built, evaluated, and interpreted.
---

# Task 4: Predicting Insurance Claim Amounts

## 🎯 Objective
Estimate a person's medical insurance charges based on their age, BMI, smoking status, and other personal factors using a regression model.

## 🗂 Dataset
- Source: Medical Cost Personal Dataset (`insurance.csv`)
- Target variable: `charges` (insurance claim amount)

## 🛠️ Process
- Encoded categorical features (`sex`, `smoker`, `region`) using Label Encoding
- Visualized relationships between `age`, `bmi`, `smoker` and `charges`
- Trained a Linear Regression model
- Evaluated performance using:
  - **MAE** (Mean Absolute Error)
  - **RMSE** (Root Mean Squared Error)

## 🔍 Key Insights
- **Smokers** have significantly higher charges
- **BMI** and **age** positively influence insurance costs
- Linear Regression provides a good baseline for cost prediction

## ✅ Status
✔ Completed – model trained, evaluated, and insights visualized


---

### 🔹 Task 5: Personal Loan Acceptance Prediction
- Dataset: Bank Marketing Dataset (UCI)
- Objective: Predict personal loan acceptance using classification.

## 🗂 Dataset
- Source: UCI Bank Marketing Dataset
- File used: `bank-additional-full.csv`
- Target: `y` (yes = 1, no = 0)

## 🛠️ Methodology
- One-hot encoded categorical variables
- Standardized numeric features using `StandardScaler`
- Trained a Logistic Regression model with tuned hyperparameters
- Evaluated with accuracy, confusion matrix, and classification report

## 📊 Key Insights
- Customers contacted in **August**, with **successful past campaigns**, and **university degrees** are more likely to accept offers.
- Customers contacted via **telephone** or in **May**, or who received multiple contacts, tend to decline.
- `duration` was the strongest predictor but is only available post-call, so not usable for pre-contact predictions.

## ✅ Status
✔ Completed – model built, interpreted, and documented.


---

## 🛠 Technologies Used
- Python
- Jupyter Notebook
- pandas, matplotlib, seaborn
- scikit-learn

## 📦 Folder Structure

