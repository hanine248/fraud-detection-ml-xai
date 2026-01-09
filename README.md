📌 Project Overview

This project focuses on detecting fraudulent credit card transactions using machine learning models on a highly imbalanced real-world dataset.
In addition to building accurate classifiers, the project emphasizes model interpretability using SHAP (SHapley Additive exPlanations) to explain both global and local predictions — a critical requirement in financial and risk-sensitive domains.

🎯 Objectives

Detect fraudulent transactions with high precision and balanced recall

Handle extreme class imbalance effectively

Compare multiple machine learning models

Provide explainable predictions at both dataset and individual-transaction levels

Reduce false positives to minimize operational cost

📊 Dataset

Name: Credit Card Fraud Detection Dataset

Source: Kaggle (ULB – Machine Learning Group)

Transactions: 284,807

Fraud cases: 492 (≈0.17%)

Due to GitHub file size limitations, the dataset is not included in this repository.

🔗 Download the dataset from:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

After downloading, place creditcard.csv in the project root directory.

🧠 Machine Learning Models

The following models were trained and evaluated:

Logistic Regression

Random Forest Classifier

✅ Best Model: Random Forest

The Random Forest classifier achieved the best overall performance:

Fraud Precision: 96%

Recall: 74%

F1-Score: 0.84

Although Logistic Regression achieved slightly higher recall, Random Forest significantly reduced false positives, making it more suitable for real-world deployment where investigation cost matters.

📈 Evaluation Metrics

Precision

Recall

F1-Score

Confusion Matrix

ROC-AUC Score

Special focus was placed on precision-recall trade-off due to dataset imbalance.

🔍 Explainable AI with SHAP
Global Explanations

SHAP summary (beeswarm) plots were used to identify the most influential features driving fraud predictions.

Features such as V12, V14, V10, and V3 had the strongest global impact.

Local Explanations

SHAP waterfall plots were generated to explain individual fraud predictions.

Each transaction prediction is decomposed into feature-level contributions showing how the model arrived at its decision.

This ensures transparency, trust, and auditability, which are critical in financial applications.

🛠️ Technologies Used

Python 3.10

NumPy

Pandas

Scikit-learn

Matplotlib / Seaborn

SHAP

Jupyter Notebook

Anaconda

🧪 Environment Setup
Create Conda Environment
conda create -n fraud_ml python=3.10 -y
conda activate fraud_ml

Install Dependencies
conda install numpy pandas scikit-learn matplotlib seaborn ipykernel -y
conda install -c conda-forge shap -y

Register Kernel
python -m ipykernel install --user --name fraud_ml --display-name "Python (fraud_ml)"


In Jupyter Notebook, select:

Kernel → Python (fraud_ml)

📂 Project Structure
fraud_detection/
│
├── Fraud_Detection.ipynb   # Main notebook (EDA, modeling, SHAP)
├── creditcard.csv          # Dataset (not included)
├── README.md               # Project documentation

🚀 How to Run

Clone the repository

Download the dataset from Kaggle

Place creditcard.csv in the project directory

Activate the fraud_ml environment

Open Fraud_Detection.ipynb

Run all cells

💡 Key Takeaways

Handling class imbalance is crucial in fraud detection

High accuracy alone is misleading; precision and recall matter more

Explainable AI bridges the gap between performance and trust

SHAP provides actionable insights for both analysts and decision-makers

📌 Future Improvements

SMOTE / advanced resampling techniques

Cost-sensitive learning

Model deployment as a REST API

Real-time fraud scoring pipeline

👤 Author

Hanin Tarhini
Machine Learning & Telecommunications Engineering
Focused on AI, Explainable ML, and Real-World Systems
