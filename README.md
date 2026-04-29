ustomer Churn Intelligence System

Overview:

This project is an end-to-end machine learning application that predicts customer churn using the Telco dataset and presents insights through an interactive Streamlit dashboard.

The system not only predicts churn but also segments customers into actionable risk categories and provides business recommendations for retention.

Objectives:

* Predict whether a customer will churn.
* Compare multiple machine learning models.
* Visualize insights through an interactive dashboard.
* Provide business-driven retention strategies.

Models Used:

* Logistic Regression (Best Performing)
* Decision Tree
* Random Forest 

Features:

* Data preprocessing and feature encoding
* Model training with cross-validation (Stratified K-Fold)
* Performance metrics: Accuracy, F1-score, ROC-AUC
* Confusion matrices and ROC curves
* Feature importance analysis
* Customer risk segmentation:

  * 🔴 High Risk
  * 🟡 Medium Risk
  * 🟢 Low Risk
* Business insights and retention strategies

Dashboard Sections:

* Overview (KPIs and model comparison)
* Exploratory Data Analysis (EDA)
* Model Performance
* Risk Segmentation
* Business Insights

Tech Stack:

* Python
* Streamlit
* Scikit-learn
* Pandas, NumPy
* Matplotlib, Seaborn


Installation & Run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Dataset:

Telco Customer Churn Dataset

Key Results:

* Best Model: Logistic Regression
* ROC-AUC: 0.84 → strong ability to distinguish churn vs non-churn
* F1-score: 0.59 → moderate performance on imbalanced classification
* Effective churn risk segmentation for business use


Business Impact:

* Identifies high-risk customers for targeted retention
* Enables data-driven decision-making
* Helps reduce churn and improve customer lifetime value


Author:

Adwaith V
MSc Data Science Student
