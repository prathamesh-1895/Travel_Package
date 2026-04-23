# 🌴 Holiday Package Prediction System

A machine learning-powered web application that predicts whether a customer is likely to purchase a holiday package based on demographic and behavioral data.

---

## 🚀 Overview

This project uses multiple machine learning models to analyze customer data and predict purchase intent. It also provides an interactive dashboard for data exploration, model comparison, and real-time predictions.

The application is built using **Streamlit** for the frontend and **Scikit-learn/XGBoost** for machine learning.

---

## ✨ Features

### 📊 Dashboard

* Overview of customer data
* Purchase vs non-purchase distribution
* Key performance metrics (ROC-AUC, F1 Score)

### 🔍 Exploratory Data Analysis

* Distribution plots for numerical features
* Categorical analysis vs purchase behavior
* Correlation heatmaps and scatter plots

### 🤖 Model Comparison

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost
* Performance comparison using multiple metrics

### 📈 Model Evaluation

* ROC Curve
* Precision-Recall Curve
* Confusion Matrix
* Threshold tuning for better predictions

### 🎯 Customer Prediction

* Real-time prediction based on user input
* Probability score with visual gauge
* Key influencing factors highlighted

---

## 🛠️ Tech Stack

* Python
* Streamlit
* Pandas & NumPy
* Scikit-learn
* Imbalanced-learn (SMOTE)
* XGBoost
* Plotly

---

## 📂 Project Structure

```
.
├── app.py                # Main Streamlit application
├── requirements.txt     # Dependencies
└── README.md            # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the application

```
streamlit run app.py
```

---

## 🌐 Deployment

You can deploy this app easily on:

* Streamlit Cloud
* Render
* Railway

Just connect your GitHub repo and it will automatically use `requirements.txt`.

---

## 📊 Machine Learning Workflow

1. Data preprocessing
2. Feature engineering
3. Handling class imbalance using SMOTE
4. Training multiple models
5. Model evaluation and selection
6. Threshold optimization
7. Real-time prediction

---

## 🎯 Use Cases

* Travel companies to identify potential customers
* Marketing teams for targeted campaigns
* Customer segmentation and behavior analysis

---

## 📌 Future Improvements

* Use real-world dataset instead of synthetic data
* Add user authentication
* Integrate database for storing predictions
* Deploy as a SaaS product

---

## 👤 Author

**Prathamesh Shendge**

---

## 📜 License

This project is for educational and demonstration purposes.
