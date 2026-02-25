# 🛡️ ChurnGuard Pro: AI-Powered Customer Retention Analytics

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

> **Developed for:** Afynix Digital  
> **Objective:** Reduce customer attrition through predictive modeling and real-time risk assessment.

## 📋 Project Overview
This project is a **Predictive Analytics Dashboard** designed for Telecommunications providers. It uses Machine Learning to identify "at-risk" customers who are likely to cancel their subscriptions (Churn). By providing early warnings and identifying risk drivers, this tool helps businesses protect revenue and improve customer loyalty.



## ✨ Key Features
* **Real-time Risk Prediction:** Instant churn probability using a Random Forest Classifier.
* **Risk Driver Analysis:** Interactive bar charts showing exactly which factors (e.g., Monthly Charges, Tenure) are driving the churn risk.
* **Technical Performance Report:** Built-in evaluation metrics including a **Confusion Matrix** and **Accuracy Score**.
* **Professional UI:** High-contrast Dark Mode interface with an organized, user-friendly layout.

## 📊 The Dataset
The model is trained on the **IBM Telco Customer Churn** dataset, which includes:
- **Demographics:** Gender, Senior Citizen status, Partners, and Dependents.
- **Services:** Phone, Multiple Lines, Internet (DSL/Fiber Optic), Online Security, Tech Support, etc.
- **Account Info:** Tenure, Contract type, Payment method, and Monthly/Total charges.

## 🚀 How to Run Locally
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Customer-Churn-Prediction.git](https://github.com/Ammaratahir252/Customer-Churn-Prediction.git)
   cd Customer-Churn-Prediction
   
Model Performance

The system utilizes a Random Forest Classifier achieving an accuracy of approximately ~80%.

Top Predictors: Tenure, Contract Type, and Monthly Charges.

Validation: The app includes a "Technical Report" expander to view the model's performance on the fly.

🛠️ Technologies Used

Frontend: Streamlit (Custom CSS injected for Dark Mode)

Data Handling: Pandas, NumPy

Machine Learning: Scikit-Learn (RandomForest, LabelEncoding)

Visualization: Seaborn, Matplotlib
