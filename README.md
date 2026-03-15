# DeepCSAT – Ecommerce Customer Satisfaction Score Prediction

Predicting customer satisfaction is crucial for improving service quality in modern e-commerce platforms. Customer interactions with support teams generate valuable data that can be used to understand customer experience and predict satisfaction levels. This project builds a machine learning and deep learning based system to predict **Customer Satisfaction (CSAT) scores** using customer support interaction data.

The project analyzes operational features such as issue categories, product information, response time, handling time, and customer remarks to identify patterns that influence customer satisfaction. By leveraging predictive analytics, organizations can proactively improve customer support strategies and enhance overall user experience.

---

## 🚀 Live Application

**Streamlit App:**  
https://deepcsat-ecommerce-customer-satisfaction-score-prediction.streamlit.app/

The deployed application allows users to input customer service parameters and obtain predicted CSAT scores.

---

## 📊 Project Workflow

### 1. Data Collection
Customer support interaction dataset containing operational and customer-related features.

### 2. Data Cleaning
- Handling missing values  
- Removing irrelevant columns  
- Data type conversions  

### 3. Exploratory Data Analysis (EDA)
- Distribution analysis  
- Category-based analysis  
- Correlation analysis  
- Feature relationship visualization  

### 4. Feature Engineering
- Time-based feature creation  
- Numerical transformations  
- Categorical encoding  
- Feature scaling  

### 5. Model Development
The following models were implemented and compared:

- Logistic Regression  
- Random Forest  
- XGBoost  
- Artificial Neural Network (ANN)  

### 6. Model Evaluation
Models were evaluated using the following metrics:

- Accuracy  
- Precision  
- Recall  
- F1 Score  

### 7. Model Deployment
- Best performing model deployed using **Streamlit**  
- Interactive web application for CSAT prediction  

---

## 🛠️ Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **XGBoost**
- **TensorFlow / Keras**
- **Streamlit**
- **Joblib**
- **Matplotlib**
- **Seaborn**

---

## ✨ Key Features

- Predict customer satisfaction scores using support interaction data
- End-to-end machine learning pipeline
- Feature engineering and preprocessing workflow
- Comparison of machine learning and deep learning models
- Interactive Streamlit web application for real-time predictions

---

## 📁 Project Structure
```
DeepCSAT-Ecommerce-Customer-Satisfaction-Score-Prediction
│
├── app.py
├── dataset.csv
├── deepcsat_model.joblib
├── scaler.joblib
├── imputer.joblib
├── feature_columns.joblib
├── notebook.ipynb
├── requirements.txt
└── README.md
```
