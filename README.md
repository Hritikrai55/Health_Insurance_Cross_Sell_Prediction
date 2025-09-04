# Health Insurance Cross Sell Prediction

## 🎯 Project Overview

This project aims to build a machine learning model that predicts whether existing health insurance policyholders will be interested in vehicle insurance offered by the same company. The solution helps insurance companies optimize their cross-selling strategies and improve conversion rates.

## 💼 Business Problem

An insurance company that provides health insurance to its customers needs to predict whether policyholders from the past year will also be interested in vehicle insurance. This prediction model will help the company:

- Improve cross-selling strategies
- Increase conversion rates
- Reduce customer acquisition costs
- Optimize marketing campaigns
- Enhance customer targeting

## 📊 Dataset Information

- **Size:** 381,109 rows × 12 columns
- **Data Quality:** No missing values or duplicates
- **Target Variable:** Response (binary: interested/not interested in vehicle insurance)

### Features Include:
- **Demographics:** Gender, Age, Region Code
- **Vehicle Information:** Vehicle Age, Vehicle Damage history
- **Policy Details:** Annual Premium, Policy Sales Channel, Vintage
- **Insurance History:** Driving License status, Previously Insured status

## 🚀 Installation & Setup

### Prerequisites
- Python 3.7 or higher
- Jupyter Notebook or Google Colab

### Required Libraries
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn xgboost imbalanced-learn mlxtend
```

### Installation Steps
1. Clone this repository:
```bash
git clone https://github.com/Hritikrai55/Health_Insurance_Cross_Sell_Prediction.git
cd Health_Insurance_Cross_Sell_Prediction
```

2. Open the Jupyter notebook:
```bash
jupyter notebook Health_Insurance_Cross_Sell_Prediction.ipynb
```

## 🔧 Usage

1. **Data Loading:** Load the training dataset (`TRAIN-HEALTH INSURANCE CROSS SELL PREDICTION.csv`)
2. **Run Analysis:** Execute all cells in the Jupyter notebook sequentially
3. **Model Training:** The notebook will automatically train and evaluate multiple models
4. **Results:** View model performance metrics and predictions

### Key Sections in the Notebook:
- Data Exploration and Analysis
- Data Preprocessing and Feature Engineering
- Hypothesis Testing
- Model Training and Evaluation
- Feature Importance Analysis

## 📁 Project Structure

```
Health_Insurance_Cross_Sell_Prediction/
├── Health_Insurance_Cross_Sell_Prediction.ipynb    # Main analysis notebook
├── TRAIN-HEALTH INSURANCE CROSS SELL PREDICTION.csv # Dataset
├── README.md                                        # Project documentation
```

## 🔬 Methodology

### 1. Data Analysis & Preprocessing
- Comprehensive Exploratory Data Analysis (EDA)
- Univariate and bivariate analysis
- Correlation analysis and visualization
- Outlier detection and treatment using capping method
- Feature engineering and encoding

### 2. Data Balancing
- **Problem:** Imbalanced dataset (46,710 positive vs 334,399 negative samples)
- **Solution:** SMOTE (Synthetic Minority Oversampling Technique)

### 3. Feature Engineering
- One-hot encoding for categorical variables
- MinMax scaling for numerical features
- Feature selection based on importance
- Removal of non-informative features

### 4. Model Development
- Train-test split (80-20 ratio)
- Multiple algorithm comparison
- Hyperparameter tuning using GridSearchCV
- Cross-validation for model validation

## 📈 Model Performance

| Model | F1-Score | Key Features |
|-------|----------|--------------|
| **Random Forest** | **87.32%** | Best performing model with hyperparameter tuning |
| XGBoost | 84% | Gradient boosting with GridSearchCV |
| Logistic Regression | 82% | Baseline linear model |

### Best Model: Random Forest Classifier
- **F1-Score:** 87%
- **Status:** No overfitting observed
- **Hyperparameter Tuning:** GridSearchCV applied

## 💡 Key Insights

### Feature Importance (Top 5):
1. **Previously_Insured_yes** (0.20) - Most important predictor
2. **Vintage** (0.175) - Customer relationship duration
3. **Annual_Premium** (0.15) - Premium amount significance
4. **Age** (0.135) - Customer age factor
5. **Vehicle_Damage_yes** (0.125) - Vehicle damage history

### Hypothesis Testing Results:
1. ✅ Average annual premium > ₹20,000
2. ✅ Average customer age > 30 years  
3. ✅ Standard deviation of annual premium = ₹10,000

### Business Recommendations:
- Target customers who were previously insured
- Focus on customers with vehicle damage history
- Consider age and premium amount in targeting strategies
- Implement incentives for cross-selling vehicle insurance

## 🛠 Technologies Used

- **Programming:** Python
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, XGBoost
- **Data Balancing:** Imbalanced-learn (SMOTE)
- **Statistical Analysis:** SciPy
- **Model Evaluation:** MLxtend

## 👨‍💻 Author

**Hritik Rai**

- LinkedIn: [https://www.linkedin.com/in/hritik-rai-/]
- Certification Link: [https://verified.sertifier.com/en/verify/67232886872076/]

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

*This project demonstrates end-to-end machine learning workflow for binary classification with comprehensive data analysis, feature engineering, and model comparison.*
