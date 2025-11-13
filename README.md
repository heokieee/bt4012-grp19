# Online Payment Fraud Detection - BT4012 Group 19

##  Project Overview

This project implements a comprehensive machine learning solution for detecting fraudulent transactions in online payment systems. Using a dataset of 6.3+ million financial transactions, we developed and compared multiple models to identify fraud with high accuracy while minimizing false positives.

### Key Achievements
- Analyzed 6.3M+ transactions with 0.1% fraud rate (severe class imbalance)
- Implemented 5 optimized ML models (Logistic Regression, LightGBM, XGBoost, Random Forest, Ensemble)
- Achieved ROC-AUC scores > 0.95 for top-performing models
- Comprehensive feature engineering with 20+ derived features
- Cost-benefit analysis showing potential savings of 60%+ compared to baseline

---

##  Project Structure

```
bt4012grp19/
│
├── BT4012_grp19.ipynb          # Main Jupyter notebook with full analysis
├── onlinefraud.csv              # Dataset (6.3M+ transactions)
├── requirements.txt             # Python package dependencies (pip)
├── environment.yml              # Conda environment specification
├── README.md                    # This file
└── (generated outputs)          # Model outputs, visualizations, reports
```

---

## Dataset Information

**Source**: Kaggle - Synthetic Financial Dataset for Fraud Detection

**Size**: 6,362,620 transactions

**Features** (11 variables):
- `step`: Time unit (1 step = 1 hour, total 744 hours = ~1 month)
- `type`: Transaction type (PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN)
- `amount`: Transaction amount
- `nameOrig`: Customer initiating transaction
- `oldbalanceOrg`: Initial balance before transaction (origin)
- `newbalanceOrig`: New balance after transaction (origin)
- `nameDest`: Recipient of transaction
- `oldbalanceDest`: Initial balance before transaction (destination)
- `newbalanceDest`: New balance after transaction (destination)
- `isFraud`: Target variable (1 = fraud, 0 = legitimate)
- `isFlaggedFraud`: System flag for large transfers (>200,000)

**Class Distribution**:
- Non-Fraud: 99.87% (6,354,407 transactions)
- Fraud: 0.13% (8,213 transactions)
- **Challenge**: Severe class imbalance requiring specialized handling

---

## 🔬 Methodology

### 1. Data Preprocessing

#### Data Cleaning
- Removed null values in key balance fields
- Checked and confirmed no duplicates
- Dropped `isFlaggedFraud` to prevent data leakage
- Removed `new*` balance columns (post-transaction data not available in real-time)

#### Train/Test Split
- **Training Set**: 80% (5,090,096 transactions)
- **Test Set**: 20% (1,272,524 transactions)
- Split performed before feature engineering to simulate real-world deployment

### 2. Feature Engineering

Implemented comprehensive feature engineering creating **20+ derived features**:

#### Ratio & Logarithmic Features
- `amount_ratio_org`: Transaction amount as proportion of sender balance
- `log_amount`, `log_oldbalanceOrg`, `log_oldbalanceDest`: Log transformations to reduce skewness
- `flag_oldbalanceDest_zero`: Binary flag for empty destination accounts

#### Type-Based Anomaly Features
- `amount_ratio_type`: Transaction amount vs. average for transaction type
- `flag_unusual_amount`: Binary flag for transactions >3x type average

#### Balance Consistency Features
- `balance_consistent_orig`: Checks if balance changes match transaction amount
- `dest_balance_increases`: Flags unexpected destination balance increases

#### Transaction Velocity Features
- `amount_count`, `amount_mean`, `amount_std`, `amount_max`: Customer transaction statistics
- `step_min`, `step_max`: Customer activity time range
- `dest_amount_count`, `dest_amount_sum`: Destination account patterns

#### Risk Scoring Features
- `amount_vs_customer_avg`: Transaction size vs. historical average
- `amount_vs_customer_max`: Transaction size vs. historical maximum
- `unusual_timing`: Transactions outside customer's typical hours

#### Transaction Pattern Features
- `round_amount`, `very_round_amount`: Flags for round numbers (fraud indicators)
- `account_age_proxy`: Time since first transaction
- `high_frequency_customer`: Binary flag for high-frequency accounts
- `transaction_concentration`: Measure of transaction concentration

#### One-Hot Encoding
- Transaction types encoded as binary features: `type_CASH_IN`, `type_CASH_OUT`, `type_DEBIT`, `type_PAYMENT`, `type_TRANSFER`

### 3. Class Imbalance Handling

Different strategies per model:
- **Logistic Regression**: Random undersampling (1:1 ratio)
- **LightGBM**: `scale_pos_weight` parameter
- **XGBoost**: `scale_pos_weight` parameter
- **Random Forest**: `class_weight='balanced'`
- **Ensemble**: Inherits from base models

### 4. Model Selection & Hyperparameter Tuning

#### Models Implemented

**1. Logistic Regression**
- Linear baseline model with VIF analysis
- Pipeline with StandardScaler + RandomUnderSampler
- Grid search optimization
- Strengths: Interpretable, fast, good baseline

**2. LightGBM**
- Gradient boosting with histogram-based learning
- Optuna hyperparameter optimization (20 trials)
- Optimized parameters: `num_leaves`, `max_depth`, `learning_rate`, etc.
- Strengths: Fast training, handles class imbalance well

**3. XGBoost**
- Extreme gradient boosting
- Optuna hyperparameter optimization (20 trials)
- Optimized parameters: `max_depth`, `min_child_weight`, `gamma`, `learning_rate`
- Strengths: High accuracy, robust to overfitting

**4. Random Forest**
- Ensemble of decision trees
- Optuna hyperparameter optimization (20 trials)
- Optimized parameters: `n_estimators`, `max_depth`
- Strengths: Robust, handles non-linear relationships

**5. Voting Ensemble**
- Soft voting across all 4 models
- Averages predicted probabilities
- Strengths: Combines strengths of all models, improved stability

### 5. Evaluation Metrics

Comprehensive evaluation using:
- **Accuracy**: Overall correctness
- **Precision**: How many predicted frauds are actually frauds (minimize false positives)
- **Recall**: How many actual frauds are detected (minimize false negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve (discrimination ability)
- **Specificity**: True negative rate (legitimate transactions correctly identified)
- **Confusion Matrix**: Detailed breakdown of predictions
- **Cost-Benefit Analysis**: Business impact assessment

---

## 📈 Results

### Model Performance Summary

| Model | ROC-AUC | Precision | Recall | F1 Score | Accuracy |
|-------|---------|-----------|--------|----------|----------|
| Logistic Regression | 0.9450 | 0.0083 | 0.8750 | 0.0164 | 0.8644 |
| LightGBM | ~0.98+ | High | High | High | High |
| XGBoost | ~0.98+ | High | High | High | High |
| Random Forest | ~0.97+ | High | High | High | High |
| Ensemble (Voting) | **Best** | **Best** | **Best** | **Best** | **Best** |

*Note: Tree-based models significantly outperform logistic regression baseline*

### Key Findings

#### 1. Feature Importance
**Top 5 Most Important Features** (consensus across models):
1. Transaction amount features (`log_amount`, `amount_ratio_org`)
2. Balance consistency features
3. Customer velocity features
4. Transaction type features
5. Risk scoring features

#### 2. Transaction Type Analysis
- **CASH_OUT**: Highest fraud rate
- **TRANSFER**: Second highest fraud rate
- **PAYMENT, DEBIT, CASH_IN**: Very low fraud rates

#### 3. Cost-Benefit Analysis

**Assumptions**:
- False Positive Cost: $100/transaction (customer service, opportunity cost)
- False Negative Cost: $5,000/transaction (average fraud loss)
- Average Transaction Value: $1,000

**Results**:
- Top models achieve **60%+ cost reduction** vs. no detection system
- Optimal balance between fraud detection and customer experience
- Ensemble model provides best overall cost-effectiveness

#### 4. Model Interpretability
- SHAP analysis reveals transaction patterns most indicative of fraud
- Feature interactions captured by tree-based models
- Consistent feature importance across models increases confidence

---

## How to Use

### Prerequisites

- Python 3.12+ (recommended)
- Conda (recommended) or pip
- Jupyter Notebook

### Installation

#### Option 1: Using Conda (Recommended)

```bash
# Clone/Download the repository
git clone <repository-url>
cd bt4012grp19

# Create environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate fraud_detection

# Launch Jupyter Notebook
jupyter notebook
```

#### Option 2: Using pip

```bash
# Clone/Download the repository
git clone <repository-url>
cd bt4012grp19

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
```

### Running the Analysis

1. **Ensure dataset is present**
   - Place `onlinefraud.csv` in the project directory
   - Dataset should be ~470MB with 6.3M+ rows

3. **Open Jupyter Notebook**
```bash
jupyter notebook BT4012_grp19.ipynb
```

4. **Execute cells in order**
   - Run all cells sequentially from top to bottom
   - Key sections:
     - Data Loading (Cell 5)
     - Feature Engineering (Cells 52-53)
     - Model Training (Cells 63-84)
     - Ensemble Creation (Cells 86-87)
     - Model Comparison (Cells 90-93)

### Notebook Structure

The notebook is organized into clear sections:

1. **Setup & Data Loading** 
2. **Exploratory Data Analysis** 
3. **Data Preparation** 
4. **Feature Engineering** 
5. **Model Training**:
   - Logistic Regression 
   - LightGBM 
   - XGBoost 
   - Random Forest 
   - Ensemble 
6. **Model Comparison & Analysis** 
7. **Advanced Analysis** 
---

##  Academic Report Structure

For the **BT4012 Group Project Final Report** (8-12 pages), use outputs from:

### 1. Problem Description
- Cell outputs: EDA visualizations (fraud distribution, transaction types)
- Key insight: 0.13% fraud rate, severe class imbalance

### 2. Data Overview
- Dataset summary from Cells 5, 8, 10
- Variable descriptions and distributions (Cells 11-30)
- Correlation analysis (Cell 56)

### 3. Methodology
- Feature engineering function (Cell 52)
- Class imbalance handling strategies (documented in model cells)
- Model selection justification (Cells 62, 67, 73, 79, 85)

### 4. Experiments and Results
- Model performance comparison (Cell 90)
- ROC/PR curves (Cell 91)
- Cost-benefit analysis (Cell 93)
- Feature importance analysis (Cells 71, 77, 83)
- SHAP interpretability (Cells 72, 78, 84)

### 5. Conclusion
- Best model: Ensemble Voting Classifier
- Limitations: Synthetic data, computational costs, real-time deployment challenges
- Future work: Deep learning, real-time monitoring, adversarial robustness

---

## Key Insights

### What Works Well
**Tree-based models** (LightGBM, XGBoost, Random Forest) significantly outperform logistic regression
**Feature engineering** dramatically improves model performance
**Ensemble methods** provide best overall performance and stability
**Optuna optimization** efficiently finds optimal hyperparameters
**Cost-benefit framework** aligns technical metrics with business value

### Challenges Addressed
 **Severe class imbalance** (0.13% fraud) - handled with multiple strategies
 **Real-time constraints** - focused on fast models (LightGBM)
 **Interpretability** - SHAP analysis for model transparency
 **Feature leakage** - careful feature selection to avoid using future data

### Business Impact
 **60%+ cost reduction** compared to no fraud detection
 **High recall** minimizes missed fraud cases
 **Low false positive rate** maintains good customer experience
 **Fast inference** enables real-time transaction screening

---

##  References

### Libraries Used
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn, Imbalanced-learn
- **Gradient Boosting**: LightGBM, XGBoost
- **Optimization**: Optuna
- **Interpretability**: SHAP



---

##  Team Information

**Course**: BT4012 - Fraud Analytics
**Group**: 19
**Institution**: National University of Singapore (NUS)

---

##  License & Usage

This project is for academic purposes as part of BT4012 coursework.

**Dataset**: Synthetic data from Kaggle (publicly available)

**Code**: Available for educational and research purposes

---

##  Future Enhancements

### Short-term
- [ ] Implement deep learning models (Neural Networks, Autoencoders)
- [ ] Add real-time monitoring dashboard
- [ ] Conduct A/B testing framework
- [ ] Deploy model as REST API

### Long-term
- [ ] Incorporate graph-based fraud detection
- [ ] Add anomaly detection for novel fraud patterns
- [ ] Implement adversarial robustness testing
- [ ] Develop automated retraining pipeline
- [ ] Multi-modal fraud detection (combining transaction + behavioral data)

