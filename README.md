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

## How to Use

### Prerequisites

- Python 3.12+ (recommended)
- Conda (recommended) or pip
- Jupyter Notebook

### Ensure dataset is present
   - Download `onlinefraud.csv` from this link: https://www.kaggle.com/datasets/jainilcoder/online-payment-fraud-detection/data
   - Place `onlinefraud.csv` in the project directory
   - Dataset should be ~470MB with 6.3M+ rows

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

