# 🚢 Titanic Data Preprocessing & Machine Learning Pipeline

## 📋 Complete Project Package

This package contains everything you need for the **Data Cleaning & Preprocessing** task, including code, datasets, visualizations, documentation, and upload guides.

---

## 📦 Package Contents

```
project_package/
├── README.md                           ← You are here!
├── requirements.txt                    ← All dependencies
│
├── data/                               ← Datasets
│   ├── Titanic-Dataset.csv            ← Original raw dataset (891 rows, 12 columns)
│   ├── titanic_cleaned.csv            ← Cleaned dataset from uploaded files
│   └── titanic_cleaned_output.csv     ← Cleaned dataset from the script
│
├── notebooks/                          ← Jupyter Notebooks
│   └── Titanic_EDA_Preprocessing.ipynb ← uploaded notebook
│
├── scripts/                            ← Python Scripts
│   └── data_preprocessing.py          ← Complete preprocessing pipeline
│
├── visualizations/                     ← All visualizations
│   ├── preprocessing_analysis.png     ← 6-panel analysis dashboard
│   ├── original_visualizations.png    ← Uploaded visualizations
│   └── upload_workflow_diagram.png    ← GitHub/Kaggle workflow diagram
│
├── docs/                               ← Documentation
│   └── [Will contain additional docs]
│
└── guides/                             ← Upload guides
    ├── UPLOAD_GUIDE.md                ← Complete upload guide (GitHub & Kaggle)
    ├── QUICK_REFERENCE.txt            ← Quick reference cheat sheet
    └── github_upload.sh               ← Automated upload script
```

---

## 🎯 What This Project Does

This project demonstrates a **complete data preprocessing pipeline** for machine learning, covering:

### 1. **Data Exploration & Understanding**
- Dataset inspection (shape, types, statistics)
- Missing value analysis
- Data type verification
- Initial data quality assessment

### 2. **Data Cleaning**
- **Missing Value Handling:**
  - Numerical features (Age): Median imputation
  - Categorical features (Embarked): Mode imputation
  - High missing percentage (Cabin): Column removal
- **Duplicate Detection:** Checked and removed if present
- **Data Type Conversion:** Ensured correct types

### 3. **Feature Engineering**
- **Categorical Encoding:**
  - Label Encoding for binary variables (Sex: male=1, female=0)
  - One-Hot Encoding for multi-class variables (Embarked: C, Q, S)
- **Feature Creation:** Created encoded versions for ML models

### 4. **Feature Scaling**
- **Standardization (Z-score):** Mean=0, Std=1
  - Applied to Age and Fare
  - Best for: Linear models, SVM, Neural Networks
- **Normalization (Min-Max):** Range [0, 1]
  - Applied to Age and Fare
  - Best for: Neural Networks with sigmoid/tanh

### 5. **Outlier Detection & Removal**
- **Method Used:** IQR (Interquartile Range)
  - Formula: Lower = Q1 - 1.5×IQR, Upper = Q3 + 1.5×IQR
  - Detected outliers in Fare and Age
  - Removed 212 outliers (23.8% of data)

### 6. **Data Visualization**
Created 6 comprehensive visualizations:
1. Missing values heatmap
2. Age distribution (before vs after imputation)
3. Fare outlier detection boxplot
4. Feature correlation heatmap
5. Standardization vs Normalization comparison
6. Survival distribution

---

## 🔧 How to Use This Package

### Prerequisites

Install required libraries:
```bash
pip install -r requirements.txt
```

Required packages:
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- scikit-learn >= 1.3.0
- scipy >= 1.10.0

---

### Method 1: Run the Python Script

```bash
# Navigate to scripts folder
cd scripts/

# Run the preprocessing pipeline
python data_preprocessing.py
```

**Output:**
- `titanic_cleaned.csv` - Cleaned dataset
- `preprocessing_visualizations.png` - Visualization dashboard
- Console output with detailed statistics

---

### Method 2: Use the Jupyter Notebook

```bash
# Navigate to notebooks folder
cd notebooks/

# Launch Jupyter
jupyter notebook Titanic_EDA_Preprocessing.ipynb
```

Run all cells to see:
- Step-by-step preprocessing
- Interactive analysis
- Detailed explanations

---

## 📊 Dataset Information

### Original Dataset: `Titanic-Dataset.csv`

**Description:** Famous Titanic passenger survival dataset

**Features (12 columns):**
- `PassengerId`: Unique passenger ID
- `Survived`: Survival status (0=No, 1=Yes) [TARGET]
- `Pclass`: Ticket class (1=1st, 2=2nd, 3=3rd)
- `Name`: Passenger name
- `Sex`: Gender (male/female)
- `Age`: Age in years
- `SibSp`: Number of siblings/spouses aboard
- `Parch`: Number of parents/children aboard
- `Ticket`: Ticket number
- `Fare`: Passenger fare
- `Cabin`: Cabin number
- `Embarked`: Port of embarkation (C=Cherbourg, Q=Queenstown, S=Southampton)

**Statistics:**
- Total Passengers: 891
- Missing Values:
  - Age: 177 (19.87%)
  - Cabin: 682 (76.54%)
  - Embarked: 2 (0.22%)
- Survival Rate: 38.4%

---

### Cleaned Dataset: `titanic_cleaned.csv`

**Features (11 columns):**
- `Survived`: Target variable (0/1)
- `Pclass`: Ticket class (1/2/3)
- `Sex_Encoded`: Gender encoded (0=female, 1=male)
- `Age`: Age (missing values filled)
- `SibSp`: Siblings/spouses count
- `Parch`: Parents/children count
- `Fare`: Passenger fare
- `Embarked_Q`: Embarked at Queenstown (0/1)
- `Embarked_S`: Embarked at Southampton (0/1)
- `Age_Standardized`: Age scaled (z-score)
- `Fare_Standardized`: Fare scaled (z-score)

**Statistics:**
- Total Rows: 679 (76.2% retained after outlier removal)
- Missing Values: 0 (all handled!)

---

## 📈 Results Summary

| Metric | Original | Cleaned |
|--------|----------|---------|
| **Rows** | 891 | 679 |
| **Columns** | 12 | 11 |
| **Missing Values** | 861 | 0 |
| **Categorical Features** | 5 | 0 |
| **Numerical Features** | 7 | 11 |
| **Outliers** | 212 | 0 |
| **Data Quality** | Raw | ML-Ready|

**Key Achievements:**
- ✅ 100% missing values handled
- ✅ All categorical features encoded
- ✅ Features scaled for ML algorithms
- ✅ Outliers detected and removed
- ✅ Data distribution preserved
- ✅ 76.2% data retention

---

## 📊 Visualizations Explained

### 1. Missing Values Heatmap
Shows where data is missing in the original dataset. Yellow indicates missing values.

**Key Insights:**
- Cabin has 76.54% missing (dropped)
- Age has 19.87% missing (imputed)
- Most features are complete

### 2. Age Distribution Comparison
Overlapping histograms showing Age before and after imputation.

**Key Insights:**
- Original distribution (blue) has gaps
- After imputation (green) fills gaps with median
- Overall distribution shape preserved

### 3. Fare Outlier Detection Boxplot
Side-by-side boxplots showing Fare with and without outliers.

**Key Insights:**
- Original data has extreme outliers (>200)
- After removal, data is more normally distributed
- 41 outliers removed from Fare

### 4. Feature Correlation Heatmap
Shows correlations between all numerical features.

**Key Insights:**
- Pclass and Fare are negatively correlated (-0.55)
- Sex and Survived show correlation (0.54)
- Age has weak correlations with most features

### 5. Standardization vs Normalization
Scatter plot comparing two scaling methods on Age.

**Key Insights:**
- Both methods maintain data relationships
- Standardization: roughly -3 to +3 range
- Normalization: exactly 0 to 1 range

### 6. Survival Distribution
Bar chart showing number of survivors vs non-survivors.

**Key Insights:**
- More passengers died (558) than survived (333)
- 62.7% mortality rate
- Class imbalance to consider for ML

---
