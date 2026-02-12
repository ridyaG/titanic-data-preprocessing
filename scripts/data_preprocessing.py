"""
Data Cleaning & Preprocessing Task
===================================
This script demonstrates comprehensive data cleaning and preprocessing techniques
using the Titanic dataset for Machine Learning preparation.

Author: Ridya Gupta
Task: Data Cleaning & Preprocessing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("DATA CLEANING & PREPROCESSING PIPELINE")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD AND EXPLORE THE DATASET
# ============================================================================
print("\n[STEP 1] Loading and Exploring Dataset...")
print("-" * 80)

# For this example, I'll create a sample Titanic dataset
# In practice, you would load it using: df = pd.read_csv('titanic.csv')
np.random.seed(42)

# Creating sample Titanic data
n_samples = 891
data = {
    'PassengerId': range(1, n_samples + 1),
    'Survived': np.random.choice([0, 1], n_samples, p=[0.62, 0.38]),
    'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55]),
    'Name': [f'Passenger_{i}' for i in range(1, n_samples + 1)],
    'Sex': np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35]),
    'Age': np.random.normal(29.7, 14.5, n_samples),
    'SibSp': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.68, 0.23, 0.05, 0.03, 0.01]),
    'Parch': np.random.choice([0, 1, 2, 3], n_samples, p=[0.76, 0.13, 0.08, 0.03]),
    'Ticket': [f'T{i}' for i in range(1, n_samples + 1)],
    'Fare': np.random.exponential(32, n_samples),
    'Cabin': [f'C{i}' if np.random.random() > 0.77 else np.nan for i in range(n_samples)],
    'Embarked': np.random.choice(['S', 'C', 'Q'], n_samples, p=[0.72, 0.19, 0.09])
}

# Add some missing values to simulate real data
data['Age'][np.random.choice(n_samples, 177, replace=False)] = np.nan
data['Embarked'][np.random.choice(n_samples, 2, replace=False)] = np.nan

df = pd.DataFrame(data)

# Basic dataset information
print("\n1.1 Dataset Shape:", df.shape)
print(f"    Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n1.2 First 5 Rows:")
print(df.head())

print("\n1.3 Dataset Info:")
print(df.info())

print("\n1.4 Statistical Summary:")
print(df.describe())

print("\n1.5 Missing Values:")
missing_data = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
})
missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
print(missing_data)

print("\n1.6 Data Types:")
print(df.dtypes)

# ============================================================================
# STEP 2: HANDLE MISSING VALUES
# ============================================================================
print("\n\n[STEP 2] Handling Missing Values...")
print("-" * 80)

# Create a copy for preprocessing
df_processed = df.copy()

# 2.1 Handle Age (numerical) - Use median imputation
print("\n2.1 Age Column:")
print(f"    Missing before: {df_processed['Age'].isnull().sum()}")
age_median = df_processed['Age'].median()
df_processed['Age'].fillna(age_median, inplace=True)
print(f"    Filled with median: {age_median:.2f}")
print(f"    Missing after: {df_processed['Age'].isnull().sum()}")

# 2.2 Handle Embarked (categorical) - Use mode imputation
print("\n2.2 Embarked Column:")
print(f"    Missing before: {df_processed['Embarked'].isnull().sum()}")
embarked_mode = df_processed['Embarked'].mode()[0]
df_processed['Embarked'].fillna(embarked_mode, inplace=True)
print(f"    Filled with mode: {embarked_mode}")
print(f"    Missing after: {df_processed['Embarked'].isnull().sum()}")

# 2.3 Handle Cabin - Drop due to high missing percentage
print("\n2.3 Cabin Column:")
print(f"    Missing percentage: {(df_processed['Cabin'].isnull().sum() / len(df_processed) * 100):.2f}%")
print("    Decision: Dropping column due to >70% missing values")
df_processed.drop('Cabin', axis=1, inplace=True)

# 2.4 Verify no missing values remain
print("\n2.4 Verification - Total Missing Values:", df_processed.isnull().sum().sum())

# ============================================================================
# STEP 3: ENCODE CATEGORICAL VARIABLES
# ============================================================================
print("\n\n[STEP 3] Encoding Categorical Variables...")
print("-" * 80)

# 3.1 Identify categorical columns
categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('Name')  # Remove Name as it's not useful for ML
categorical_cols.remove('Ticket')  # Remove Ticket as it's not useful for ML
print(f"\n3.1 Categorical columns to encode: {categorical_cols}")

# 3.2 Label Encoding for binary categorical variable (Sex)
print("\n3.2 Label Encoding for 'Sex':")
label_encoder = LabelEncoder()
df_processed['Sex_Encoded'] = label_encoder.fit_transform(df_processed['Sex'])
print(f"    Mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
print(df_processed[['Sex', 'Sex_Encoded']].head())

# 3.3 One-Hot Encoding for multi-class categorical variable (Embarked)
print("\n3.3 One-Hot Encoding for 'Embarked':")
embarked_dummies = pd.get_dummies(df_processed['Embarked'], prefix='Embarked', drop_first=True)
df_processed = pd.concat([df_processed, embarked_dummies], axis=1)
print(f"    New columns created: {embarked_dummies.columns.tolist()}")
print(df_processed[['Embarked'] + embarked_dummies.columns.tolist()].head())

# Drop original categorical columns (except for reference)
df_processed.drop(['Sex', 'Embarked'], axis=1, inplace=True)

# ============================================================================
# STEP 4: FEATURE SCALING (NORMALIZATION & STANDARDIZATION)
# ============================================================================
print("\n\n[STEP 4] Feature Scaling...")
print("-" * 80)

# Select numerical features for scaling
numerical_features = ['Age', 'Fare', 'SibSp', 'Parch']

print("\n4.1 Original Numerical Features Statistics:")
print(df_processed[numerical_features].describe())

# 4.2 Standardization (Z-score normalization)
print("\n4.2 Standardization (Mean=0, Std=1):")
scaler_standard = StandardScaler()
df_processed[['Age_Standardized', 'Fare_Standardized']] = scaler_standard.fit_transform(
    df_processed[['Age', 'Fare']]
)
print("    Applied to: Age, Fare")
print(df_processed[['Age', 'Age_Standardized', 'Fare', 'Fare_Standardized']].head())

# 4.3 Normalization (Min-Max scaling)
print("\n4.3 Normalization (Range 0-1):")
scaler_minmax = MinMaxScaler()
df_processed[['Age_Normalized', 'Fare_Normalized']] = scaler_minmax.fit_transform(
    df_processed[['Age', 'Fare']]
)
print("    Applied to: Age, Fare")
print(df_processed[['Age', 'Age_Normalized', 'Fare', 'Fare_Normalized']].head())

print("\n4.4 Scaled Features Statistics:")
print(df_processed[['Age_Standardized', 'Fare_Standardized', 'Age_Normalized', 'Fare_Normalized']].describe())

# ============================================================================
# STEP 5: OUTLIER DETECTION AND REMOVAL
# ============================================================================
print("\n\n[STEP 5] Outlier Detection and Removal...")
print("-" * 80)

# 5.1 Detect outliers using IQR method
def detect_outliers_iqr(data, column):
    """Detect outliers using Interquartile Range (IQR) method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

print("\n5.1 Outlier Detection for 'Fare':")
outliers_fare, lower_fare, upper_fare = detect_outliers_iqr(df_processed, 'Fare')
print(f"    Lower Bound: {lower_fare:.2f}")
print(f"    Upper Bound: {upper_fare:.2f}")
print(f"    Number of outliers: {len(outliers_fare)} ({len(outliers_fare)/len(df_processed)*100:.2f}%)")

print("\n5.2 Outlier Detection for 'Age':")
outliers_age, lower_age, upper_age = detect_outliers_iqr(df_processed, 'Age')
print(f"    Lower Bound: {lower_age:.2f}")
print(f"    Upper Bound: {upper_age:.2f}")
print(f"    Number of outliers: {len(outliers_age)} ({len(outliers_age)/len(df_processed)*100:.2f}%)")

# 5.3 Remove outliers
print("\n5.3 Removing Outliers:")
df_no_outliers = df_processed[
    (df_processed['Fare'] >= lower_fare) & 
    (df_processed['Fare'] <= upper_fare) &
    (df_processed['Age'] >= lower_age) & 
    (df_processed['Age'] <= upper_age)
].copy()
print(f"    Rows before: {len(df_processed)}")
print(f"    Rows after: {len(df_no_outliers)}")
print(f"    Rows removed: {len(df_processed) - len(df_no_outliers)}")

# ============================================================================
# STEP 6: CREATE VISUALIZATIONS
# ============================================================================
print("\n\n[STEP 6] Creating Visualizations...")
print("-" * 80)

# Create visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 6.1 Missing values heatmap (original data)
ax1 = axes[0, 0]
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis', ax=ax1)
ax1.set_title('Missing Values Heatmap (Original Data)', fontsize=12, fontweight='bold')

# 6.2 Age distribution before and after imputation
ax2 = axes[0, 1]
ax2.hist(df['Age'].dropna(), bins=30, alpha=0.5, label='Original', color='blue', edgecolor='black')
ax2.hist(df_processed['Age'], bins=30, alpha=0.5, label='After Imputation', color='green', edgecolor='black')
ax2.set_xlabel('Age')
ax2.set_ylabel('Frequency')
ax2.set_title('Age Distribution: Before vs After Imputation', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 6.3 Boxplot for outlier detection (Fare)
ax3 = axes[0, 2]
ax3.boxplot([df_processed['Fare'], df_no_outliers['Fare']], 
            labels=['With Outliers', 'Without Outliers'])
ax3.set_ylabel('Fare')
ax3.set_title('Fare: Outlier Detection using Boxplot', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 6.4 Correlation heatmap
ax4 = axes[1, 0]
correlation_cols = ['Survived', 'Pclass', 'Sex_Encoded', 'Age', 'SibSp', 'Parch', 'Fare']
correlation_matrix = df_processed[correlation_cols].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax4, cbar_kws={'shrink': 0.8})
ax4.set_title('Feature Correlation Heatmap', fontsize=12, fontweight='bold')

# 6.5 Standardization vs Normalization comparison
ax5 = axes[1, 1]
ax5.scatter(df_processed['Age_Standardized'], df_processed['Age_Normalized'], alpha=0.5)
ax5.set_xlabel('Age (Standardized)')
ax5.set_ylabel('Age (Normalized)')
ax5.set_title('Standardization vs Normalization', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)

# 6.6 Class distribution
ax6 = axes[1, 2]
survival_counts = df_processed['Survived'].value_counts()
ax6.bar(['Not Survived', 'Survived'], survival_counts.values, color=['#ff6b6b', '#51cf66'], edgecolor='black')
ax6.set_ylabel('Count')
ax6.set_title('Survival Distribution', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(survival_counts.values):
    ax6.text(i, v + 10, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('/home/claude/preprocessing_visualizations.png', dpi=300, bbox_inches='tight')
print("    ✓ Visualizations saved to 'preprocessing_visualizations.png'")

# ============================================================================
# STEP 7: PREPARE FINAL DATASET
# ============================================================================
print("\n\n[STEP 7] Preparing Final Dataset...")
print("-" * 80)

# Drop unnecessary columns
final_features = [
    'Survived', 'Pclass', 'Sex_Encoded', 'Age', 'SibSp', 'Parch', 'Fare',
    'Embarked_Q', 'Embarked_S', 'Age_Standardized', 'Fare_Standardized'
]

df_final = df_no_outliers[final_features].copy()

print("\n7.1 Final Dataset Shape:", df_final.shape)
print("\n7.2 Final Features:")
print(df_final.columns.tolist())

print("\n7.3 Final Dataset Preview:")
print(df_final.head(10))

print("\n7.4 Final Dataset Statistics:")
print(df_final.describe())

# Save the cleaned dataset
df_final.to_csv('/home/claude/titanic_cleaned.csv', index=False)
print("\n7.5 ✓ Cleaned dataset saved to 'titanic_cleaned.csv'")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n\n" + "=" * 80)
print("PREPROCESSING SUMMARY")
print("=" * 80)
print(f"""
Original Dataset:
  - Rows: {df.shape[0]}
  - Columns: {df.shape[1]}
  - Missing Values: {df.isnull().sum().sum()}

Final Dataset:
  - Rows: {df_final.shape[0]} ({(df_final.shape[0]/df.shape[0]*100):.1f}% retained)
  - Columns: {df_final.shape[1]}
  - Missing Values: {df_final.isnull().sum().sum()}

Steps Completed:
  ✓ Data exploration and profiling
  ✓ Missing value handling (imputation)
  ✓ Categorical encoding (Label & One-Hot)
  ✓ Feature scaling (Standardization & Normalization)
  ✓ Outlier detection and removal
  ✓ Data visualization
  ✓ Final dataset preparation

Output Files:
  1. titanic_cleaned.csv - Cleaned dataset
  2. preprocessing_visualizations.png - Visual analysis
  3. data_preprocessing.py - Complete pipeline code
""")

print("=" * 80)
print("PREPROCESSING COMPLETED SUCCESSFULLY!")
print("=" * 80)
