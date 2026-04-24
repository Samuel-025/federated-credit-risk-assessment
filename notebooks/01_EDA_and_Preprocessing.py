"""
Notebook 01: Exploratory Data Analysis and Preprocessing
Project: Federated Learning for Credit Risk Assessment
Author: [Your Name]
Date: February 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*70)
print("PHASE 1: DATA PREPARATION - EDA AND PREPROCESSING")
print("="*70)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[STEP 1] Loading Dataset...")
df = pd.read_csv('../data/raw/credit_data.csv')

print(f"✓ Dataset loaded successfully")
print(f"  Shape: {df.shape}")
print(f"  Features: {df.shape[1]-2} (excluding customer_id and target)")
print(f"  Samples: {df.shape[0]}")

# ============================================================================
# STEP 2: BASIC DATA EXPLORATION
# ============================================================================
print("\n[STEP 2] Basic Data Exploration")
print("-" * 70)

print("\nDataset Info:")
print(df.info())

print("\nFirst 10 rows:")
print(df.head(10))

print("\nStatistical Summary:")
print(df.describe())

print("\nTarget Variable Distribution:")
print(df['credit_risk'].value_counts())
print(f"\nClass Balance:")
print(f"  Good Credit (1): {(df['credit_risk']==1).sum()} ({(df['credit_risk']==1).mean()*100:.1f}%)")
print(f"  Bad Credit (0): {(df['credit_risk']==0).sum()} ({(df['credit_risk']==0).mean()*100:.1f}%)")

# Check for missing values
print("\nMissing Values:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("  ✓ No missing values found")
else:
    print(missing[missing > 0])

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicate Rows: {duplicates}")

# ============================================================================
# STEP 3: FEATURE ANALYSIS
# ============================================================================
print("\n[STEP 3] Feature Analysis")
print("-" * 70)

# Separate numerical and categorical features
numerical_features = ['duration_months', 'credit_amount', 'installment_rate', 
                     'residence_since', 'age', 'existing_credits', 'num_dependents']

categorical_features = ['checking_status', 'credit_history', 'purpose', 
                       'savings_status', 'employment', 'personal_status',
                       'other_parties', 'property_magnitude', 'other_payment_plans',
                       'housing', 'job', 'telephone', 'foreign_worker']

print(f"\nNumerical Features ({len(numerical_features)}):")
for feat in numerical_features:
    print(f"  - {feat}: min={df[feat].min()}, max={df[feat].max()}, mean={df[feat].mean():.2f}")

print(f"\nCategorical Features ({len(categorical_features)}):")
for feat in categorical_features:
    unique_vals = df[feat].nunique()
    print(f"  - {feat}: {unique_vals} unique values")

# ============================================================================
# STEP 4: CORRELATION ANALYSIS
# ============================================================================
print("\n[STEP 4] Correlation Analysis")
print("-" * 70)

# Correlation with target
correlations = df[numerical_features + ['credit_risk']].corr()['credit_risk'].sort_values(ascending=False)
print("\nCorrelation with Credit Risk (Target):")
print(correlations)

# Top correlated features
print("\nTop 5 Positively Correlated Features:")
print(correlations[1:6])  # Exclude target itself

print("\nTop 5 Negatively Correlated Features:")
print(correlations[-5:])

# ============================================================================
# STEP 5: VISUALIZATIONS
# ============================================================================
print("\n[STEP 5] Creating Visualizations...")
print("-" * 70)

# Create results directory if not exists
import os
os.makedirs('../results/eda_plots', exist_ok=True)

# 1. Target Distribution
plt.figure(figsize=(8, 5))
df['credit_risk'].value_counts().plot(kind='bar', color=['#e74c3c', '#2ecc71'])
plt.title('Credit Risk Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Credit Risk (0=Bad, 1=Good)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('../results/eda_plots/01_target_distribution.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: 01_target_distribution.png")
plt.close()

# 2. Age Distribution by Credit Risk
plt.figure(figsize=(10, 5))
df.boxplot(column='age', by='credit_risk', grid=False)
plt.suptitle('')
plt.title('Age Distribution by Credit Risk', fontsize=14, fontweight='bold')
plt.xlabel('Credit Risk (0=Bad, 1=Good)', fontsize=12)
plt.ylabel('Age', fontsize=12)
plt.tight_layout()
plt.savefig('../results/eda_plots/02_age_by_credit_risk.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: 02_age_by_credit_risk.png")
plt.close()

# 3. Credit Amount Distribution
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(df['credit_amount'], bins=30, color='skyblue', edgecolor='black')
plt.title('Credit Amount Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Credit Amount', fontsize=10)
plt.ylabel('Frequency', fontsize=10)

plt.subplot(1, 2, 2)
df.boxplot(column='credit_amount', by='credit_risk', grid=False)
plt.suptitle('')
plt.title('Credit Amount by Risk', fontsize=12, fontweight='bold')
plt.xlabel('Credit Risk', fontsize=10)
plt.ylabel('Credit Amount', fontsize=10)

plt.tight_layout()
plt.savefig('../results/eda_plots/03_credit_amount_analysis.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: 03_credit_amount_analysis.png")
plt.close()

# 4. Correlation Heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df[numerical_features].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1)
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../results/eda_plots/04_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: 04_correlation_heatmap.png")
plt.close()

# 5. Duration vs Credit Amount (colored by risk)
plt.figure(figsize=(10, 6))
colors = ['red' if x == 0 else 'green' for x in df['credit_risk']]
plt.scatter(df['duration_months'], df['credit_amount'], c=colors, alpha=0.5)
plt.xlabel('Duration (months)', fontsize=12)
plt.ylabel('Credit Amount', fontsize=12)
plt.title('Duration vs Credit Amount (Red=Bad Credit, Green=Good Credit)', 
          fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('../results/eda_plots/05_duration_vs_amount.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: 05_duration_vs_amount.png")
plt.close()

# 6. Categorical features vs Credit Risk
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
cat_features_to_plot = ['checking_status', 'credit_history', 'savings_status', 
                        'employment', 'housing', 'job']

for idx, feature in enumerate(cat_features_to_plot):
    ax = axes[idx // 2, idx % 2]
    cross_tab = pd.crosstab(df[feature], df['credit_risk'], normalize='index') * 100
    cross_tab.plot(kind='bar', ax=ax, color=['#e74c3c', '#2ecc71'], width=0.7)
    ax.set_title(f'{feature.replace("_", " ").title()}', fontsize=11, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Percentage', fontsize=10)
    ax.legend(['Bad Credit', 'Good Credit'], fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

plt.tight_layout()
plt.savefig('../results/eda_plots/06_categorical_analysis.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: 06_categorical_analysis.png")
plt.close()

print("\n✓ All visualizations saved to results/eda_plots/")

# ============================================================================
# STEP 6: DATA PREPROCESSING
# ============================================================================
print("\n[STEP 6] Data Preprocessing")
print("-" * 70)

# Create a copy for preprocessing
df_processed = df.copy()

# Drop customer_id (not needed for modeling)
df_processed = df_processed.drop('customer_id', axis=1)

# Separate features and target
X = df_processed.drop('credit_risk', axis=1)
y = df_processed['credit_risk']

print(f"✓ Features shape: {X.shape}")
print(f"✓ Target shape: {y.shape}")

# Feature scaling for numerical features
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[numerical_features] = scaler.fit_transform(X[numerical_features])

print(f"\n✓ Applied StandardScaler to {len(numerical_features)} numerical features")

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✓ Train-Test Split:")
print(f"  Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"  Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"  Train class distribution: Good={sum(y_train==1)} ({sum(y_train==1)/len(y_train)*100:.1f}%), Bad={sum(y_train==0)} ({sum(y_train==0)/len(y_train)*100:.1f}%)")
print(f"  Test class distribution: Good={sum(y_test==1)} ({sum(y_test==1)/len(y_test)*100:.1f}%), Bad={sum(y_test==0)} ({sum(y_test==0)/len(y_test)*100:.1f}%)")

# ============================================================================
# STEP 7: SAVE PREPROCESSED DATA
# ============================================================================
print("\n[STEP 7] Saving Preprocessed Data")
print("-" * 70)

# Save full preprocessed dataset
df_processed.to_csv('../data/processed/credit_data_preprocessed.csv', index=False)
print("✓ Saved: data/processed/credit_data_preprocessed.csv")

# Save train-test splits
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

train_data.to_csv('../data/processed/train_data.csv', index=False)
test_data.to_csv('../data/processed/test_data.csv', index=False)

print("✓ Saved: data/processed/train_data.csv")
print("✓ Saved: data/processed/test_data.csv")

# Save scaler for later use
import pickle
with open('../data/processed/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Saved: data/processed/scaler.pkl")

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("\n" + "="*70)
print("PHASE 1 COMPLETION SUMMARY")
print("="*70)

print("\n✓ Dataset Generated: 1000 samples, 20 features")
print("✓ EDA Completed: 6 visualization plots created")
print("✓ Data Preprocessed: Scaled and split")
print("✓ Files Saved:")
print("  - data/raw/credit_data.csv")
print("  - data/processed/credit_data_preprocessed.csv")
print("  - data/processed/train_data.csv")
print("  - data/processed/test_data.csv")
print("  - data/processed/scaler.pkl")
print("  - 6 EDA plots in results/eda_plots/")

print("\n" + "="*70)
print("NEXT STEPS: Phase 2 - Centralized Baseline Models")
print("="*70)
print("\nReady to proceed with:")
print("  1. Logistic Regression baseline")
print("  2. Random Forest classifier")
print("  3. Neural Network model")
print("\n" + "="*70)
