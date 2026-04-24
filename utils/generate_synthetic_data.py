"""
Synthetic Credit Risk Dataset Generator
Based on German Credit Dataset structure
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

def generate_credit_dataset(n_samples=1000, random_state=42):
    """Generate synthetic credit risk dataset with realistic features"""
    
    np.random.seed(random_state)
    
    data = {}
    
    # Numerical features
    data['duration_months'] = np.random.randint(3, 73, n_samples)
    data['credit_amount'] = np.random.randint(500, 20001, n_samples)
    data['installment_rate'] = np.random.randint(1, 5, n_samples)
    data['residence_since'] = np.random.randint(1, 5, n_samples)
    data['age'] = np.random.randint(19, 76, n_samples)
    data['existing_credits'] = np.random.randint(1, 5, n_samples)
    data['num_dependents'] = np.random.randint(0, 3, n_samples)
    
    # Categorical features
    data['checking_status'] = np.random.choice([0, 1, 2, 3], n_samples, p=[0.25, 0.35, 0.25, 0.15])
    data['credit_history'] = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.05, 0.25, 0.40, 0.20, 0.10])
    data['purpose'] = np.random.choice(range(11), n_samples)
    data['savings_status'] = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.3, 0.25, 0.2, 0.15, 0.1])
    data['employment'] = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.1, 0.15, 0.25, 0.25, 0.25])
    data['personal_status'] = np.random.choice([0, 1, 2, 3, 4], n_samples)
    data['other_parties'] = np.random.choice([0, 1, 2], n_samples, p=[0.75, 0.15, 0.10])
    data['property_magnitude'] = np.random.choice([0, 1, 2, 3], n_samples, p=[0.25, 0.30, 0.25, 0.20])
    data['other_payment_plans'] = np.random.choice([0, 1, 2], n_samples, p=[0.15, 0.10, 0.75])
    data['housing'] = np.random.choice([0, 1, 2], n_samples, p=[0.40, 0.50, 0.10])
    data['job'] = np.random.choice([0, 1, 2, 3], n_samples, p=[0.05, 0.20, 0.50, 0.25])
    data['telephone'] = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
    data['foreign_worker'] = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    
    # Generate credit risk based on features
    credit_risk = []
    for i in range(n_samples):
        score = 0
        if data['checking_status'][i] >= 2: score += 2
        if data['credit_history'][i] in [1, 2]: score += 3
        if data['savings_status'][i] >= 2: score += 2
        if data['employment'][i] >= 3: score += 2
        if data['age'][i] >= 35: score += 1
        if data['credit_amount'][i] < 5000: score += 1
        if data['credit_history'][i] in [3, 4]: score -= 3
        if data['employment'][i] == 0: score -= 2
        if data['duration_months'][i] > 36: score -= 1
        if data['installment_rate'][i] >= 3: score -= 1
        
        if score >= 4:
            credit_risk.append(1 if np.random.random() > 0.15 else 0)
        elif score <= 0:
            credit_risk.append(0 if np.random.random() > 0.2 else 1)
        else:
            credit_risk.append(1 if np.random.random() > 0.3 else 0)
    
    data['credit_risk'] = credit_risk
    df = pd.DataFrame(data)
    df.insert(0, 'customer_id', [f'CUST_{i:05d}' for i in range(n_samples)])
    
    return df

# Generate and save
print("Generating Synthetic Credit Risk Dataset...")
df = generate_credit_dataset(n_samples=1000, random_state=42)
df.to_csv('data/raw/credit_data.csv', index=False)

print(f"✓ Dataset saved to data/raw/credit_data.csv")
print(f"  Shape: {df.shape}")
print(f"  Good Credit (1): {(df['credit_risk']==1).sum()} ({(df['credit_risk']==1).sum()/len(df)*100:.1f}%)")
print(f"  Bad Credit (0): {(df['credit_risk']==0).sum()} ({(df['credit_risk']==0).sum()/len(df)*100:.1f}%)")
print("\nFirst 5 rows:")
print(df.head())
