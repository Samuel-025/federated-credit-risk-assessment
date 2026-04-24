"""
Data Preprocessing Utilities
Handles feature engineering, encoding, scaling for German Credit Dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from pathlib import Path
import joblib

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


class CreditDataPreprocessor:
    """
    Comprehensive preprocessing pipeline for German Credit Dataset
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = None
        self.feature_names = None
        self.categorical_features = None
        self.numerical_features = None
        
    def identify_feature_types(self, df):
        """
        Identify numerical and categorical features
        
        Args:
            df (pd.DataFrame): Raw dataset
            
        Returns:
            tuple: (numerical_features, categorical_features)
        """
        # Numerical features (excluding target)
        self.numerical_features = [
            'duration', 'credit_amount', 'installment_rate',
            'residence_since', 'age', 'existing_credits', 'num_dependents'
        ]
        
        # Categorical features
        self.categorical_features = [
            'checking_status', 'credit_history', 'purpose', 'savings_status',
            'employment', 'personal_status', 'other_parties', 'property_magnitude',
            'other_payment_plans', 'housing', 'job', 'own_telephone', 'foreign_worker'
        ]
        
        return self.numerical_features, self.categorical_features
    
    def encode_target(self, df):
        """
        Encode target variable: 1 (good) -> 0, 2 (bad) -> 1
        
        Args:
            df (pd.DataFrame): Dataset with 'class' column
            
        Returns:
            pd.DataFrame: Dataset with binary target (0/1)
        """
        df = df.copy()
        # Convert class: 1 (good credit) -> 0, 2 (bad credit) -> 1
        df['target'] = (df['class'] == 2).astype(int)
        df = df.drop('class', axis=1)
        return df
    
    def encode_categorical_features(self, df, method='label', fit=True):
        """
        Encode categorical features
        
        Args:
            df (pd.DataFrame): Dataset
            method (str): 'label' for label encoding, 'onehot' for one-hot encoding
            fit (bool): Whether to fit encoders (True for train, False for test)
            
        Returns:
            pd.DataFrame: Dataset with encoded features
        """
        df = df.copy()
        
        if method == 'label':
            for col in self.categorical_features:
                if col in df.columns:
                    if fit:
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col].astype(str))
                        self.label_encoders[col] = le
                    else:
                        if col in self.label_encoders:
                            # Handle unseen categories
                            le = self.label_encoders[col]
                            df[col] = df[col].map(lambda x: le.transform([str(x)])[0] 
                                                  if str(x) in le.classes_ 
                                                  else -1)
        
        elif method == 'onehot':
            # One-hot encoding
            df = pd.get_dummies(df, columns=self.categorical_features, drop_first=True)
        
        return df
    
    def scale_numerical_features(self, df, method='standard', fit=True):
        """
        Scale numerical features
        
        Args:
            df (pd.DataFrame): Dataset
            method (str): 'standard' or 'minmax'
            fit (bool): Whether to fit scaler
            
        Returns:
            pd.DataFrame: Dataset with scaled features
        """
        df = df.copy()
        
        # Select numerical columns present in df
        num_cols = [col for col in self.numerical_features if col in df.columns]
        
        if fit:
            if method == 'standard':
                self.scaler = StandardScaler()
            else:
                self.scaler = MinMaxScaler()
            
            df[num_cols] = self.scaler.fit_transform(df[num_cols])
        else:
            if self.scaler is not None:
                df[num_cols] = self.scaler.transform(df[num_cols])
        
        return df
    
    def feature_engineering(self, df):
        """
        Create additional features
        
        Args:
            df (pd.DataFrame): Dataset
            
        Returns:
            pd.DataFrame: Dataset with new features
        """
        df = df.copy()
        
        # Credit utilization (if we had credit limit, but we don't - so create proxy)
        # Duration to amount ratio
        df['duration_amount_ratio'] = df['duration'] / (df['credit_amount'] + 1)
        
        # Age groups
        df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 100], 
                                 labels=['young', 'middle', 'senior', 'elderly'])
        
        # Credit amount categories
        df['amount_category'] = pd.cut(df['credit_amount'], 
                                       bins=[0, 2000, 5000, 10000, 20000],
                                       labels=['low', 'medium', 'high', 'very_high'])
        
        # Duration categories
        df['duration_category'] = pd.cut(df['duration'],
                                         bins=[0, 12, 24, 36, 100],
                                         labels=['short', 'medium', 'long', 'very_long'])
        
        return df
    
    def handle_class_imbalance(self, X, y, method='smote', random_state=42):
        """
        Handle class imbalance
        
        Args:
            X (array): Features
            y (array): Target
            method (str): 'smote', 'oversample', or 'none'
            random_state (int): Random seed
            
        Returns:
            tuple: (X_resampled, y_resampled)
        """
        if method == 'smote':
            smote = SMOTE(random_state=random_state)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            return X_resampled, y_resampled
        
        elif method == 'oversample':
            # Simple random oversampling of minority class
            from sklearn.utils import resample
            
            # Combine X and y
            data = pd.concat([pd.DataFrame(X), pd.Series(y, name='target')], axis=1)
            
            # Separate classes
            majority = data[data['target'] == 0]
            minority = data[data['target'] == 1]
            
            # Oversample minority
            minority_upsampled = resample(minority, 
                                         replace=True,
                                         n_samples=len(majority),
                                         random_state=random_state)
            
            # Combine
            upsampled = pd.concat([majority, minority_upsampled])
            
            y_resampled = upsampled['target'].values
            X_resampled = upsampled.drop('target', axis=1).values
            
            return X_resampled, y_resampled
        
        else:
            return X, y
    
    def preprocess_pipeline(self, df, encoding='label', scaling='standard', 
                           feature_eng=True, test_size=0.3, random_state=42,
                           balance_method='none'):
        """
        Complete preprocessing pipeline
        
        Args:
            df (pd.DataFrame): Raw dataset
            encoding (str): Categorical encoding method
            scaling (str): Numerical scaling method
            feature_eng (bool): Whether to create engineered features
            test_size (float): Test set proportion
            random_state (int): Random seed
            balance_method (str): Class balancing method
            
        Returns:
            dict: Preprocessed train/test splits and metadata
        """
        print("Starting preprocessing pipeline...")
        
        # 1. Identify feature types
        self.identify_feature_types(df)
        print(f"✓ Identified {len(self.numerical_features)} numerical and "
              f"{len(self.categorical_features)} categorical features")
        
        # 2. Encode target
        df = self.encode_target(df)
        print("✓ Target variable encoded (0: good, 1: bad)")
        
        # 3. Feature engineering (optional)
        if feature_eng:
            df = self.feature_engineering(df)
            print("✓ Feature engineering completed")
            # Update categorical features list
            new_cats = ['age_group', 'amount_category', 'duration_category']
            self.categorical_features.extend(new_cats)
        
        # 4. Encode categorical features
        df = self.encode_categorical_features(df, method=encoding, fit=True)
        print(f"✓ Categorical features encoded using {encoding} encoding")
        
        # 5. Scale numerical features
        df = self.scale_numerical_features(df, method=scaling, fit=True)
        print(f"✓ Numerical features scaled using {scaling} scaling")
        
        # 6. Split features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # 7. Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"✓ Data split: {len(X_train)} train, {len(X_test)} test samples")
        
        # 8. Handle class imbalance (only on training data)
        if balance_method != 'none':
            X_train, y_train = self.handle_class_imbalance(
                X_train, y_train, method=balance_method, random_state=random_state
            )
            print(f"✓ Class imbalance handled using {balance_method}")
        
        # 9. Return preprocessed data
        result = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': self.feature_names,
            'preprocessor': self,
            'n_features': len(self.feature_names),
            'class_distribution_train': pd.Series(y_train).value_counts().to_dict(),
            'class_distribution_test': pd.Series(y_test).value_counts().to_dict()
        }
        
        print("\n" + "="*70)
        print("PREPROCESSING COMPLETE")
        print("="*70)
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Features: {len(self.feature_names)}")
        print(f"Train class distribution: {result['class_distribution_train']}")
        print(f"Test class distribution: {result['class_distribution_test']}")
        print("="*70 + "\n")
        
        return result
    
    def save_preprocessor(self, filepath):
        """
        Save preprocessor for later use
        
        Args:
            filepath (str): Path to save preprocessor
        """
        joblib.dump(self, filepath)
        print(f"✓ Preprocessor saved to {filepath}")
    
    @staticmethod
    def load_preprocessor(filepath):
        """
        Load saved preprocessor
        
        Args:
            filepath (str): Path to preprocessor file
            
        Returns:
            CreditDataPreprocessor: Loaded preprocessor
        """
        preprocessor = joblib.load(filepath)
        print(f"✓ Preprocessor loaded from {filepath}")
        return preprocessor


def save_processed_data(data_dict, prefix='german_credit'):
    """
    Save preprocessed data to files
    
    Args:
        data_dict (dict): Dictionary with train/test splits
        prefix (str): Filename prefix
    """
    # Save as numpy arrays
    np.save(PROCESSED_DATA_DIR / f'{prefix}_X_train.npy', data_dict['X_train'])
    np.save(PROCESSED_DATA_DIR / f'{prefix}_X_test.npy', data_dict['X_test'])
    np.save(PROCESSED_DATA_DIR / f'{prefix}_y_train.npy', data_dict['y_train'])
    np.save(PROCESSED_DATA_DIR / f'{prefix}_y_test.npy', data_dict['y_test'])
    
    # Save as CSV (for inspection)
    pd.DataFrame(data_dict['X_train'], columns=data_dict['feature_names']).to_csv(
        PROCESSED_DATA_DIR / f'{prefix}_X_train.csv', index=False
    )
    pd.DataFrame(data_dict['X_test'], columns=data_dict['feature_names']).to_csv(
        PROCESSED_DATA_DIR / f'{prefix}_X_test.csv', index=False
    )
    
    # Save metadata
    metadata = {
        'feature_names': data_dict['feature_names'],
        'n_features': data_dict['n_features'],
        'n_train': len(data_dict['X_train']),
        'n_test': len(data_dict['X_test']),
        'class_distribution_train': data_dict['class_distribution_train'],
        'class_distribution_test': data_dict['class_distribution_test']
    }
    
    import json
    with open(PROCESSED_DATA_DIR / f'{prefix}_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Processed data saved to {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    """
    Test preprocessing pipeline
    """
    from data_loader import load_german_credit_data
    
    print("\n" + "="*70)
    print("TESTING PREPROCESSING PIPELINE")
    print("="*70 + "\n")
    
    # Load data
    df = load_german_credit_data()
    
    if df is not None:
        # Create preprocessor
        preprocessor = CreditDataPreprocessor()
        
        # Run pipeline
        result = preprocessor.preprocess_pipeline(
            df,
            encoding='label',
            scaling='standard',
            feature_eng=True,
            test_size=0.3,
            balance_method='none'  # Try 'smote' for balanced data
        )
        
        # Save preprocessed data
        save_processed_data(result)
        
        # Save preprocessor
        preprocessor.save_preprocessor(
            PROCESSED_DATA_DIR / 'preprocessor.pkl'
        )
        
        print("\n✅ Preprocessing pipeline test complete!")
