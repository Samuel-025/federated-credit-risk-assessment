"""
Data Loader Module
Handles loading and initial processing of credit risk datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class CreditDataLoader:
    """
    Unified data loader for credit risk datasets.
    Supports: German Credit, Lending Club, Home Credit
    """
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize data loader.
        
        Args:
            data_dir: Path to raw data directory
        """
        self.data_dir = Path(data_dir)
        
    def load_german_credit(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load German Credit dataset from UCI Repository.
        
        Returns:
            X: Feature dataframe
            y: Target series (1=Good, 0=Bad)
        """
        # Column names as per UCI documentation
        column_names = [
            'checking_status', 'duration', 'credit_history', 'purpose',
            'credit_amount', 'savings_status', 'employment', 
            'installment_commitment', 'personal_status', 'other_parties',
            'residence_since', 'property_magnitude', 'age', 
            'other_payment_plans', 'housing', 'existing_credits',
            'job', 'num_dependents', 'own_telephone', 'foreign_worker',
            'class'
        ]
        
        # Load data
        file_path = self.data_dir / "german_credit" / "german.data"
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"German credit data not found at {file_path}\n"
                f"Please download from: "
                f"https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data"
            )
        
        # Read space-separated file
        df = pd.read_csv(file_path, sep=' ', names=column_names, header=None)
        
        # Separate features and target
        X = df.drop('class', axis=1)
        y = df['class'].map({1: 1, 2: 0})  # 1=Good, 0=Bad (default)
        
        print(f"German Credit Data Loaded:")
        print(f"  - Records: {len(df)}")
        print(f"  - Features: {len(X.columns)}")
        print(f"  - Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def load_lending_club(self, sample_frac: float = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load Lending Club dataset.
        
        Args:
            sample_frac: Fraction of data to sample (for quick testing)
            
        Returns:
            X: Feature dataframe
            y: Target series (1=Fully Paid, 0=Charged Off/Default)
        """
        file_path = self.data_dir / "lending_club" / "accepted_2007_to_2018Q4.csv.gz"
        
        if not file_path.exists():
            # Try without .gz
            file_path = self.data_dir / "lending_club" / "accepted_2007_to_2018Q4.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"Lending Club data not found at {file_path}\n"
                f"Please download from: "
                f"https://www.kaggle.com/datasets/wordsforthewise/lending-club"
            )
        
        print("Loading Lending Club data (this may take a moment)...")
        
        # Read CSV (handles both .csv and .csv.gz)
        df = pd.read_csv(file_path, low_memory=False)
        
        # Sample if requested (for testing)
        if sample_frac:
            df = df.sample(frac=sample_frac, random_state=42)
            print(f"  - Sampled {sample_frac*100}% of data")
        
        # Filter to relevant loan statuses
        # Keep only completed loans (exclude current/in-grace)
        relevant_statuses = [
            'Fully Paid', 'Charged Off', 'Default',
            'Does not meet the credit policy. Status:Fully Paid',
            'Does not meet the credit policy. Status:Charged Off'
        ]
        df = df[df['loan_status'].isin(relevant_statuses)].copy()
        
        # Create binary target
        good_statuses = ['Fully Paid', 'Does not meet the credit policy. Status:Fully Paid']
        df['target'] = df['loan_status'].apply(lambda x: 1 if x in good_statuses else 0)
        
        # Separate features and target
        # Drop target and identifier columns
        drop_cols = ['loan_status', 'target', 'id', 'member_id', 'url', 'desc']
        drop_cols = [col for col in drop_cols if col in df.columns]
        
        X = df.drop(drop_cols, axis=1)
        y = df['target']
        
        print(f"Lending Club Data Loaded:")
        print(f"  - Records: {len(df):,}")
        print(f"  - Features: {len(X.columns)}")
        print(f"  - Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def load_home_credit(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load Home Credit Default Risk dataset (main application table).
        
        Returns:
            X: Feature dataframe
            y: Target series (1=No default, 0=Default)
        """
        file_path = self.data_dir / "home_credit" / "application_train.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"Home Credit data not found at {file_path}\n"
                f"Please download from: "
                f"https://www.kaggle.com/c/home-credit-default-risk/data"
            )
        
        # Load main application table
        df = pd.read_csv(file_path)
        
        # Separate features and target
        X = df.drop(['TARGET', 'SK_ID_CURR'], axis=1)
        y = 1 - df['TARGET']  # Invert: 1=No default, 0=Default
        
        print(f"Home Credit Data Loaded:")
        print(f"  - Records: {len(df):,}")
        print(f"  - Features: {len(X.columns)}")
        print(f"  - Target distribution: {y.value_counts().to_dict()}")
        print(f"  - Note: Highly imbalanced dataset!")
        
        return X, y
    
    def get_dataset_info(self) -> Dict[str, Dict]:
        """
        Get information about available datasets.
        
        Returns:
            Dictionary with dataset information
        """
        info = {
            'german_credit': {
                'name': 'German Credit Data',
                'size': '1,000 records',
                'features': '20',
                'source': 'UCI ML Repository',
                'difficulty': 'Beginner',
                'recommended_for': 'Initial prototype and testing'
            },
            'lending_club': {
                'name': 'Lending Club Loan Data',
                'size': '2+ million records',
                'features': '150+',
                'source': 'Kaggle',
                'difficulty': 'Intermediate',
                'recommended_for': 'Final experiments and results'
            },
            'home_credit': {
                'name': 'Home Credit Default Risk',
                'size': '300k+ records',
                'features': '122 (main table)',
                'source': 'Kaggle Competition',
                'difficulty': 'Advanced',
                'recommended_for': 'Optional - Additional experiments'
            }
        }
        return info


def main():
    """
    Example usage and testing of data loader.
    """
    loader = CreditDataLoader(data_dir="../../data/raw")
    
    # Display available datasets
    print("=" * 60)
    print("AVAILABLE CREDIT RISK DATASETS")
    print("=" * 60)
    info = loader.get_dataset_info()
    for dataset_key, dataset_info in info.items():
        print(f"\n{dataset_info['name']}:")
        for key, value in dataset_info.items():
            if key != 'name':
                print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print("\n" + "=" * 60)
    print("LOADING TEST")
    print("=" * 60)
    
    # Try to load German Credit (smallest)
    try:
        print("\nAttempting to load German Credit...")
        X, y = loader.load_german_credit()
        print("✓ Successfully loaded German Credit!")
        print(f"\nFirst few rows:")
        print(X.head())
        print(f"\nTarget distribution:")
        print(y.value_counts())
    except FileNotFoundError as e:
        print(f"✗ {e}")


if __name__ == "__main__":
    main()
