import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import numpy as np

def load_data(file_path):
    """Load CSV data from the given file path."""
    data = pd.read_csv(file_path)
    print(f"Loaded data from {file_path} with shape {data.shape}")
    return data

def preprocess_data(df, target_column):
    """
    Preprocess data by:
    - Dropping rows with missing target
    - Encoding categorical features
    - Imputing missing values with KNN imputer
    - Scaling features
    - Splitting into train and test sets
    """
    # Drop rows missing target
    df = df.dropna(subset=[target_column])
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode categorical columns (convert to category codes)
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].astype('category').cat.codes

    # Impute missing values
    imputer = KNNImputer()
    X_imputed = imputer.fit_transform(X)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Stratify only if multiple classes exist in target to avoid errors
    stratify_param = y if len(np.unique(y)) > 1 else None

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=stratify_param
    )

    print(f"Preprocessing complete: Training samples = {X_train.shape[0]}, Test samples = {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test
