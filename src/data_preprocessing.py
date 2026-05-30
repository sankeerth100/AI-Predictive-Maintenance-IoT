import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    df = df.copy()

    # Drop unnecessary columns if available
    columns_to_drop = ["UDI", "Product ID"]
    for col in columns_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    # Encode categorical column
    if "Type" in df.columns:
        encoder = LabelEncoder()
        df["Type"] = encoder.fit_transform(df["Type"])

    # Target column
    target = "Machine failure"

    X = df.drop(target, axis=1)
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    ), scaler, X.columns