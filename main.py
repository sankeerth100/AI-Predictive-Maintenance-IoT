import os
import pandas as pd

from src.data_preprocessing import load_data, preprocess_data
from src.train_model import train_all_models, evaluate_model, save_model
from src.visualization import (
    plot_failure_distribution,
    plot_confusion_matrix
)

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

DATA_PATH = "data/predictive_maintenance.csv"

df = load_data(DATA_PATH)

print("Dataset loaded successfully")
print(df.head())

plot_failure_distribution(df)

split_data, scaler, feature_columns = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data

best_model, results_df = train_all_models(X_train, X_test, y_train, y_test)

print("\nModel Comparison")
print(results_df)

y_pred = evaluate_model(best_model, X_test, y_test)

plot_confusion_matrix(y_test, y_pred)

save_model(best_model, scaler)

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

results.to_csv("outputs/predictions.csv", index=False)

print("\nBest model saved successfully.")
print("Project executed successfully.")