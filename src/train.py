import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = os.path.join("data", "Supermart-Grocery-Sales-Retail-Analytics-Dataset.csv")
MODEL_PATH = os.path.join("models", "sales_model.joblib")

def main():
    df = pd.read_csv(DATA_PATH)

    # Basic cleanup similar to PDF guidance (handle duplicates / missing)
    df = df.drop_duplicates()
    df = df.dropna()

    # Date parsing + features (PDF suggests deriving day/month/year)
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce", dayfirst=False)
    df = df.dropna(subset=["Order Date"])

    df["order_day"] = df["Order Date"].dt.day
    df["order_month"] = df["Order Date"].dt.month
    df["order_year"] = df["Order Date"].dt.year

    # Target (Sales)
    y = df["Sales"]

    # Features: include Profit (as you requested) + other useful columns from dataset
    X = df.drop(columns=["Sales", "Order ID", "Customer Name", "Order Date"])

    cat_cols = ["Category", "Sub Category", "City", "Region", "State"]
    num_cols = ["Discount", "Profit", "order_day", "order_month", "order_year"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)

    print(f"Saved model: {MODEL_PATH}")
    print(f"MSE: {mse:.4f}")
    print(f"R2: {r2:.4f}")

if __name__ == "__main__":
    main()
