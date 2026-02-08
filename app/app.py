import os
import sys
import joblib
import pandas as pd
import streamlit as st
import sklearn

# Paths
DATA_PATH = os.path.join("data", "Supermart-Grocery-Sales-Retail-Analytics-Dataset.csv")
MODEL_PATH = os.path.join("models", "sales_model.joblib")

st.set_page_config(page_title="Supermart Sales Prediction", layout="centered")
st.title("Supermart Sales Prediction")
st.caption("Predict Sales using order details (includes Profit as a feature).")

# Load model + data with caching

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    # Keep only columns we need for app options + date parsing
    needed = ["Category", "Sub Category", "City", "Region", "State", "Discount", "Profit", "Order Date"]
    df = df[needed].dropna()

    # Robust date parsing for mixed formats in this dataset
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="mixed", errors="coerce")
    df = df.dropna(subset=["Order Date"])

    return df

# Load
df = load_data()
model = load_model()

# Options
categories = sorted(df["Category"].unique().tolist())
cities = sorted(df["City"].unique().tolist())
regions = sorted(df["Region"].unique().tolist())
states = sorted(df["State"].unique().tolist())

# UI
st.subheader("Select inputs")

col1, col2 = st.columns(2)

with col1:
    category = st.selectbox("Category", categories)

    # Dependent dropdown: Sub Category based on Category
    subcats = sorted(df.loc[df["Category"] == category, "Sub Category"].unique().tolist())
    sub_category = st.selectbox("Sub Category", subcats)

    city = st.selectbox("City", cities)

with col2:
    region = st.selectbox("Region", regions)
    state = st.selectbox("State", states)

    discount = st.slider("Discount", 0.0, 1.0, 0.20, 0.01)
    profit = st.number_input("Profit", value=float(df["Profit"].median()), step=10.0)

# Date input (default to the most common date)
default_date = df["Order Date"].dt.date.mode().iloc[0]
order_date = st.date_input("Order Date", value=default_date)

# Feature engineering (same idea as training)
order_day = order_date.day
order_month = order_date.month
order_year = order_date.year

if st.button("Predict Sales"):
    X = pd.DataFrame([{
        "Category": category,
        "Sub Category": sub_category,
        "City": city,
        "Region": region,
        "State": state,
        "Discount": float(discount),
        "Profit": float(profit),
        "order_day": int(order_day),
        "order_month": int(order_month),
        "order_year": int(order_year),
    }])

    pred = model.predict(X)[0]
    st.success(f"Predicted Sales: {pred:.2f}")

st.divider()

# Optional helper: autofill from a real random row
if st.checkbox("Autofill from a random real row"):
    sample = df.sample(1, random_state=None).iloc[0]
    st.write("Sample row:", {
        "Category": sample["Category"],
        "Sub Category": sample["Sub Category"],
        "City": sample["City"],
        "Region": sample["Region"],
        "State": sample["State"],
        "Discount": float(sample["Discount"]),
        "Profit": float(sample["Profit"]),
        "Order Date": str(sample["Order Date"].date()),
    })
