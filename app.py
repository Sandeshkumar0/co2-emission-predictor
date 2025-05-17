
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("co2.csv")

# Model training
X = df[["Engine Size (L)", "Cylinders", "Fuel Consumption Comb (L/100 km)"]]
y = df["CO2 Emissions (g/km)"]
model = LinearRegression()
model.fit(X, y)

# ---------- Streamlit App UI ----------

# Page config
st.set_page_config(page_title="CO₂ Emission Predictor", page_icon="🌱", layout="centered")

# Sidebar for inputs
st.sidebar.title("Input Vehicle Details")
engine_size = st.sidebar.slider("Engine Size (L)", 1.0, 7.0, 2.0, 0.1)
cylinders = st.sidebar.selectbox("Cylinders", options=[3, 4, 5, 6, 8, 10, 12])
fuel_cons = st.sidebar.slider("Fuel Consumption (L/100 km)", 4.0, 20.0, 8.5, 0.1)

# Header
st.title("🌿 CO₂ Emission Prediction App")
st.markdown("This app predicts the **CO₂ emissions** of your vehicle based on its engine specifications using a trained **Multiple Linear Regression model**.")

# Layout: split
st.write("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Your Input")
    st.write(f"**Engine Size:** {engine_size} L")
    st.write(f"**Cylinders:** {cylinders}")
    st.write(f"**Fuel Consumption:** {fuel_cons} L/100km")

with col2:
    # Prediction
    pred_input = [[engine_size, cylinders, fuel_cons]]
    prediction = model.predict(pred_input)[0]
    st.subheader("💨 Predicted CO₂ Emission")
    st.metric("CO₂ (g/km)", f"{prediction:.2f}")

# Show raw data (optional)
with st.expander("🔍 View Sample Dataset"):
    st.dataframe(df.head(10))
