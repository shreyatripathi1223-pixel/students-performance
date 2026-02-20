import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Title
st.title("🏠 House Price Prediction App")

# Sample Dataset (Same as your CSV)
data = {
    "Size_sqft": [800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900],
    "Bedrooms": [2,2,3,3,3,4,4,4,5,5,5,6],
    "Age_years": [10,8,7,5,6,4,3,2,2,1,1,1],
    "Price": [30000,35000,42000,46000,48000,54000,60000,65000,70000,76000,82000,88000]
}

df = pd.DataFrame(data)

# ---------- House Background Image ----------
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: 
        linear-gradient(rgba(0, 0, 0, 0.55), rgba(0, 0, 0, 0.55)),
        url("https://images.unsplash.com/photo-1560518883-ce09059eeffa");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

[data-testid="stToolbar"] {
    right: 2rem;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Features and Target
X = df[["Size_sqft", "Bedrooms", "Age_years"]]
y = df["Price"]

# Train Model
model = LinearRegression()
model.fit(X, y)

# User Input
st.header("Enter House Details")

size = st.number_input("Size (sqft)", min_value=500, max_value=3000, value=1000)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
age = st.number_input("Age of House (years)", min_value=0, max_value=50, value=5)

# Prediction
if st.button("Predict Price"):
    input_data = np.array([[size, bedrooms, age]])
    prediction = model.predict(input_data)
    st.success(f"Estimated House Price: ₹ {int(prediction[0])}")