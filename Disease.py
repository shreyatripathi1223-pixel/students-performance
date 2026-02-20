import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# ---------------- Page Title ----------------
st.set_page_config(page_title="Disease Prediction App", page_icon="🩺")
st.title("🩺 Disease Prediction System")

# ---------------- Sample Dataset ----------------
data = {
    "Fever":      [1,1,0,1,1,1,1,1,0,1,0,0,0,0,0],
    "Cough":      [1,1,1,1,0,1,1,0,1,1,0,1,0,1,0],
    "Fatigue":    [1,1,1,0,1,1,1,1,1,0,1,0,1,0,0],
    "Body_Ache":  [1,0,1,1,1,1,0,1,1,1,1,1,0,0,1],
    "Headache":   [1,1,0,1,0,0,0,1,1,0,1,0,1,1,0],
    "Disease": [
        "Flu","Flu","Flu","Flu","Flu",
        "COVID-19","COVID-19","COVID-19","COVID-19","COVID-19",
        "Cold","Cold","Cold","Cold","Cold"
    ]
}

df = pd.DataFrame(data)
# ---------- Background Image ----------
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: 
        linear-gradient(rgba(0, 40, 85, 0.6), rgba(0, 40, 85, 0.6)),
        url("https://images.unsplash.com/photo-1576091160550-2173dba999ef");
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
# ---------------- Model Training ----------------
X = df.drop("Disease", axis=1)
y = df["Disease"]

model = DecisionTreeClassifier()
model.fit(X, y)

# ---------------- User Input ----------------
st.header("Enter Symptoms")

fever = st.selectbox("Fever", ["No", "Yes"])
cough = st.selectbox("Cough", ["No", "Yes"])
fatigue = st.selectbox("Fatigue", ["No", "Yes"])
body_ache = st.selectbox("Body Ache", ["No", "Yes"])
headache = st.selectbox("Headache", ["No", "Yes"])

# Convert Yes/No to 1/0
input_data = [[
    1 if fever == "Yes" else 0,
    1 if cough == "Yes" else 0,
    1 if fatigue == "Yes" else 0,
    1 if body_ache == "Yes" else 0,
    1 if headache == "Yes" else 0
]]

# ---------------- Prediction ----------------
if st.button("Predict Disease"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Disease: {prediction[0]}")