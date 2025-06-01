import streamlit as st
import joblib
import numpy as np

# Load the saved decision tree model
model = joblib.load('iris_model.joblib')

target_names = ['setosa', 'versicolor', 'virginica']

st.title("🌸 Iris Flower Species Prediction (Decision Tree)")

sepal_length = st.number_input("Sepal Length (cm)", value=5.1)
sepal_width = st.number_input("Sepal Width (cm)", value=3.5)
petal_length = st.number_input("Petal Length (cm)", value=1.4)
petal_width = st.number_input("Petal Width (cm)", value=0.2)

if st.button("Predict Species"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    st.success(f"🌼 Predicted species: **{target_names[prediction].capitalize()}**")
