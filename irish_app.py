import streamlit as st
import numpy as np
import joblib
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()

# Load your trained Decision Tree model
model = joblib.load('decision_tree_model.pkl')

# Streamlit UI
st.title("🌸 Iris Flower Species Predictor - Decision Tree")

st.write("Enter flower measurements below:")

# Input fields
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, value=5.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, value=3.5)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, value=1.4)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, value=0.2)

# Prediction button
if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    predicted_species = iris.target_names[prediction[0]]
    
    st.success(f"🌼 Predicted Species: **{predicted_species.capitalize()}**")