import streamlit as st
import pandas as pd
import joblib
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Load data
iris = load_iris()

@st.cache_resource
def load_model():
    try:
        return joblib.load('iris_model.joblib')
    except:
        st.warning("Using fallback model")
        model = DecisionTreeClassifier(random_state=42)
        return model.fit(iris.data, iris.target)

model = load_model()

st.title("Iris Flower Classifier")

# Input fields
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, value=5.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, value=3.5)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, value=1.4)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, value=0.2)

if st.button("Predict"):
    input_df = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                          columns=iris.feature_names)
    prediction = model.predict(input_df)
    species = iris.target_names[prediction[0]].capitalize()
    st.success(f"Predicted species: {species}")