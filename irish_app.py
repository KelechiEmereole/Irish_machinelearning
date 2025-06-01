import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Load data
iris = load_iris()

def load_model():
    try:
        return joblib.load('iris_model.joblib')
    except:
        print("Using fallback model (training new one)...")
        model = DecisionTreeClassifier(random_state=42)
        return model.fit(iris.data, iris.target)

model = load_model()

print("Iris Flower Classifier\n")

# Get user inputs via console
sepal_length = float(input("Sepal Length (cm): "))
sepal_width = float(input("Sepal Width (cm): "))
petal_length = float(input("Petal Length (cm): "))
petal_width = float(input("Petal Width (cm): "))

# Prepare input data for prediction
input_df = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                        columns=iris.feature_names)

# Make prediction
prediction = model.predict(input_df)
species = iris.target_names[prediction[0]].capitalize()

print(f"\nPredicted species: {species}")

# Save the output to a text file

output_text = (
    f"Sepal length: {sepal_length}\n"
    f"Sepal width: {sepal_width}\n"
    f"Petal length: {petal_length}\n"
    f"Petal width: {petal_width}\n"
    f"Predicted species: {species}\n"
)

# Save to a text file
with open("iris_prediction_output.txt", "w") as f:
    f.write(output_text)
