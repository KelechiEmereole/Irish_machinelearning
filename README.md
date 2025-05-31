# 🌸 Iris Flower Classifier 

This project uses the classic Iris dataset to build and compare machine learning models that classify iris flowers into one of three species based on petal and sepal measurements.

---

##  Project Objectives
- Train multiple models to compare performance
- Evaluate accuracy and classification metrics
- Build a simple CLI for real-time predictions
- Practice an end-to-end ML workflow using scikit-learn

---

##  Dataset Details
- Features:
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
- Target:
  - Iris Setosa
  - Iris Versicolor
  - Iris Virginica
- Source: `sklearn.datasets.load_iris()`

---

##  Models Trained

| Model                    | Accuracy |
|--------------------------|----------|
| Logistic Regression      | 96.67%   |
| K-Nearest Neighbors (KNN)| 96.67%   |
| Decision Tree Classifier | ✅ 96.67% (Selected) |

---

##  Why I Chose Decision Tree

Although all three models achieved similar accuracy, the **Decision Tree Classifier** was chosen as the final model because:
- It offers **clear interpretability** through feature importance
- It performed with **perfect precision and recall for Setosa and Versicolor**
- It allows easy deployment and logic tracing, making it suitable for small-scale real-world applications

---

##  Decision Tree Evaluation

**Accuracy:** 96.67%  
**Confusion Matrix:**
[[11 0 0]
[ 0 12 1]
[ 0 0 6]]


**Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 (Setosa)     | 1.00 | 1.00 | 1.00 | 11 |
| 1 (Versicolor) | 1.00 | 0.92 | 0.96 | 13 |
| 2 (Virginica)  | 0.86 | 1.00 | 0.92 | 6 |
| **Avg/Total**  | 0.97 | 0.97 | 0.97 | 30 |

---

##  Feature Importances (Decision Tree)

| Feature             | Importance |
|---------------------|------------|
| Sepal Length (cm)   | 0.0%       |
| Sepal Width (cm)    | 0.0%       |
| Petal Length (cm)   | 69.9%      |
| Petal Width (cm)    | 30.1%      |

>  *Petal dimensions are far more influential than sepal dimensions in classifying iris species.*

---

##  How to Use
1. Clone the repo  
   `git clone https://github.com/KelechiEmereole/Irish_machinelearning.git`
2. Open and run the notebook:  
   `Iris.ipynb`
3. Use the CLI to input flower dimensions and get predictions

---

##  Tools Used
- Python
- Scikit-learn
- Pandas & NumPy
- Jupyter Notebook
- streamlit

##  Author
**Kelechi Emereole**  
> Passionate about machine learning, analytics, and building real-world projects. 

