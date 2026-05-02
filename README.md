# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Acquisition & Preprocessing
Load the Iris dataset and separate the features (sepal/petal measurements) from the target labels (species). Apply Feature Scaling using StandardScaler to normalize the data, ensuring the gradient descent optimizer converges efficiently.

2. Dataset Partitioning
Split the processed data into two distinct sets: a Training Set (typically 80%) to teach the model and a Testing Set (20%) to evaluate its performance on unseen data. Use a fixed random_state to ensure results are reproducible.

3. Model Training (SGD)
Initialize the Stochastic Gradient Descent (SGD) Classifier. Fit the model using the training data, where the algorithm iteratively adjusts weights to minimize the loss function (linear SVM by default) across the feature space.

4. Model Evaluation
Generate predictions using the testing set. Compare these predictions against the ground truth labels to calculate the Accuracy Score and generate a Classification Report (Precision, Recall, and F1-score) to verify the model's reliability across all three Iris species.
## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: SARFARAZ I
RegisterNumber:  212225230252
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
iris = load_iris()
X = iris.data
y = iris.target
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = SGDClassifier(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
*/
```

## Output:

<img width="543" height="298" alt="image" src="https://github.com/user-attachments/assets/a1a48681-832b-4a72-bdad-95d31e7f3624" />



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
