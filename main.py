# main.py

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Load example data
data = load_iris()
X = data.data
y = data.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train basic models
models = {
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    results[name] = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1
    }

results_df = pd.DataFrame(results).T
print("\n=== Baseline Results ===")
print(results_df)

# GridSearchCV example for Random Forest
param_grid_rf = {
    'n_estimators': [50, 100],
    'max_depth': [None, 5],
    'min_samples_split': [2, 5]
}

grid_search_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=3, scoring='accuracy')
grid_search_rf.fit(X_train, y_train)
print("\nBest RF Params:", grid_search_rf.best_params_)

# RandomizedSearchCV example for SVM
param_dist_svm = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'linear']
}

random_search_svm = RandomizedSearchCV(SVC(), param_dist_svm, n_iter=4, cv=3, scoring='accuracy')
random_search_svm.fit(X_train, y_train)
print("\nBest SVM Params:", random_search_svm.best_params_)

# Evaluate tuned models
print("\n=== Tuned Random Forest ===")
y_pred_rf = grid_search_rf.best_estimator_.predict(X_test)
print(classification_report(y_test, y_pred_rf))

print("\n=== Tuned SVM ===")
y_pred_svm = random_search_svm.best_estimator_.predict(X_test)
print(classification_report(y_test, y_pred_svm))
