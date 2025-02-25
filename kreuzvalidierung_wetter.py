from datenmanagement_wetter import x_train, x_test, x_vali, y_train, y_test, y_vali
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import cross_validate

PARAMS = {"max_depth": 5,
          "min_samples_split": 400,
          "min_samples_leaf": 200}

cls = DecisionTreeClassifier(**PARAMS)
cv = cross_validate(cls, x_train, y_train, cv=5, scoring= "recall")
print(cv)
print(cv["test_score"].mean())
