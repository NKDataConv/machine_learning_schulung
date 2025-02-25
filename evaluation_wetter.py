from datenmanagement_wetter import x_train, x_test, x_vali, y_train, y_test, y_vali
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score


PARAMS = {"max_depth": 5,
          "min_samples_split": 400,
          "min_samples_leaf": 200}

cls = DecisionTreeClassifier(**PARAMS)
cls.fit(x_train, y_train)

y_pred_train = cls.predict(x_train)
df_pred = pd.DataFrame({'actual': y_train,
                        'prediction': y_pred_train})

accuracy_train = accuracy_score(df_pred["actual"], df_pred["prediction"])
print("Accuracy Train: ", accuracy_train)
precision_train = precision_score(df_pred["actual"], df_pred["prediction"], pos_label=1)
print("Precision Train: ", precision_train)
recall_train = recall_score(df_pred["actual"], df_pred["prediction"], pos_label=1)
print("Recall Train: ", recall_train)

cm = pd.crosstab(index=df_pred["prediction"],
                 columns=df_pred["actual"],
                 margins=True)
print(cm)

y_pred_vali = cls.predict(x_vali)
df_pred_vali = pd.DataFrame({'actual': y_vali,
                        'prediction': y_pred_vali})

accuracy_vali = accuracy_score(df_pred_vali["actual"], df_pred_vali["prediction"])
print("Accuracy Vali: ", accuracy_vali)
precision_vali = precision_score(df_pred_vali["actual"], df_pred_vali["prediction"], pos_label=1)
print("Precision Vali: ", precision_vali)
recall_vali = recall_score(df_pred_vali["actual"], df_pred_vali["prediction"], pos_label=1)
print("Recall Train: ", recall_vali)

print("Overfitting: ", recall_train - recall_vali)

y_pred_proba = cls.predict_proba(x_vali)
