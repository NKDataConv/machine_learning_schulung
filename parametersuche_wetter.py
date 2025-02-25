from datenmanagement_wetter import x_train, x_test, x_vali, y_train, y_test, y_vali
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score
import mlflow
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint as sp_randint

mlflow.start_run()

cls = DecisionTreeClassifier()
param_dist = {"max_depth": sp_randint(3, 10),
              "max_features": sp_randint(1, 11)}

random_search = RandomizedSearchCV(cls, param_distributions=param_dist, cv=3, scoring="recall", n_iter=100)
random_search.fit(x_train, y_train)

cls = random_search.best_estimator_

for param in random_search.best_params_:
    mlflow.log_param(param, random_search.best_params_[param])

mlflow.log_param("classifier", "DecisionTreeClassifier")

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
# print(cm)

y_pred_vali = cls.predict(x_vali)
df_pred_vali = pd.DataFrame({'actual': y_vali,
                        'prediction': y_pred_vali})

print("=======================================")
accuracy_vali = accuracy_score(df_pred_vali["actual"], df_pred_vali["prediction"])
print("Accuracy Vali: ", accuracy_vali)
precision_vali = precision_score(df_pred_vali["actual"], df_pred_vali["prediction"], pos_label=1)
print("Precision Vali: ", precision_vali)
recall_vali = recall_score(df_pred_vali["actual"], df_pred_vali["prediction"], pos_label=1)
print("Recall Train: ", recall_vali)

mlflow.log_metric("accuracy_train", accuracy_train)
mlflow.log_metric("accuracy_vali", accuracy_vali)

print("Overfitting Recall: ", recall_train - recall_vali)

y_pred_proba = cls.predict_proba(x_vali)[:,1]
y_pred = [1 if y > 0.7 else 0 for y in y_pred_proba]

df_pred_proba = pd.DataFrame({'actual': y_vali,
                        'prediction': y_pred})

print("=======================================")
accuracy_proba = accuracy_score(df_pred_proba["actual"], df_pred_proba["prediction"])
print("Accuracy Proba: ", accuracy_proba)
precision_proba = precision_score(df_pred_proba["actual"], df_pred_proba["prediction"], pos_label=1)
print("Precision Proba: ", precision_proba)
recall_proba = recall_score(df_pred_proba["actual"], df_pred_proba["prediction"], pos_label=1)
print("Recall Proba: ", recall_proba)



mlflow.end_run()
