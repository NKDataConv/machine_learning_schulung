from datenmanagement_wetter import x_train, x_test, x_vali, y_train, y_test, y_vali
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import roc_curve, auc
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import mlflow
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint as sp_randint

mlflow.start_run()

# mlflow.log_artifact("evaluation_wetter.py")
# PARAMS = {"max_depth": 6,
#           "n_estimators": 100}
#
# mlflow.log_param("max_depth", PARAMS["max_depth"])

cls = DecisionTreeClassifier()
# param_grid = {"max_depth": [3,7],
#               "max_features": [5,15]}
param_dist = {"max_depth": sp_randint(3, 10),
              "max_features": sp_randint(1, 11)}

# grid_search = GridSearchCV(cls, param_grid=param_grid, cv=3, scoring="recall")
# grid_search.fit(x_train, y_train)

random_search = RandomizedSearchCV(cls, param_distributions=param_dist, cv=3, scoring="recall", n_iter=100)
random_search.fit(x_train, y_train)

cls = random_search.best_estimator_

for param in random_search.best_params_:
    mlflow.log_param(param, random_search.best_params_[param])

mlflow.log_param("classifier", "GradientBoostingClassifier")

# cls.fit(x_train, y_train)

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


# fpr, tpr, thresholds = roc_curve(y_vali, y_pred_proba)
# roc_auc = auc(fpr, tpr)
#
# fig = go.Figure()
#
# # ROC Curve hinzufügen
# fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
#                          name=f'ROC Curve (AUC = {roc_auc:.2f})',
#                          line=dict(color='blue', width=2)))
#
# # Diagonale Linie (Random Classifier)
# fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
#                          name='Random Guess',
#                          line=dict(color='gray', dash='dash')))
#
# # Layout anpassen
# fig.update_layout(title='ROC Curve',
#                   xaxis_title='False Positive Rate',
#                   yaxis_title='True Positive Rate',
#                   showlegend=True,
#                   width=700, height=500)
#
# # Plot anzeigen
# fig.show()
#
# cutoff_wert = max(thresholds[tpr > 0.8])
