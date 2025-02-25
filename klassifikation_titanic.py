from datenmanagement_titanic import x_train, x_test, x_vali, y_train, y_test, y_vali
from sklearn.tree import DecisionTreeClassifier
import mlflow
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import roc_curve, auc
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


mlflow.start_run()

mlflow.log_param("algo", "RandomForestClassifier")

# param_distribution = {"max_depth": [2, 3, 4],
#                       "n_estimators": [5, 7, 10, 12, 15],
#                       "min_samples_split": [2, 3, 4],
#                       "min_samples_leaf": [1, 2, 3]}
#
# random_search = RandomizedSearchCV(RandomForestClassifier(), param_distributions=param_distribution,
#                                       cv=3, scoring="accuracy", n_iter=50)
# random_search.fit(x_train, y_train)

# clf = random_search.best_estimator_
# PARAMS = random_search.best_params_
# print(PARAMS)
PARAMS = {"max_depth": 5, "n_estimators": 10}

for key, value in PARAMS.items():
    mlflow.log_param(key, value)

clf = RandomForestClassifier(**PARAMS)
clf.fit(x_train, y_train)

df_feature_importance = pd.DataFrame({"col": x_train.columns, "importance": clf.feature_importances_})
print(df_feature_importance.sort_values(by="importance", ascending=False))

y_pred_train = clf.predict(x_train)
y_pred_vali = clf.predict(x_vali)
y_pred_test = clf.predict(x_test)

accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_vali = accuracy_score(y_vali, y_pred_vali)
accuracy_test = accuracy_score(y_test, y_pred_test)
overfitting = accuracy_train - accuracy_vali
print("Accuracy Train: ", accuracy_train)
print("Accuracy Vali: ", accuracy_vali)
print("Accuracy Test: ", accuracy_test)
print("Overfitting: ", overfitting)
print("Information Leakage: ", accuracy_train - accuracy_test)

mlflow.log_metric("accuracy_train", accuracy_train)
mlflow.log_metric("accuracy_vali", accuracy_vali)

mlflow.end_run()

y_score = clf.predict_proba(x_vali) # [:, 1]
y_score = y_score[:, 1]

fpr, tpr, thresholds = roc_curve(y_vali, y_score)
roc_auc = auc(fpr, tpr)

fig = go.Figure()

# ROC Curve hinzufügen
fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                         name=f'ROC Curve (AUC = {roc_auc:.2f})',
                         line=dict(color='blue', width=2)))

# Diagonale Linie (Random Classifier)
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                         name='Random Guess',
                         line=dict(color='gray', dash='dash')))

# Layout anpassen
fig.update_layout(title='ROC Curve',
                  xaxis_title='False Positive Rate',
                  yaxis_title='True Positive Rate',
                  showlegend=True,
                  width=700, height=500)

# Plot anzeigen
fig.show()

