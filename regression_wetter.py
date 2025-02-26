from datenmanagement_wetter import x_train, x_vali, x_test, risk_mm_vali, risk_mm_train, risk_mm_test
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

PARAMS = {"n_estimators": 20, "max_features": 4, "max_depth": 3}
reg = GradientBoostingRegressor(**PARAMS)
reg.fit(X=x_train, y=risk_mm_train)

predictions_train = reg.predict(x_train)
df_pred = pd.DataFrame({'actual': risk_mm_train, 'prediction': predictions_train})
df_pred["fehler"] = abs(df_pred["actual"] - df_pred["prediction"])
print("Gesamtfehler Train:", df_pred["fehler"].mean())

mae = mean_absolute_error(risk_mm_train, predictions_train)
print("Gesamtfehler Train:", mae)

mse = mean_squared_error(risk_mm_train, predictions_train)
print("MSE Train", mse)

predictions_vali = reg.predict(x_vali)
mae = mean_absolute_error(risk_mm_vali, predictions_vali)
print("Gesamtfehler Vali:", mae)

mse = mean_squared_error(risk_mm_vali, predictions_vali)
print("MSE Vali", mse)

import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=risk_mm_vali, y=predictions_vali, mode='markers'))
fig.show()
