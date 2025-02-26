from datenmanagement_wetter import x_train, x_test, x_vali, y_train, y_test, y_vali
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor


PARAMS = {"hidden_layer_sizes": (50, 20, 10)}
cls = MLPClassifier(**PARAMS)
cls.fit(x_train, y_train)

y_pred_train = cls.predict(x_train)
df_pred = pd.DataFrame({'actual': y_train,
                        'prediction': y_pred_train})

df_pred["correct"] = df_pred["actual"] == df_pred["prediction"]
anzahl_richtig_train = df_pred["correct"].sum()
print("Auf Trainingsdaten wurden ", anzahl_richtig_train, " von ", len(df_pred), " richtig klassifiziert.")
accuracy_train = anzahl_richtig_train / len(df_pred)
print("Accuracy: ", accuracy_train)

y_pred_vali = cls.predict(x_vali)
df_pred_vali = pd.DataFrame({'actual': y_vali,
                        'prediction': y_pred_vali})
df_pred_vali["correct"] = df_pred_vali["actual"] == df_pred_vali["prediction"]
anzahl_richtig_vali = df_pred_vali["correct"].sum()
print("Auf Validierungsdaten wurden ", anzahl_richtig_vali, " von ", len(df_pred_vali), " richtig klassifiziert.")
accuracy_vali = anzahl_richtig_vali / len(df_pred_vali)
print("Accuracy: ", accuracy_vali)

print("Overfitting: ", accuracy_train - accuracy_vali)

# from sklearn.tree import plot_tree
# import matplotlib.pyplot as plt
# plot_tree(cls,
#           feature_names=x_train.columns,
#           filled=True,
#           fontsize=5)
# plt.show(block=True)

# Feature Importances:
# df_features = pd.DataFrame({"features": cls.feature_importances_,
#                             "columns": x_train.columns})
# df_features = df_features.sort_values(by="features", ascending=False)
# print(df_features.head())
