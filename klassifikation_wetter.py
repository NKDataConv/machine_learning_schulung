from datenmanagement_wetter import x_train, x_test, x_vali, y_train, y_test, y_vali
from sklearn.tree import DecisionTreeClassifier

PARAMS = {"max_depth": 4}

cls = DecisionTreeClassifier(**PARAMS)
cls.fit(x_train, y_train)

y_pred_train = cls.predict(x_train)

import pandas as pd
df_pred = pd.DataFrame({'actual': y_train,
                        'prediction': y_pred_train})

df_pred["correct"] = df_pred["actual"] == df_pred["prediction"]
anzahl_richtig = df_pred["correct"].sum()
print("Auf Trainingsdaten wurden ", anzahl_richtig, " von ", len(df_pred), " richtig klassifiziert.")
print("Accuracy: ", anzahl_richtig / len(df_pred))

y_pred_vali = cls.predict(x_vali)
df_pred_vali = pd.DataFrame({'actual': y_vali,
                        'prediction': y_pred_vali})
df_pred_vali["correct"] = df_pred_vali["actual"] == df_pred_vali["prediction"]
anzahl_richtig = df_pred_vali["correct"].sum()
print("Auf Trainingsdaten wurden ", anzahl_richtig, " von ", len(df_pred_vali), " richtig klassifiziert.")
print("Accuracy: ", anzahl_richtig / len(df_pred_vali))

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plot_tree(cls,
          feature_names=x_train.columns,
          filled=True,
          fontsize=5)
plt.show(block=True)