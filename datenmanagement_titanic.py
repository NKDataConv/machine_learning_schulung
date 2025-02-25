import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import plotly.graph_objs as go

pd.set_option("display.max_columns", 20)
df = pd.read_csv("daten/titanic.csv")

# print(df.columns)

# fig = go.Figure()
# fig.add_trace(go.Histogram(x=df["Pclass"], name="Pclass"))
# fig.show()
#
# fig = go.Figure()
# fig.add_trace(go.Histogram(x=df["Age"], name="Age"))
# fig.show()
#
# fig = go.Figure()
# fig.add_trace(go.Histogram(x=df["Survived"], name="Survived"))
# fig.show()

df.Embarked.unique()
df.Age.value_counts()
df.Survived.value_counts()
df.Sex.unique()

le = LabelEncoder()
le.fit(df["Sex"])
df["Sex"] = le.transform(df["Sex"])

columns_to_drop = ["PassengerId", "Name", "Cabin", "Ticket", "Embarked"]
df = df.drop(columns = columns_to_drop)

df = df.dropna()

y = df["Survived"]
df = df.drop(columns=["Survived"])

x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.4, random_state=42, stratify=y)

x_test, x_vali, y_test, y_vali = train_test_split(x_test, y_test, test_size=0.75, random_state=42, stratify=y_test)
