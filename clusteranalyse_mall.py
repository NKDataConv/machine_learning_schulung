import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import plotly.graph_objs as go

df = pd.read_csv("daten/Mall_Customers.csv")
# print(df.head())
# print(df.shape)

df = df.drop(columns=["CustomerID"])
df = df.rename(columns={"Annual Income (k$)": "Income", "Spending Score (1-100)": "Score"})

le = LabelEncoder()
le.fit(df['Gender'])
df["Gender"] = le.transform(df["Gender"])

scaler = StandardScaler()
scaler.fit(df)
x = scaler.transform(df)

clustering = KMeans(n_clusters=4, n_init=10, random_state=42)
clustering.fit(x)

df['Cluster'] = clustering.labels_

print(df.head())

## Analyse der Cluster
df_agg = df.groupby("Cluster").agg({"Age": "mean", "Income": "mean", "Score": "mean", "Gender": "mean"})
print(df_agg)


df["symbol"] = df["Gender"].map(lambda x: "circle" if x == 0 else "x")

fig = go.Figure()
fig.add_trace(go.Scatter3d(x=df['Income'],
                           y=df['Score'],
                           z=df['Age'],
                           mode='markers',
                           marker_symbol = df["symbol"],
                           marker=dict(color=df['Cluster'],
                                       size=3),
                           text=df['Cluster']))
fig.update_layout(scene=dict(xaxis_title='Income',
                             yaxis_title='Score',
                             zaxis_title='Age'))
fig.show()