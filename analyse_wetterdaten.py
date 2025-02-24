import pandas as pd

df = pd.read_csv("daten/weatherAUS.csv")

df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date")

df.columns

# Aufgabe 1:
max_temp = df["MaxTemp"].max()
mask = df["MaxTemp"] == max_temp
df_max_temp = df.loc[mask, "Location"]
print("Datum: ", df_max_temp.index[0])
print("Location: ", df_max_temp.values[0])

# Aufgabe 2:
df_agg = df.groupby("Location").agg({"Temp9am": "mean"})
df_agg = df_agg.sort_values(by="Temp9am", ascending=False)

# alternativ:
df_agg.idxmax()

# Aufgabe 3:
df_agg = df.groupby("Location").agg({"MaxTemp": "std"})
df_agg.sort_values(by="MaxTemp")

df_agg.idxmin()

# Aufgabe 4:
df["month"] = df.index.month
df_agg = df.groupby("month").agg({"MinTemp": "min"})
df_agg.idxmin()

df_agg = df.groupby("month").agg({"Rainfall": "sum"})
df_agg.idxmax()

# Aufgabe 5:
df["year"] = df.index.year
df.groupby("year").agg({"MaxTemp": "mean", "Temp9am": "mean"})
