import pandas as pd

pd.set_option("display.max_columns", 10)

df = pd.read_csv("daten/BTC_daily.csv")

df.dtypes
df["Date"] = pd.to_datetime(df["Date"])

df = df.set_index("Date")

df = df.dropna()

# Selektionen
df["High"]
df[["Close", "High"]]

## attribute Access
df.High

type(df["High"])
type(df.High)
type(df)

df.columns

# Selektion einzelner Werte
df.loc["2014-09-17"]
df.loc["2014-09-17", "Open"]

df.iloc[0, 0]
df.iloc[0, 1]

df.index

mask = df["High"] > 70000
df.loc[mask, "High"]

df.loc[mask, :]

df.iloc[100:102,:]

# Analysen

# - Tiefpunkte insgesamt
niedrigster_kurs = df["Low"].min()
print("Der niedrigste Kurs war ", niedrigster_kurs)

# - Tag mit meistem Handel
### Option 1:
hoechste_volumen = df["Volume"].max()
mask = df["Volume"] == hoechste_volumen
tag = df.loc[mask].index[0]
print("Tag mit meistem Handel war ", tag)

### Option 2:
tag = df["Volume"].idxmax()
print("Tag mit meistem Handel war ", tag)

# - Tag mit größten Änderung zwischen Open und Close
df["diff"] = abs(df["Close"] - df["Open"])
df["diff"].idxmax()

# - Tiefpunkt pro Jahr
df["Year"] = df.index.year
df_agg = df.groupby("Year").agg({"Low": "min"})
df_agg

# - 1000€ Investition am Anfang, wieviel wäre das heute?
erste_tag = df.index.min()
kurs_am_ersten_tag = df.loc[erste_tag, "Open"]

anzahl_bitcoins = 1000 / kurs_am_ersten_tag

letzter_tag = df.index.max()
kurs_am_letzten_tag = df.loc[letzter_tag, "Close"]

wert = anzahl_bitcoins * kurs_am_letzten_tag
print("Die 1000€ wären heute ", wert, " wert")

# - bei täglicher Investition von 10€, wieviel wäre das heute?
df["invest"] = 10
df["anzahl_bitcoins"] = df["invest"] / df["Open"]

gesamt_zahl_bitcoin = df["anzahl_bitcoins"].sum()

wert = gesamt_zahl_bitcoin * kurs_am_letzten_tag
print("Bei 10€ täglich wären das heute ", wert, " wert")

df["Close"].min()
df["Close"].max()
df["Close"].std()
df["Close"].mean()
df["Close"].median()