import pandas as pd

df = pd.read_csv("daten/weatherAUS.csv")

df = df.dropna()

df.RainToday.unique()

# rain_mapping = {"No": 0, "Yes": 1}
# df["RainToday"] = df["RainToday"].map(rain_mapping)
# df["RainTomorrow"] = df["RainTomorrow"].map(rain_mapping)

# alternative:
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(df["RainToday"])
df["RainToday"] = le.transform(df["RainToday"])
df["RainTomorrow"] = le.transform(df["RainTomorrow"])

risk_mm = df["RISK_MM"]
y = df["RainTomorrow"]

df = df.drop(columns=["RISK_MM", "RainTomorrow", "Date", "WindGustDir", "WindDir9am", "WindDir3pm"])

df = pd.get_dummies(df)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.4, random_state=42, stratify=y)

x_test, x_vali, y_test, y_vali = train_test_split(x_test, y_test, test_size=0.75, random_state=42, stratify=y_test)
