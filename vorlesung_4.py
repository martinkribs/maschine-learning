import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

# Einlesen der Datei
file = pd.read_csv("./DataSet/data_health.csv", sep=";", decimal=".")
print(file.head(10))

# Datenvorverarbeitung
# Anzahl Leerzeilen
print("\n Anzahl Leerzeilen:\n", file.isnull().sum())

# Verfahren zum Füllen der Leerzeilen
file.dropna(axis=0, how="all")

file["Duration"] = file["Duration"].fillna(45)

file["Pulse"] = file["Pulse"].ffill()

mw = file["Maxpulse"].median()
file["Maxpulse"] = file["Maxpulse"].fillna(mw)

mw = file["Calories"].mean().round(1)
print("Mittelwert 2: ", mw)
file["Calories"] = file["Calories"].fillna(mw)

# Berechnung der Korrelation
print(file.corr().round(2))

# Lineare Regression zwischen Calories und Duration → weil höchste Korrelation
# Preprocessing durch MinMaxScaler()
# Überführung in numpy Array
X = file[["Duration", "Maxpulse"]].values
Y = file[["Calories"]].values

scale = preprocessing.MinMaxScaler()
X = scale.fit_transform(X)
Y = scale.fit_transform(Y)

xDF = pd.DataFrame(X)
yDF = pd.DataFrame(Y)
print(xDF.describe())

# manuelle Einteilung in Trainings- und Testdatensatz 0-130 : 130-EOF
# X = file[["Duration"]].iloc[:130]
# print(type(X))

# X = file[["Duration", "Maxpulse"]].iloc[:130].values
# print(type(X))
# Y = file[["Calories"]].iloc[:130].values

# X_test = file[["Duration", "Maxpulse"]].iloc[130:].values
# Y_test = file[["Calories"]].iloc[130:].values

X = xDF.iloc[:130].values
Y = yDF.iloc[:130].values

X_test = xDF.iloc[130:].values
Y_test = yDF.iloc[130:].values

# Trainieren des Modells
model = LinearRegression().fit(X, Y)

# Ausgabe des Modells
print("w_0: ", model.intercept_)
print("w_1", model.coef_)

# Evaluation des Modells
print("R²: ", model.score(X, Y), "\n")

# Evaluation des Testdatensatzes
print("\nAusgabe des Residuen")
# print(model.predict(X_test) - Y_test)
print("R² des Testdatensatzes: ", model.score(X_test, Y_test), "\n")
