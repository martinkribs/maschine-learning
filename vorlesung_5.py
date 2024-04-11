import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

# Datei einlesen und Spalten umbenennen
file = pd.read_csv(
    "DataSet/data_diagnosis.csv",
    sep=";",
    header=0,
    names=["dia", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"],
)
# print(file.shape)
# print(file.head(10))

# Prüfung auf Leerzellen und Vorverarbeitung
print(file.isnull().sum())

# Korrelationsmatrix und univariate Analyse zur kontrolle
X = file.iloc[:, 1:]
y = file["dia"]
print(X.corr().round(2), "\n")
# print(X.describe().round(2))

# MinMax-Scaler
X = MinMaxScaler().fit_transform(X)
X = pd.DataFrame(X)
print(X.describe().round(2))

# Automatisierte Einteilung in Trainings- und Testdatensatz
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, shuffle=True, random_state=42
)
# print("Größe:",X_train.shape)
# print(X_train.head(10))
print("Verhältnis Train M/B: ", y_train.value_counts())
print("Verhältnis Test M/B: ", y_test.value_counts())

# Training logistische Regression
model = LogisticRegression().fit(X_train, y_train)
# Testdatensatz fü Vorhersage
y_pred = model.predict(X_test)

# Ausgaben
print("\na_0= ", model.intercept_)
print("Koeff: ", model.coef_)
print("Ausgabe Vorhersage: \n")
print(y_pred)

# Evaluation
print("\nKonfusionsmatrix:")
print(confusion_matrix(y_test, y_pred))
