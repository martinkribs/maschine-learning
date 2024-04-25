import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

file = pd.read_csv("DataSet/Iris.csv")
print(file.head(10))

# Löschen Spalte ID
del file["Id"]
print(file.head(10))

# Vorverarbeitung
# print(file.isnull().sum())

# Einteilung Trainings- und Testdatensatz
y = file["Species"]
del file["Species"]
X = file
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)
print(X_train.head(10), "\n")
print(y_train.head(10), "\n")

# Training des Modells
model = LogisticRegression().fit(X_train, y_train)
print("vector b_0: ", model.intercept_)
print("Regressionskoeffizienten: ")
print(model.coef_.round(3))
