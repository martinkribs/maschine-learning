import pandas as pd
from sklearn.metrics import classification_report, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# binäres Problem
health = pd.read_csv("DataSet/data_diagnosis.csv", sep=";")
print(health.isnull().sum())
print(health.head(5))
print()

# tertiäres Problem
iris = pd.read_csv("DataSet/Iris.csv")
# print(iris.isnull().sum())
iris = iris.drop(columns=["Id"])
print(iris.head(5))
print()

# Regression diskret
abalone = pd.read_csv("DataSet/abalone.csv")
# print(iris.isnull().sum())
abalone = abalone.drop(columns=["Sex"])
print(abalone.head(5))
print()

# Einteilung feature und label
y_h = health["diagnosis"]
X_h = health.drop(columns=["diagnosis"])

y_i = iris["Species"]
X_i = iris.drop(columns=["Species"])

y_a = abalone["Rings"]
X_a = abalone.drop(columns=["Rings"])

# Training-Testsets
X_h_train, X_h_test, y_h_train, y_h_test = train_test_split(
    X_h, y_h, test_size=0.2, random_state=42
)
X_i_train, X_i_test, y_i_train, y_i_test = train_test_split(
    X_i, y_i, test_size=0.2, random_state=42
)
X_a_train, X_a_test, y_a_train, y_a_test = train_test_split(
    X_a, y_a, test_size=0.2, random_state=42
)

# Klassifier für Maligne/benigne
modell_h = KNeighborsClassifier(n_neighbors=11, p=3).fit(X_h_train, y_h_train)
h_pred = modell_h.predict(X_h_test)
print("Klassifikationsreport:\n")
print(classification_report(y_h_test, h_pred))

# Klassifier für Iris-Datensatz
modell_i = KNeighborsClassifier(n_neighbors=3).fit(X_i_train, y_i_train)
i_pred = modell_i.predict(X_i_test)
print("Klassifikationsreport:\n")
print(classification_report(y_i_test, i_pred))

# Klassifier für Abalone-Datensatz
modell_a = KNeighborsRegressor(n_neighbors=3, p=1).fit(X_a_train, y_a_train)
a_pred = modell_a.predict(X_a_test)
print("RMSE: ", root_mean_squared_error(y_a_test, a_pred).round(2))
