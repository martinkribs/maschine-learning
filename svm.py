import pandas as pd
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Datei einlesen und Spalten umbenennen
file = pd.read_csv(
    "DataSet/data_diagnosis.csv",
    sep=";",
    header=0,
    names=["dia", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"],
)

# Prüfung auf Leerzellen und Vorverarbeitung
print(file.isnull().sum())

# Label - Features
X = file.iloc[:, 1:]
y = file["dia"]

# Trainings- und Testdatensatz
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Training SVM
model = svm.SVC(kernel="linear").fit(X_train, y_train)

# Vorhersage auf Testdatensatz
y_predict = model.predict(X_test)
y_predict2 = model.predict(X_train)

# Evaluation Testdatensatz
print("\nKonfusionsmatrix:")
print(confusion_matrix(y_test, y_predict))
print("\nAbgeleitete Metriken:")
print(classification_report(y_test, y_predict))

# Evaluation Trainingsdatensatz
print("\nKonfusionsmatrix:")
print(confusion_matrix(y_train, y_predict2))
print("\nAbgeleitete Metriken:")
print(classification_report(y_train, y_predict2))
