import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Datei einlesen und Spalten umbenennen
file = pd.read_csv("DataSet/Iris.csv")

# Löschen Spalte ID
del file["Id"]

# Vorverarbeitung
print(file.isnull().sum())

# Label - Features
y = file["Species"]
del file["Species"]
X = file
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Training des Modells
tree = DecisionTreeClassifier().fit(X_train, y_train)

# Vorhersage auf Testdatensatz
y_predict = tree.predict(X_test)
y_predict2 = tree.predict(X_train)

# Evaluation
print("\nKonfusionsmatrix:")
print(confusion_matrix(y_test, y_predict))
print("\nAbgeleitete Metriken:")
print(classification_report(y_test, y_predict))
print("\nOverfitting:")
print(classification_report(y_train, y_predict2))

# Visualisierung
plt.figure(figsize=(12, 8))
plot_tree(tree, feature_names=X.columns, class_names=y.unique(), filled=True)
plt.show()
