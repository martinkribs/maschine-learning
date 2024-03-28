import pandas as pd

# Einlesen der Datei
file = pd.read_csv("./DataSet/data_health.csv", sep=";", decimal=".")
print(file.head(10))

# Datenvorverarbeitung
# Anzahl Leerzeilen
print("\n Anzahl Leerzeilen:\n", file.isnull().sum())

# Verfahren zum Füllen der Leerzeilen
file.dropna(axis=0,how="all",inplace=True)
print(file.head)