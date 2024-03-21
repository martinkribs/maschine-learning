import pandas as pd

# einfachste Datenstruktur Series
mySerie = pd.Series(
    [2, 1, 4, 2, 1], index=["math", "english", "geogra.", "chemistry", "sports"]
)
print(mySerie)
print("Chemienote: ", mySerie["chemistry"])

# DataFrame
myFrame = pd.DataFrame([[1, 2], [3, 4], [1, 4], [3, 2]], columns=["s1", "s2"])
print()
print("DataFrame")
print(myFrame)

# Zugriff auf Spalten und Zeilen
print("Spaltenzugriff")
print("Mittelwert von S1: ", myFrame["s1"].mean().round(2))

print("\nZeilenzugriff: ")
print(myFrame["s1"].iloc[1:3])

# Einlesen der Datei
file = pd.read_csv("./DataSet/data_health.csv", sep=";", decimal=".")
print(file.head(10))

# Auswahl für Merkmale
# [[]] ist DataFrame
file1 = file[["Duration", "Pulse"]]
# [] ist Series
file2 = file["Maxpulse"]
print()
print(file1.head(10))
print()
print(file2.head(10))

count = 0
for i in file["Duration"]:
    if i >= 60:
        count += 1

print("Anzahl der Werte über 60: ", count)

# Datenvorverarbeitung
# Anzahl Leerzeilen
print("\n Anzahl Leerzeilen:\n", file.isnull().sum())
