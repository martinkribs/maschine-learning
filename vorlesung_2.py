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
