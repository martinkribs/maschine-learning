import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Einlesen der Datei
file = pd.read_csv("./DataSet/data_health.csv", sep=";", decimal=".")
print(file.head(10))

# Datenvorverarbeitung
# Anzahl Leerzeilen
print("\n Anzahl Leerzeilen:\n", file.isnull().sum())

# Verfahren zum Füllen der Leerzeilen
file.dropna(axis=0, how="all")
print(file.head(6))

file["Duration"] = file["Duration"].fillna(45)
print(file[20:25])

file["Pulse"] = file["Pulse"].ffill()

mw = file["Maxpulse"].median()
print("Median: ", mw)
file["Maxpulse"] = file["Maxpulse"].fillna(mw)

mw = file["Calories"].mean().round(1)
print("Mittelwert 2: ", mw)
file["Calories"] = file["Calories"].fillna(mw)

# Probeausgabe
print("Probe Leerzeilen: ")
print(file.isnull().sum(), "\n")

# univariate
print(file.median())

# einfache Methode
print(file.describe().round(1))

# Pairplot
# sns.pairplot(file)
# plt.show()

# Boxplot
# sns.boxplot(file, showfliers=True)
# plt.xlabel(["Duration"],["Puls"],["Maxpulse"],["Calories"])
# plt.show()

# Berechnung der
print()
print(file.corr(method="spearman").round(2))

# Heatmap
plt.title("Heatmap von Data Health")
sns.heatmap(file.corr(),annot=True)
plt.show()
