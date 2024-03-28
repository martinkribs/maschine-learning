import pandas as pd
from autoimpute.imputations import SingleImputer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 1. Importieren der .csv-Datei und Ausgabe der ersten 10 Zeilen
file = pd.read_csv("./DataSet/Auto_ml.csv", sep=";", decimal=",")
print(file.head(10))

# 2. Anzahl der NaN pro Spalte anzeigen
print("\nAnzahl der NaN pro Spalte:")
print(file.isnull().sum())

# 3. Löschen von Zeilen ohne einen einzigen Eintrag und Imputation fehlender Werte
file_cleaned = file.dropna(axis=0, how="all")
imputer = SingleImputer(strategy='median')
file_imputed = imputer.fit_transform(file_cleaned)

# 4. Univariate Analyse und Prüfung auf Ausreißer
print("\nUnivariate Analyse und Ausreißerprüfung:")
sns.boxplot(file_imputed, showfliers=True)
plt.show()

# 5. Ausgabe der Korrelationsmatrix
print("\nKorrelationsmatrix der numerischen Spalten:")
print(file_imputed.corr().round(2))

# 6. Standardisierung der Daten und erneute univariate Analyse
scaler = StandardScaler()
file_scaled = scaler.fit_transform(file_imputed)
file_scaled = pd.DataFrame(file_scaled, columns=file_imputed.columns)

print("\nUnivariate Analyse nach Standardisierung:")
sns.boxplot(file_scaled, showfliers=True)
plt.show()