import pandas as pd

# 1
file = pd.read_csv("./DataSet/Auto_ml.csv", sep=";")
print(file.head(10))

# 2
print()
print(file.isnull().sum())

# 3
print()
file.dropna(axis=0, how="all")

# 4
print()
file.median()

# 5
print()
file.corr().round(2)

# 6
print()

