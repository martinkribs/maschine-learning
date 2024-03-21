import math as m
from statistics import mean, stdev, median
import numpy as np

print("Hallo World")
# eingabe = input("Wie heisst Du?\n")
# print("Herzlich Willkommen", eingabe, "!")
# alter = int(input("Wie alt bist du?"))
# print(alter + 3)
# print(type(alter))

# Operatoren
x = 7 // 3
print(x)
print("Rest ", 5 % 3)
print("Exponential: ", x**2)
# Runden von Werten
y = 12 / 7
print("gerundet: ", round(y, 3))

# logische Operator
print(x == 2 and y != 1)
print(m.pow(x, 2))

# Kontrollstrukturen if else
geschlecht = "m"
alter = 17

if geschlecht == "m" and alter >= 18:
    print("Sie dürfen den Führerschein machen!")
else:
    print("Sie müssen noch warten")

# Schleifen
zahl = 1
while zahl < 5:
    print("Zahl: ", zahl)
    zahl += 1

print()
liste = [1, 2, 3, 4, 5]
for x in liste:
    print(x)

print("neue Ausgabe:")
for i in range(1, 4):
    print(liste[i])

print("Länge:", len(liste))

# statistische Auswertungen
print("Mittelwert: ", mean(liste))
print("Standabw: ", stdev(liste).__round__())
print(type(liste))

# Konvertierung Liste in Numpy-Array
arr = np.array(liste)
print("Array: ", arr)
print(type(arr))

print("\n Mittelwert: ", arr.mean())