# Initialisierung (Beispiel Perceptron Algorithmus)
w = [1, 1, 1]
n = 0.5
c = [1, 1, 0, 0]
x = [[1, 0, 1.8], [1, 2, 0.6], [1, -1.2, 1.4], [1, 0.4, -1]]
counter = 0


# Funktion des Classifiers
def klassifier(w, x):
    return w[0] * x[0] + w[1] * x[1] + w[2] * x[2]


# tabellarische Ausgabe
def printVektor(w):
    for i in range(len(w)):
        print(round(w[i], 2), " | ", end=" ")


# Durchführung des Lernalgorithmus
def learningAlgorithm(x, w, n):
    for i in range(len(x)):
        h = klassifier(w, x[i])
        if h > 0:  # was muss hier hin?
            h = 1
        else:  # was muss hier hin?
            h = 0
        for j in range(len(w)):
            w[j] = w[j] + n * (c[i] - h) * x[i][j]
        printVektor(w)
        print()


def stoppingCriteria():
    count = 0
    for i in range(len(x)):
        h = klassifier(w, x[i])
        if h > 0:  # was muss hier hin?
            h = 1
        else:  # was muss hier hin?
            h = 0
        if c[i] != h:
            count += 1
    return count


# Hauptprogramm)
while stoppingCriteria() > 0:
    learningAlgorithm(x, w, n)
