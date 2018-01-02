from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
import csv
import numpy as np
from datetime import datetime as dt
import time

start = time.time()

FILE = list(csv.reader(open('transport_data.csv', 'r')))

data = []
target = []
test = []

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=90, max_features=None, n_jobs=-1),
    AdaBoostClassifier(),
    ExtraTreesClassifier()]

names = ["Nearest Neighbors", "Decision Tree", "Random Forest", "AdaBoost", "ExtraTreesClassifier"]


for i in range(1, len(FILE)):
    if FILE[i][4] != "-" and FILE[i][4] != "?":
        if 0 <= int(FILE[i][4]) <= 2:
            time = float(dt.fromtimestamp(int(FILE[i][3])).hour) + float(dt.fromtimestamp(int(FILE[i][3])).minute) / 60. \
                   + float(dt.fromtimestamp(int(FILE[i][3])).second) / 3600.
            data.append([float(FILE[i][0]), float(FILE[i][1]), time])
            target.append(FILE[i][4])
    if FILE[i][4] == "?":
        time = float(dt.fromtimestamp(int(FILE[i][3])).hour) + float(dt.fromtimestamp(int(FILE[i][3])).minute) / 60. \
               + float(dt.fromtimestamp(int(FILE[i][3])).second) / 3600.
        test.append([float(FILE[i][0]), float(FILE[i][1]), time])

b = True

for name, clf in zip(names, classifiers):
    clf.fit(data, target)
    print name + " traind"
    if b == True:
        predictions = clf.predict(test)
        b = False
    else:
        predictions = np.vstack((predictions,clf.predict(test)))
    print name + " predicted"

np.savetxt('Predictions.txt', predictions, fmt="%s")

prediction = []
for i in range(len(predictions[0])):
    print i
    predLab0 = 0
    predLab1 = 0
    predLab2 = 0
    for j in range(len(predictions)):
        # print v[j][i]
        if predictions[j][i] == "0":
            predLab0 = predLab0 + 1
        if predictions[j][i] == "1":
            predLab1 = predLab1 + 1
        if predictions[j][i] == "2":
            predLab2 = predLab2 + 1

    print(predLab0, predLab1, predLab2)
    if i==0:
        if ((predLab0 >= predLab1) and (predLab0 >= predLab2)):
            prediction = [0]
        else:
            if ((predLab1 >= predLab0) and (predLab1 >= predLab2)):
                prediction = [1]
            else:
                prediction = [2]
    else:
        if ((predLab0 >= predLab1) and (predLab0 >= predLab2)):
            prediction = prediction + [0]
        else:
            if ((predLab1 >= predLab0) and (predLab1 >= predLab2)):
                prediction = prediction + [1]
            else:
                prediction = prediction + [2]


np.savetxt('Prediction.txt', prediction, fmt="%s")

print("Done")
