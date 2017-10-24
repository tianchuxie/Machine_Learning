import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import svm
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier

def train(data):
    try:
        csvdata = np.loadtxt(data, delimiter = ',')
        X = np.delete(csvdata, 57, axis = 1)
        y = csvdata[: , 57]
    except:
        print('Error reading data')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.33, random_state = 5)
    X = preprocessing.scale(X_train)
    reg = KNeighborsClassifier(n_neighbors=9, algorithm='ball_tree')
    reg = reg.fit(X, y_train)
    XV = preprocessing.scale(X_val)
    y_pred = reg.predict(XV)
    print(accuracy_score(y_val, y_pred))
    print(confusion_matrix	(y_val, y_pred))
    return reg

def predict(C, p):
    pred = C.predict(p)
    return pred
    
def main():
    C = train('train.csv')
    testdata = np.loadtxt('ts4.csv', delimiter = ',')
    TD = preprocessing.scale(testdata)
    result = predict(C, TD)
    #cmap = {0: 'red', 1:'yellow'}
    #colors = [cmap[x] for x in result]
    #plt.figure(figsize = (20, 15))
    #for i in range (8, 12):
    #    for j in range(8, 12):
    #        plt.subplot(4, 4, (i - 8) * 4 + (j - 8) + 1)
    #        plt.scatter(testdata[:, i], testdata[:, j], c = colors)
    #        plt.xlabel(i)
    #        plt.ylabel(j)
    #plt.show()
    #print(result)
    presult = C.predict_proba(TD)
    result = result[np.newaxis].T
    np.savetxt('pred_4.csv', result, delimiter = ',')
    np.savetxt('prob_4.csv', presult, delimiter = ',')
if __name__ == '__main__':
    main()
