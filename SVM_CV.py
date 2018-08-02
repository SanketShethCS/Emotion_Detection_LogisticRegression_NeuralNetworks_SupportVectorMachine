

#Author: Sanket Sheth (sas6792@g.rit.edu)

import numpy as np
from sklearn import svm #importing sci-kit learn's svm classifier


def getBinaryfer13Data(filename):
    '''
    Helper function used for converting input files into input data
    '''
    Y = []
    X = []
    first = True
    for line in open(filename, 'r'):
        if first:
            first = False
        else:
            row = line.split(',')
            y = int(row[0])
            type = row[2]
            if y == 3 or y == 4:	#3=happy; 4=sad
                Y.append(abs(y-4))  #we want to store 1 for happy, 0 for sad
                X.append([int(p) for p in row[1].split()])
    return np.array(X) / 255.0, np.array(Y)


def svmClassifier():
    '''
    This function is the implementation of a svm binary classsifier that 
    imports input facial data to predict emotion using image pixel as features.
    '''
    filename="fer3and4train.csv"
    X,Y=getBinaryfer13Data(filename) #importing trainin data
    clf=svm.SVC()
    print("Training:")    
    clf.fit([p for p in X], Y) #trainig the data and features
    print("Testing:")
    testfile="fer3and4test.csv"
    X,Y=getBinaryfer13Data(testfile) #importing testing data
    print(clf.score([p for p in X],Y)) #Accuracy based on training data for test data 
    
svmClassifier()