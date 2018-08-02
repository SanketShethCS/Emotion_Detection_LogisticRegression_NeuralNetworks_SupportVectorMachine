

#Author: Sanket Sheth (sas6792@g.rit.edu)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from helper import y2indicator, getBinaryfer13Data, sigmoid, sigmoid_cost, error_rate, softmax
import random


class NNModule(object):
    def __init__(self):
        pass
    
    def train(self, X, Y, step_size=10e-7, epochs=10000):
        '''
        Training function used to train a neural network model to given data
        '''
        X, Y = shuffle(X, Y)
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        X, Y = X[:-1000], Y[:-1000]   
        K = len(set(Y))        
        Y = y2indicator(Y, K)
        Yvalid = y2indicator(Yvalid, K)
        M, D = X.shape
        m1=5 #number of neurons for the hidden layer
        W1=[[random.uniform(0,1) for j in range(m1)] for i in range(D)] #initial weights for first layer 
        W1=np.array(W1)
        b1=0 #initial first layer bias
        W2=[[random.uniform(0,1) for j in range(2)] for i in range(m1)] #initial weights for output layer
        W2=np.array(W2)
        b2=0 #bias for output layer
        train_costs = []
        valid_costs = []
        best_validation_error = 1
        errorV=[] 
        errorT=[]
        for i in range(epochs): #for the given number of epochs
            print("Epoch: ",i)
            pY,z=self.forward(X,W1,W2,b1,b2) #forward propogation for train data
            W2=np.subtract(W2,step_size*(sigmoid((z.transpose()).dot((np.subtract(pY,Y)))))) #updating w2 by gradient descent
            b2=b2-step_size*(np.subtract(pY,Y)) #updating bias
            s=0
            s=s+sum([b for b in b2])
            b2=s
            z=z.transpose().dot(z)
            j=((np.subtract(pY,Y)).dot(W2.transpose())).dot((np.subtract(1,(z))).transpose())     #calculating error in activation        
            W1=np.subtract(W1,step_size*(X.transpose()).dot(j)) #using error in activation for back propogation
            b1=b1-step_size*j #updatin output bias
            s=0
            s=s+sum([b for b in b2])
            b1=s
            
            Pans=[  [] for k in range(len(pY)) ]
            for k in range(len(pY)): #normalising the predicted labels according to class labels
                if pY[k][0] > 0.5:
                    Pans[k].append(0)
                else:
                    Pans[k].append(1)
                if pY[k][1] > 0.5:
                    Pans[k].append(0)
                else:
                    Pans[k].append(1)
            Pans=np.array(Pans)
            train_costs.append(sigmoid_cost(Y,Pans)) #error cost for prediction (train)
            et=error_rate(Y,Pans) #error in predictions
            errorT.append(et) #error in train set over time
            
            Pvalid,zValid=self.forward(Xvalid,W1,W2,b1,b2) #forward propogation for validtion set based on updated weights
            PVans=[  [] for k in range(len(Pvalid)) ]
            for k in range(len(Pvalid)): #normalising prediictions based on class labels
                if Pvalid[k][0] > 0.5:
                    PVans[k].append(0)
                else:
                    PVans[k].append(1)
                if Pvalid[k][1] > 0.5:
                    PVans[k].append(0)
                else:
                    PVans[k].append(1)
            PVans=np.array(PVans)
            valid_costs.append(sigmoid_cost(Yvalid,PVans)) #error cost for prediction (validation)
            eV=error_rate(Yvalid,PVans) #error in predictions
            if eV < best_validation_error: #finding the best validation error
                        best_w1=W1
                        best_w2=W2
                        best_b1=b1
                        best_b2=b2
                        best_validation_error=eV
            errorV.append(eV) #error in validation set over time
        plt.plot(errorT) #Plotting error in train set over time
        #plt.plot(errorV) #Uncomment this to plot error in validation set over time
        print(best_validation_error)
        return best_w1,best_w2,best_b1,best_b2
            

    def forward(self,X,W1,W2,b1,b2):
        '''
        Forward propogation function for Neural Network
        '''
        p1=(X).dot(W1)+b1 #Outputs the predictions from layer one
        Z1=sigmoid(p1)       #Sigmoid values from layer one
        p2 = Z1.dot(W2) + b2 #Outputs from hidden layer (layer two)
        pY=softmax(p2) #Final Predictions based on layer two input
        return pY,p1
    
    def predict(self, P_Y_given_X,W1,W2,b1,b2):
        '''
        Prediction function for LR model
        '''
        Pvalid,Z=self.forward(P_Y_given_X,W1,W2,b1,b2) #forward propogation prediction for test set
        PVans=[  [] for k in range(len(Pvalid)) ]
        for k in range(len(Pvalid)): #normalising test predictions as per class labels
                if Pvalid[k][0] > 0.5:
                    PVans[k].append(0)
                else:
                    PVans[k].append(1)
                if Pvalid[k][1] > 0.5:
                    PVans[k].append(0)
                else:
                    PVans[k].append(1)
        PVans=np.array(PVans)   
        return PVans
        
    def classification_rate(self, Y, P):
        '''
        Score function to calculate accuracy for test predictions
        '''
        Y = y2indicator(Y, 2)
        Accuracy=(1-error_rate(Y,P))*100
        print(Accuracy)
        
    def cross_entropy(self, T, pY):
        '''
        Error function to calculate error cost using cross entropy loss function
        '''
        e=sigmoid_cost(T, pY)
        return e

def main():
    filename="fer3and4train.csv"
    X,Y=getBinaryfer13Data(filename) #importing training data
    clf=NNModule()
    W1,W2,b1,b2=clf.train(X,Y) #training
    filenameTest="fer3and4test.csv"
    X,Y=getBinaryfer13Data(filenameTest) #importing testing data
    p=clf.predict(X,W1,W2,b1,b2) #predicting test labels
    clf.classification_rate(Y,p) #Accuracy of predictd test labels
     
    
if __name__ == '__main__':
    main()
   