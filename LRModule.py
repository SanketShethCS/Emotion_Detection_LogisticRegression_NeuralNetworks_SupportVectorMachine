

#Author: Sanket Sheth (sas6792@g.rit.edu)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from helper import getBinaryfer13Data, sigmoid, sigmoid_cost, error_rate
import random



class LRModule(object):
    def __init__(self):
        pass
    

    def train(self, X, Y, step_size=10e-7, epochs=10000):
        '''
        Training function used to train a logistic regression model to given data
        '''
        X, Y = shuffle(X, Y)
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        X, Y = X[:-1000], Y[:-1000]
        N, D = X.shape
        W=[random.uniform(0,1) for i in range(2304)] #assigning random initial weights
        W=np.array(W) #converting to numpy array
        b=0 #initialising bias
        costs = []
        best_validation_error = 1
        error=[]
        for i in range(epochs): #for the given number of epochs
                print("Epoch: ",i)
                pY=self.forward(X,W,b) #forward propogation error for the training data for given epoch
                W-=step_size*X.transpose().dot((np.subtract(pY , Y))) #updating the weights based on preddictions
                b-=step_size*(np.subtract(pY , Y)) #updating bias based on predicted values
                b=np.mean(b) 
                pValid=self.forward(Xvalid,W,b) #forward propogation for validation set
                ans=[]
                for v in pValid: #for value in validation set
                    if v > 0.5: #normalising to calss label 0
                        ans.append(1)
                    else: #normalising to class label 1
                        ans.append(0)
                pValid=np.array(ans) 
                sc=sigmoid_cost(Yvalid,pValid) #error cost of prediction
                costs.append(sc)
                e=error_rate(Yvalid,pValid) #error rate for prediction
                if e < best_validation_error: #finding best error rate
                        best_w=W
                        best_b=b
                        best_validation_error=e
                error.append(e) #error over time
        plt.plot(error) #plotting change in error over time
        print(best_validation_error)
        return best_w,best_b
            
    def forward(self, X,W,b):
        '''
        Forward propogation function for logistic regression
        '''
        temp=X.dot(W.transpose()) + b
        pY=sigmoid(temp)
        return pY
            
    def predict(self, X,W,b):
        '''
        Prediction function for LR model
        '''
        pY=self.forward(X,W,b) #predict labels for test set based on learned weights
        ans=[]
        for v in pY: #normalise the predicted values
         if v > 0.5:
                        ans.append(1)
         else:
                        ans.append(0)
         pY=np.array(ans)
         return pY

    def score(self, X, Y):
        '''
        Score function to calculate accuracy for test predictions
        '''
        Accuracy=(1-error_rate(Y,X))*100
        print(Accuracy)
    
def main():
    filename="fer3and4train.csv"
    X,Y=getBinaryfer13Data(filename) #importing train data
    clf=LRModule()
    w,b=clf.train(X,Y) #training
    testfile="fer3and4test.csv"
    X,Y=getBinaryfer13Data(testfile) #importing test data
    p=clf.predict(X,w,b) #test predictions
    clf.score(p,Y) #Test prediction accuracy
    
if __name__ == '__main__':
    main()
        