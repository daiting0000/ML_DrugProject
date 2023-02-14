import numpy as np
import matplotlib.pyplot as plt
import matplotlib.markers as mks
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

class LogisticRegressionClass(object):
    def __init__(self, dataX, dataY, testX, testY, learningRate, loopNum):
        self.trainX = dataX
        self.trainY = dataY
        self.learningRate = learningRate
        self.loopNum = loopNum
        self.Y_Train = np.array([self.trainY]).T
        self.X_Train = np.column_stack((dataX, np.repeat(1, dataX.shape[0])))
        self.parameter = np.array([np.ones(self.X_Train.shape[1])]).T
        self.testX = testX
        self.testY = testY
        self.Y_Test = np.array([self.testY]).T
        self.X_Test = np.column_stack((testX, np.repeat(1, testX.shape[0])))
        self.WTX = None
        self.error = None
        self.derivative = None

    def GradientDecent(self):
        for i in range(self.loopNum):
            self.WTX = np.dot(self.X_Train, self.parameter)
            Sig_Wtx = SigmoidFunction(self.WTX)
            self.error = Sig_Wtx - self.Y_Train
            self.derivative = np.dot(self.X_Train.T, self.error)
            self.parameter -= self.learningRate * self.derivative
            self.show_training()
            plt.pause(0.03)
        plt.show()


    def Train_acc(self):
        predict_y = np.dot(self.X_Train, self.parameter)
        predict_y = SigmoidFunction(predict_y)
        predict_y = np.where(predict_y > 0.5, 1, 0)
        predict_y = predict_y ^ self.Y_Train
        acc = (self.X_Train.shape[0] - np.sum(predict_y)) / self.X_Train.shape[0]
        print('Train_acc= {:.2%}'.format(acc))

    def Test_acc(self):
        predict_y = np.dot(self.X_Test, self.parameter)
        predict_y = SigmoidFunction(predict_y)
        predict_y = np.where(predict_y > 0.5, 1, 0)
        predict_y = predict_y ^ self.Y_Test
        acc = (self.testX.shape[0] - np.sum(predict_y)) / self.testX.shape[0]
        print('Test_acc= {:.2%}'.format(acc))

    def show_training(self):
        Param = Para_manager(self.parameter)
        self.train_show(Param)


def SigmoidFunction(X):
    return 1. / (1. + np.exp(-X))


def Para_manager(Param):
    Param[0] /= -Param[1]
    Param[2] /= -Param[1]
    Param = np.array([Param[0], Param[2]])
    return Param

def TunePara(x, y, ax=None, m=None, **kw):
    if not ax:
        ax = plt.gca()
    sc = ax.scatter(x, y, **kw)
    if m is not None and len(m) == len(x):
        paths = []
        for marker in m:
            if isinstance(marker, mks.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mks.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


if __name__ == "__main__":
    filename = 'precossed_final.csv'
    data = pd.read_csv(filename)
    data.dropna(axis=0,inplace=True)
    data = data[data['rating'] != 3]
    data['Positively Rated'] = np.where(data['rating'] > 3, 1, 0)
    data['Positively Rated'].mean()
    
    data_X = data['review1']
    data_Y = data['Positively Rated']

    
    X_train, X_test, train_Y, test_Y = train_test_split(data_X, data_Y, test_size=0.3, random_state=123)
    vect = CountVectorizer().fit(X_train)
    vect.get_feature_names()[:2000]
    train_X = vect.transform(X_train)
    test_X = vect.transform(X_test)
    Scaler = StandardScaler().fit(train_X)
    train_X = Scaler.transform(train_X)
    test_X = Scaler.transform(test_X)

    LogisticX = LogisticRegressionClass(train_X, train_Y, test_X, test_Y, 0.01, 100)
    LogisticX.GradientDecent()
    LogisticX.Test_acc()