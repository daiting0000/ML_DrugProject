import random
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from textblob import TextBlob, Word, Blobber

import numpy as np
from numpy import unique
from numpy import asarray
from numpy import argmax

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import tensorflow as tf
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten

test = pd.read_csv('E:/PythonProject/machine_learning/MLproject/drugsComTest_raw.csv')
data = pd.read_csv('E:/PythonProject/machine_learning/MLproject/drugsComTrain_raw.csv')
# print(data.isnull().sum())

# 去除我们不关心的内容
data.dropna(axis=0, inplace=True)
data.drop(['uniqueID', 'condition', 'date', 'usefulCount'], axis=1, inplace=True)
# print(data.shape)

# 减小训练集
data = data[data.groupby('drugName')['drugName'].transform('size') > 20]
data = data.head(10000)

# print('the review column data types is:', data['review'].dtypes)
data['review'] = data['review'].astype(str)

# 大写转换为小写
data['review1'] = data['review'].apply(lambda x: " ".join(x.lower() for x in x.split()))
# print(data['review1'].head())

# 去除符号
data['review1'] = data['review1'].str.replace('[^\w\s]', '')
# print(data['review1'].head())

# 去除stopwords
stop = stopwords.words('english')
data['review1'] = data['review1'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
# print(data['review1'].head())

# 去除生僻词
freq = pd.Series(' '.join(data['review1']).split()).value_counts()
less_freq = list(freq[freq == 1].index)
data['review1'] = data['review1'].apply(lambda x: " ".join(x for x in x.split() if x not in less_freq))
# print(data['review1'].head())

st = PorterStemmer()

data['review1'] = data['review1'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
data['review1'] = data['review1'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
# print(data['review1'].head())

data['review_len'] = data['review'].astype(str).apply(len)
data['word_count'] = data['review'].apply(lambda x: len(str(x).split()))

data['polarity'] = data['review1'].map(lambda text: TextBlob(text).sentiment.polarity)
# print(data.head())

# print(data.rating.describe())
# print(data.rating.value_counts())

# 根据rating分为1，2，3，4，5这五个等级
def function(x):
    if x <= 2:
        y = 1
    elif x <= 4:
        y = 2
    elif x <= 6:
        y = 3
    elif x <= 8:
        y = 4
    else:
        y = 5
    return y
data['Rating grade'] = data['rating'].apply(lambda x: function(x))

# # Remove any Neutral ratings equal to 3
# data = data[data['rating'] != 3]
# data['Positively Rated'] = np.where(data['rating'] > 3, 1, 0)
# # print(data.head(10))

data['Rating grade'] = data['Rating grade']-1



# 对测试集的文本进行同样的预处理
test['review'] = test['review'].astype(str)
test['review1'] = test['review'].apply(lambda x: " ".join(x.lower() for x in x.split()))
test['review1'] = test['review1'].str.replace('[^\w\s]', '')
test['review1'] = test['review1'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

freq_test = pd.Series(' '.join(test['review1']).split()).value_counts()
less_freq_test = list(freq_test[freq_test == 1].index)
test['review1'] = test['review1'].apply(lambda x: " ".join(x for x in x.split() if x not in less_freq_test))

test['review1'] = test['review1'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
test['review1'] = test['review1'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

test['review_len'] = test['review'].astype(str).apply(len)
test['word_count'] = test['review'].apply(lambda x: len(str(x).split()))

test['polarity'] = test['review1'].map(lambda text: TextBlob(text).sentiment.polarity)

# 根据rating分为1，2，3，4，5这五个等级
def function(x):
    if x <= 2:
        y = 1
    elif x <= 4:
        y = 2
    elif x <= 6:
        y = 3
    elif x <= 8:
        y = 4
    else:
        y = 5
    return y
test['Rating grade'] = test['rating'].apply(lambda x: function(x))

# test = test[test['rating'] != 3]
# test['Positively Rated'] = np.where(test['rating'] > 3, 1, 0)

test['Rating grade'] = test['Rating grade']-1


# nn
x_train, y_train = data['polarity'], data['Rating grade']
x_test, y_test = test['polarity'], test['Rating grade']

# fix the random seed
random.seed(2)
np.random.seed(2)
tf.random.set_seed(seed=2)

# reshape data to have a single channel
x_train = x_train.values.reshape((x_train.shape[0], 1))
x_test = x_test.values.reshape((x_test.shape[0], 1))

# determine the shape of the input images
in_shape = x_train.shape[1:]

# determine the number of classes
n_classes = len(unique(y_train))
# print(in_shape, n_classes)

# normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# define model
model = Sequential()

# # Convolution layer with 8 3 by 3 filters, the activation is relu
# model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=in_shape))
# # Max pooling layer with 2 by 2 pooling window.
# model.add(MaxPool2D(pool_size=(2, 2)))

# # Flatten layer
model.add(Flatten())

# # First hidden layer with 100 hidden nodes
model.add(Dense(units=200, activation='relu'))

# # The output layer with 10 classes output.
# # Use the softmax activation function for classification
model.add(Dense(units=n_classes, activation='softmax'))

# define loss function and optimizer
# set the optimizer to 'sgd', then you may switch to 'adam'.
# use cross entropy as the loss for multi-class classification
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# fit the model
model.fit(x_train, y_train, epochs=12, batch_size=400, verbose=2, validation_data=(x_test, y_test))
model.fit(x_train, y_train, epochs=12, batch_size=400, verbose=2, validation_data=(x_test, y_test))
# evaluate the model on training set and test set

loss, acc = model.evaluate(x_train, y_train, verbose=0)
print('Test Accuracy on the training set: %.3f' % acc)
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print('Test Accuracy on the test set: %.3f' % acc)
