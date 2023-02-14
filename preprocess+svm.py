from os import removedirs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print('setup Completed^___^')

#Read the data
data = pd.read_csv("drugsComTrain_raw.csv")
data.head()
data.isnull().sum()
data.dropna(axis=0, inplace=True)
data.drop(['uniqueID', 'condition', 'date','usefulCount'], axis=1, inplace=True)
data.tail()
data.shape

#Make the data a bit smaller
data = data[data.groupby('drugName')['drugName'].transform('size') > 20]
data = data.head(10000)

#preprocessing
print('the review column data types is:',data['review'].dtypes)
data['review'] = data['review'].astype(str)

#Converting to lowerCase
data['review1'] = data['review'].apply(lambda x: " ".join(x.lower() for x in x.split()))
print("\n1.converted to lower case.\n")

#Removing Punctuations
data['review1'] = data['review1'].str.replace('[^\w\s]', '')
print("\n2.removed the punctuations already!\n")

#Removing StopWords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')

data['review1'] = data['review1'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
data['review1'].head()
print("\n3.removed the stopwords already!\n")

#Remove the Rare Words
freq = pd.Series(' '.join(data['review1']).split()).value_counts()
less_freq = list(freq[freq == 1].index)
data['review1'] = data['review1'].apply(lambda x: " ".join(x for x in x.split() if x not in less_freq))
data['review1'].head()
print("\n4.removed the rare words already!\n")

data.to_csv("removed_rare_data10000.csv",index=False)

#Stemming and lemmatization
from textblob import TextBlob, Word, Blobber
from nltk.stem import PorterStemmer
st = PorterStemmer()

data['review1'] = data['review1'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))

data['review1'] = data['review1'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
data['review1'].head()

data['review_len'] = data['review'].astype(str).apply(len)
data['word_count'] = data['review'].apply(lambda x: len(str(x).split()))

data['polarity'] = data['review1'].map(lambda text: TextBlob(text).sentiment.polarity)
print("\n5.Stemming and lemmatization finished!\n")

ax1 = data[['review_len', 'word_count','polarity', 'rating']].hist(bins=20, figsize=(15, 10), color='firebrick', alpha=0.9)

#Rating VS Polarity
plt.figure(figsize=(10, 8))
sns.set_style('white')
sns.set(font_scale= 1.5)
sns.boxplot(x= 'rating', y='polarity', data=data)
plt.xlabel('Rating')
plt.ylabel('Polarity')
plt.title('Ratings vs Polarity')
plt.savefig("./fig2.png",dpi=300)
plt.show()

mean_pol = data.groupby('rating')['polarity'].agg([np.mean])
mean_pol.columns = ['mean_polarity']
fig, ax = plt.subplots(figsize=(8, 6))
plt.bar(mean_pol.index, mean_pol.mean_polarity, width=0.3)
for i in ax.patches:
    ax.text(i.get_x(), i.get_height()+0.01, str("{:.2f}".format(i.get_height())))
    plt.title("Polarity of Ratings", fontsize=18)
plt.ylabel("Polarity", fontsize=16)
plt.xlabel("Rating", fontsize=16)
plt.ylim(0, 0.35)
plt.savefig("./fig3.png",dpi=300)
plt.show()

plt.figure(figsize=(8, 6))
sns.set_style('white')
ax = sns.countplot(x="rating", data=data, palette="Set3")
plt.savefig("./fig4.count_plot.png",dpi=300)
plt.show()

plt.figure(figsize=(10, 6))
sns.pointplot(x = 'rating', y='review_len', data=data)
plt.xlabel('Rating')
plt.ylabel('Reviwe Length')
plt.title('Drug Condition Rating VS Rreview Length')
plt.savefig("./fig5_Drug_Condition_Rating_VS_Rreview_Length.png",dpi=300)
plt.show()

#top 30 drugs
condition_pol = data.groupby('drugName')['polarity'].agg([np.mean])
condition_pol.columns = ['polarity']
condition_pol = condition_pol.sort_values('polarity', ascending=False)
condition_pol = condition_pol.head(30)

#WordCloud
text = " ".join(review for review in data.review1)

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
stopwords = set(STOPWORDS)
stopwords = stopwords.union(["ha", "thi", "now", "onli", "im", "becaus", "wa", "will", "even", "go", "realli", "didnt", "abl"])
wordcl = WordCloud(stopwords = stopwords, background_color='white', max_font_size = 50, max_words = 5000).generate(text)
plt.figure(figsize=(14, 12))
plt.imshow(wordcl, interpolation='bilinear')
plt.axis('off')
plt.savefig("./fig6_wordcloud.png",dpi=300)
plt.show()

#plot Frequency Charts
from sklearn.feature_extraction.text import CountVectorizer
def get_top_n_words(corpus, n=None):
    vec=CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_words(data['review1'], 20)
df1 = pd.DataFrame(common_words, columns = ['Review', 'count'])
df1.head()

ax1 = df1.groupby('Review').sum()['count'].sort_values(ascending=False).plot(kind='bar',color='firebrick',figsize = (12, 6))
xlabel = 'Top Words'
ylabel = 'Count'
title = 'BarChart represent the Top Words Frequency'
fig1 = ax1.get_figure()
fig1.savefig("./fig7_topwords.png",dpi=300) #有问题
fig1.show()

#two consecutive words or three consecutive words are more helpful
def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2,2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words2 = get_top_n_bigram(data['review1'], 30)
df2 = pd.DataFrame(common_words2, columns=['Review', "Count"])
df2.head()

ax2 = df2.groupby('Review').sum()['Count'].sort_values(ascending=False).plot(kind='bar',figsize=(12,7), color='firebrick')
xlabel = "Bigram Words"
ylabel = "Count"
title = "Bar chart of Bigrams Frequency"
fig2 = ax2.get_figure()
fig2.tight_layout()
fig2.savefig("./fig8_topphrase.png",dpi=300)
fig2.show()

def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words3 = get_top_n_trigram(data['review1'], 30)
df3 = pd.DataFrame(common_words3, columns = ['Review' , 'Count'])
ax3 = df3.groupby('Review').sum()['Count'].sort_values(ascending=False).plot(kind='bar',figsize=(12,9), color='firebrick')
xlabel = "Trigram Words"
ylabel = "Count"
title = "Bar chart of Trigrams Frequency"
fig3 = ax3.get_figure()
fig3.tight_layout()
fig3.savefig("./fig9_Bar_chart_of_Trigrams_Frequency.png",dpi=300)
fig3.show()

##Sentiment Analysis
data.head()
data.rating.describe()
data.rating.value_counts()
# Remove any Neutral ratings equal to 3 :
data = data[data['rating'] != 3]
data['Positively Rated'] = np.where(data['rating'] > 3, 1, 0)
data.head(10)

data['Positively Rated'].mean()

#train set & test set spilting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data['review1'], data['Positively Rated'], random_state = 0)

print('X_train first entry:\n\n', X_train.iloc[0])
print('\n\nX_train shape: ', X_train.shape)

#CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
# fit the countvectorizer to the training data:
vect = CountVectorizer().fit(X_train)
vect.get_feature_names()[:2000]
len(vect.get_feature_names())

X_train_vectorized = vect.transform(X_train)


#svm二分类
from sklearn import svm
from sklearn.metrics import accuracy_score
#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel
#Train the model using the training sets
clf.fit(X_train_vectorized, y_train)
#Predict the response for test dataset
predictions = clf.predict(vect.transform(X_test))

print("SVM Accuracy:", accuracy_score(y_test, predictions))
#svm accuracy:0.8177257525083612

#svm多分类
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.metrics import accuracy_score
#Create a svm Classifier
X_train, X_test, y_train, y_test = train_test_split(data['review1'], data['Rating grade'], random_state = 0)
vect = CountVectorizer().fit(X_train)
vect.get_feature_names()[:2000]
X_train_vectorized = vect.transform(X_train)
X_test_vectorized = vect.transform(X_test)
clf1 = svm.SVC(decision_function_shape='ovr',max_iter=10000,C=0.1)
#Train the model using the training sets
clf1.fit(X_train_vectorized, y_train)
#Predict the response for test dataset
predictions = clf1.predict(vect.transform(X_test))

model_test_pred=clf1.predict(X_test_vectorized)
model_train_pred=clf1.predict(X_train_vectorized)
print(classification_report(model_test_pred,y_test))

print("SVM'accuracy on training set:{:.3f}".format(clf1.score(X_train_vectorized,y_train)))
print("SVM'classifier'accuracy on test set:{:.3f}".format(clf1.score(X_test_vectorized,y_test)))
