import numpy as np
import pandas as pd
import nltk 
import re
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Step 1: Data Analysis 
#Create two separate dataframes for each data set 
data = pd.concat(map(pd.read_csv, ['train.csv', 'test.csv']))
#I am required to use a small sample of the data, since my computer literally can not allocate enough space for the models to use 1 million points of data 
data = data.sample(frac=0.01, random_state=1)

print(data.isna().sum())
#Since there are no missing values in the set, we are able to move on to text preprocessing


#Step 2: Text Preprocessing 
#In this set, I will make all words lowercase, remove handles and links, punctuation, and lemmate all words 
corpus = []

for index, row in data.iterrows(): 
    process = row['Text'] 
    #Makes all strings lowercase
    process = process.lower()
    #Removes twitter mentions from all strings 
    process = re.sub('@\S*', '', process)
    #Removes any embedded links in the tweet 
    process = re.sub('http\S*', '', process)
    
    #Given list of contractions to change 
    process = re.sub(r'won\'t', 'will not', process)
    process = re.sub(r'can\'t', 'can not', process)
    process = re.sub(r'n\'t', ' not', process)
    process = re.sub(r'\'re', ' are', process)
    process = re.sub(r'\'s', ' is', process)
    process = re.sub(r'\'d', ' would', process)
    process = re.sub(r'\'ll', ' will', process)
    process = re.sub(r'\'t', ' not', process)
    process = re.sub(r'\'ve', ' have', process)
    process = re.sub(r'\'m', ' am', process)
    
    #Removes any punctuation and numbers 
    process = re.sub('[^a-z" "]', '', process)
    process = re.sub(r'  ', ' ', process)
    
    #Appends all newly formed sentences to the corpus 
    corpus.append(process)
    

#Now that text has been processed, the corpus is ready for feature extraction 

#Step 3: Linguistic Feature Extraction 
#First: Bag-of-words (Using CountVectorizer given in sklearn)
#All this vectorizer does is count every word given in the corupus and places it in an array, which is exactly what bag-of-words is 
vectorizer = CountVectorizer()
bag_of_words = vectorizer.fit_transform(corpus)


#Second: tf*idf (Using given sklearn method)
tfIdf =  TfidfVectorizer()
tfIdfVector = tfIdf.fit_transform(corpus)


#Third: Word2Vec (Using given gensim method)
#I am using the defaults given for vector size and window to make the models run more efficiently 
word2Vec = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4)

# I actually looked up some of this code with help from ChatGPT. I could not find a way to convert the Word2Vec model to an array that 
# the scikit models could use, so I put it in an empty vector and then append if the word is found in the Word2Vec model
X = []
for sentence in corpus:
    sentence_vector = [word2Vec.wv[word] for word in sentence if word in word2Vec.wv]
    if sentence_vector:
        X.append(np.mean(sentence_vector, axis=0))
word2Vec = np.array(X)


#Step 4: Build sentiment classification model 
#Large data set would not let me iterate through enough data, so had to change the value to allow this large # 
#Each model is going to be ran with each linguistic extraction matrix 
lrbow = LogisticRegression(max_iter=2000000)
lrtf = LogisticRegression(max_iter=2000000)
lrwv = LogisticRegression(max_iter=2000000)
svcbow = SVC(max_iter=2000000)
svctf = SVC(max_iter=2000000)
svcwv = SVC(max_iter=2000000)
nbcbow = GaussianNB()
nbctf = GaussianNB()
nbcwv = GaussianNB()
rfcbow = RandomForestClassifier()
rfctf = RandomForestClassifier()
rfcwv = RandomForestClassifier()

#Generates data from the large, pre-processed set for all three linguistic extraction methods 
bowx_train, bowx_test, bowy_train, bowy_test = train_test_split(bag_of_words, data['Sentiment'])
tfx_train, tfx_test, tfy_train, tfy_test = train_test_split(tfIdfVector, data['Sentiment'])
wordx_train, wordx_test, wordy_train, wordy_test = train_test_split(word2Vec, data['Sentiment'])

#Trains all models 
lrbow.fit(bowx_train, bowy_train)
svcbow.fit(bowx_train, bowy_train)
nbcbow.fit(bowx_train.toarray(), bowy_train)
rfcbow.fit(bowx_train, bowy_train)

lrtf.fit(tfx_train, tfy_train)
svctf.fit(tfx_train, tfy_train)
nbctf.fit(tfx_train.toarray(), tfy_train)
rfctf.fit(tfx_train, tfy_train)

lrwv.fit(wordx_train, wordy_train)
svcwv.fit(wordx_train, wordy_train)
nbcwv.fit(wordx_train, wordy_train)
rfcwv.fit(wordx_train, wordy_train)

#Step 5: Model Evaluation
#Uses trained models to predict values using test data 
lrbow_pred = lrbow.predict(bowx_test)
svcbow_pred = svcbow.predict(bowx_test)
nbcbow_pred = nbcbow.predict(bowx_test.toarray())
rfcbow_pred = rfcbow.predict(bowx_test)

lrtf_pred = lrtf.predict(tfx_test)
svctf_pred = svctf.predict(tfx_test)
nbctf_pred = nbctf.predict(tfx_test.toarray())
rfctf_pred = rfctf.predict(tfx_test)

lrwv_pred = lrwv.predict(wordx_test)
svcwv_pred = svcwv.predict(wordx_test)
nbcwv_pred = nbcwv.predict(wordx_test)
rfcwv_pred = rfcwv.predict(wordx_test)

#Prints out results of data performance for each model 
print("Linear Regression w/ BoW Results:\n")
print(classification_report(bowy_test, lrbow_pred))
print("Support Vector Machine w/ BoW Results:\n")
print(classification_report(bowy_test, svcbow_pred))
print("Naive Bayes w/ BoW Results:\n")
print(classification_report(bowy_test, nbcbow_pred))
print("Random Forrest w/ BoW Results:\n")
print(classification_report(bowy_test, rfcbow_pred))

print("Linear Regression w/ tf*idf Results:\n")
print(classification_report(tfy_test, lrtf_pred))
print("Support Vector Machine w/ tf*idf Results:\n")
print(classification_report(tfy_test, svctf_pred))
print("Naive Bayes w/ tf*idf Results:\n")
print(classification_report(tfy_test, nbctf_pred))
print("Random Forrest w/ tf*idf Results:\n")
print(classification_report(tfy_test, rfctf_pred))

print("Linear Regression w/ Word2Vec Results:\n")
print(classification_report(wordy_test, lrwv_pred))
print("Support Vector Machine w/ Word2Vec Results:\n")
print(classification_report(wordy_test, svcwv_pred))
print("Naive Bayes w/ Word2Vec Results:\n")
print(classification_report(wordy_test, nbcwv_pred))
print("Random Forrest w/ Word2Vec Results:\n")
print(classification_report(wordy_test, rfcwv_pred))


