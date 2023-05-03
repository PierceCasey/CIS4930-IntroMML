import numpy as np
import pandas as pd 
import matplotlib as mpl
import re
import os
from nltk.corpus import stopwords
import cv2 as cv
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.sparse

#For bag-of-words linguistic feature extraction from title 
from sklearn.feature_extraction.text import CountVectorizer

#For various model development
from sklearn.ensemble import RandomForestRegressor
#from sklearn.neural_network import MLPRegressor
#from sklearn.linear_model import LinearRegression
#from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error

#For potential pipeline construction for model
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

#Data preprocessing 
df = pd.read_csv('yt_data.csv')
df = df[['Id', 'Channel', 'Title', 'Length', 'Subscribers', 'Views']]

lengths = []

#Converts video length from string of varied numbers into minutes, rounded up to whole minute 
for index, row in df.iterrows():
    process = row['Length']
    
    process = re.sub(r'^(\d{1,2}):{1}.*', r'\g<1>', process)
    
    process = int(process)
    process += 1
    lengths.append(process)

#Length of video must be renamed to an integer to avoid bug at later step where columns can not be of string type
df[-1] = lengths

subs = []
views = []

#Converts strings of various types in the millions and thousands to integers to be processed with regex
for index, row in df.iterrows():
    process = row['Subscribers']
    
    #Specific for 1K, 12K, 123K examples
    process = re.sub(r'(\d+)K{1}.*', r'\g<1>000', process)
    #Specific for 1.23M example
    process = re.sub(r'(\d)\.{1}(\d{2})M{1}.*', r'\g<1>\g<2>0000', process)
    #Specific for 1.2M, 12.3M examples
    process = re.sub(r'(\d+)\.{1}(\d{1})M{1}.*', r'\g<1>\g<2>00000', process)
    #Specific for 1M, 12M, 123M examples
    process = re.sub(r'(\d+)M{1}.*', r'\g<1>000000', process)
    
    process = int(process)
    subs.append(process)
    
    process = row['Views']
    #Specific for 1K, 12K, 123K examples
    process = re.sub(r'(\d+)K{1}.*', r'\g<1>000', process)
    #Specific for 1.23M example
    process = re.sub(r'(\d)\.{1}(\d{2})M{1}.*', r'\g<1>\g<2>0000', process)
    #Specific for 1.2M, 12.3M examples
    process = re.sub(r'(\d+)\.{1}(\d{1})M{1}.*', r'\g<1>\g<2>00000', process)
    #Specific for 1M, 12M, 123M examples
    process = re.sub(r'(\d+)M{1}.*', r'\g<1>000000', process)

    process = int(process)
    views.append(process)

#Divides both lists to obtain a video's ratio of views to subscribers, appends this list to dataframe
ratio = [a/b for a, b in zip(views, subs)]
df['Ratio'] = ratio

#Length to Ratio correlation testing
corr = df[['Ratio', -1]]
views = pd.DataFrame(views, columns=['Views'])
subs = pd.DataFrame(subs, columns=['Subs'])
corr = pd.concat([corr, views, subs], axis=1)
print(corr)
print(corr.corr(method='pearson')[-1].sort_values())
#Shows a SMALL negative correlation in whether or not a video's views are based on length, around 10%
#Views and subs do not show any correlation

#Plots ratios on a histogram plot; most values are around .5 to 1.5, meaning the view count and subscriber count 
# are very dependent on eachother
sns.histplot(data=df['Ratio'], palette='bright')
plt.show()

#Image processing 
#Training data 
images = []

for index, row in df.iterrows():
    channel = row['Channel']
    id = row['Id']
    #Finds desired image in folder through table ID
    file = "./images/" + channel + "/" + id + ".jpg"
    if os.path.exists(file):
        #Just reads file and scales it down. All image processing steps I attempted made the model train worse
        img = cv.imread(file)
        img = cv.resize(img, (50, 50))
        
        #Code for showing image being processed
        #cv.imshow('None', img)
        #cv.waitKey(0)
        
        #Flattens image into rgb array to be used by model 
        flatten = img.flatten()
        images.append(flatten)
    #Some images are not present in set, which returned an error. Must delete value if image does not exist. 
    else:
        df = df.drop(index)
        
#Resets indices of dataset after values are dropped
df = df.reset_index()
df = df.drop('index', axis = 1)

image_df = pd.DataFrame(images)

#Text preprocessing into bag-of-words
#Imports stopwords to be taken out of titles 
stop_words = set(stopwords.words('english'))
corpus = []

for index, row in df.iterrows():
    process = row['Title']
    
    #Makes titles lowercase 
    process = process.lower()
    
    #Removes contractions from titles 
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
    
    #Removes punctuation, extra spaces, and turns numbers into # 
    process = re.sub('["]', '', process)
    process = re.sub('[^a-z0-9" +"]', '', process)
    process = re.sub(r" +", " ", process)
    process = re.sub(r'\d+', '#', process)
    
    #Removes stopwords
    process = [w for w in process.split() if not w in stop_words]
    process = (" ").join(process)
    
    corpus.append(process)

#Uses a simple count vectorizer to count all words present in corpus
vectorizer = CountVectorizer()
bag_of_words = vectorizer.fit_transform(corpus)

#Creates sparse dataframe to hold transformed results, since the resulting matrix is sparse
bag_df = pd.DataFrame.sparse.from_spmatrix(bag_of_words)

#After processing all data, splits into training and testing data 
df = df[['Ratio', -1]]

frames = [df, bag_df, image_df]
df = pd.concat(frames, axis=1)
df = df.fillna(0)

train_df = df.sample(frac=0.7)
test_df = df.drop(train_df.index)

#Sets y values to variables and drops the y value from each dataset
train_y = train_df['Ratio']
test_y = test_df['Ratio']

train_df = train_df.drop(columns=['Ratio'])
test_df = test_df.drop(columns=['Ratio'])

#Model Development
#Random Forest Regression yielded the best results with estimators = 100
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(train_df, train_y)
y_pred = model.predict(test_df)

#Simple scores to explain model accuracy and error 
print("R square score: ", r2_score(test_y, y_pred))
print("Mean squared error score: ", mean_squared_error(test_y, y_pred))
