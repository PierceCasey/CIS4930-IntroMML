import os 
import matplotlib.pyplot as plt
import random
import shutil
from sklearn.preprocessing import MaxAbsScaler

import numpy as np
import pandas as pd

import opensmile
import librosa

#Step 2: Exploratory data analysis
#Uses plots shown in example script in class using librosa 
#MUST be done before splitting data set, as not sure what file will be moved in split 

signal, sample_rate = librosa.load("./audio-features/angry/YAF_merge_angry.wav")

plt.figure(1)
librosa.display.waveshow(y=signal, sr=sample_rate)
plt.xlabel('Time / second')
plt.ylabel('Amplitude')
plt.show()

k = np.arange(len(signal))
T = len(signal)/sample_rate
frequency = k/T

data = np.fft.fft(signal)
abs_data = abs(data)
plt.figure(2)
plt.plot(frequency, abs_data)
plt.xlabel("Frequency / Hz")
plt.ylabel("Amplitude / dB")
plt.xlim([0, 1000])
plt.show()

#Step 1: Split testing and training
#I opted to move the files themselves into different folders, as openSMILE has a process_folder function 
# that I used to make my split

source = ".\\audio-features\\angry"
files = os.listdir(source)
destination = ".\\audio-features\\angry_test"

for filename in random.sample(files, 30):
    shutil.move(os.path.join(source, filename), destination)
    
source = ".\\audio-features\\fear"
files = os.listdir(source)
destination = ".\\audio-features\\fear_test"

for filename in random.sample(files, 30):
    shutil.move(os.path.join(source, filename), destination)
    
source = ".\\audio-features\\happy"
files = os.listdir(source)
destination = ".\\audio-features\\happy_test"

for filename in random.sample(files, 30):
    shutil.move(os.path.join(source, filename), destination)
    
source = ".\\audio-features\\sad"
files = os.listdir(source)
destination = ".\\audio-features\\sad_test"

for filename in random.sample(files, 30):
    shutil.move(os.path.join(source, filename), destination)

#Step 3: Acoustic feature extraction featuring openSMILE
#Extracts low-level features (25 features instead of 85), and then splits the data into train and test data sets

smile = opensmile.Smile(feature_set = opensmile.FeatureSet.eGeMAPSv02, 
                        feature_level = opensmile.FeatureLevel.LowLevelDescriptors)

#Process_folder function mentioned earlier 
angry_train = smile.process_folder(root=".\\audio-features\\angry", 
                          filetype='wav')

fear_train = smile.process_folder(root=".\\audio-features\\fear", 
                          filetype='wav')

happy_train = smile.process_folder(root=".\\audio-features\\happy", 
                          filetype='wav')

sad_train = smile.process_folder(root=".\\audio-features\\sad", 
                          filetype='wav')

angry_test = smile.process_folder(root=".\\audio-features\\angry_test", 
                          filetype='wav')

fear_test = smile.process_folder(root=".\\audio-features\\fear_test", 
                          filetype='wav')

happy_test = smile.process_folder(root=".\\audio-features\\happy_test", 
                          filetype='wav')

sad_test = smile.process_folder(root=".\\audio-features\\sad_test", 
                          filetype='wav')

#Moves files back, makes my life easier for testing
#Differing syntax from moving initially due to a bug I solved before, don't 
# particularly want to change back to initial "\\" syntax

source = "./audio-features/angry_test/"
destination = "./audio-features/angry/"
for filename in os.listdir(source):
    spath = source + filename
    dpath = destination + filename
    shutil.move(spath, dpath)
    
source = "./audio-features/fear_test/"
destination = "./audio-features/fear/"
for filename in os.listdir(source):
    spath = source + filename
    dpath = destination + filename
    shutil.move(spath, dpath)
    
source = "./audio-features/happy_test/"
destination = "./audio-features/happy/"
for filename in os.listdir(source):
    spath = source + filename
    dpath = destination + filename
    shutil.move(spath, dpath)
    
source = "./audio-features/sad_test/"
destination = "./audio-features/sad/"
for filename in os.listdir(source):
    spath = source + filename
    dpath = destination + filename
    shutil.move(spath, dpath)


print(angry_train)

#Step 4: Feature post-processing, 3 steps discussed in class
#Opted to complete steps on each set individually, as scaling must be fair in each time window regardless
# of speaker sentiment 

scaler = MaxAbsScaler()

#This function definitely took me the longest to make, scoured the internet for assistance so will explain

#Creates dataframe that will feature scaled and condensed windows 
angry_train_post = pd.DataFrame()

#Uses range of 'angry' training set, and divides that range into 10 rows of dataframe
for i in range(0, len(angry_train), 10):
    #Chunk of 10 rows is ready to be processed, uses temporary dataframe
    temp = angry_train.iloc[i:i+10]
    
    #Step 1: Feature Matrix Scaling 
    #Uses MaxAbs Scaler for values from -1 to 1 
    transform = scaler.fit_transform(temp)
    subset = pd.DataFrame(transform, columns=temp.columns)
    
    #Step 2: Feature Concatenation
    #I am fairly certain openSMILE is already taking care of the feature concatenation when I use 
    # the process_folder call 
    
    #Step 3: Feature Averaging 
    #Takes advantage of the built-in mean function and transpose (.T) to format frames
    average = pd.DataFrame(subset.mean(axis=0)).T
    angry_train_post = pd.concat([angry_train_post, average], ignore_index=True)

#Last row in dataframe can be very skewed, would rather delete all last rows for better results /
#less outlying data 
angry_train_post = angry_train_post[:-1]
#Adds y-value of speaker emotion onto the post-processed dataframe 
angry_train_post['Emotion'] = "Angry"
print(angry_train_post)

angry_test_post = pd.DataFrame()
for i in range(0, len(angry_test), 10):
    temp = angry_test.iloc[i:i+10]
    
    transform = scaler.fit_transform(temp)
    subset = pd.DataFrame(transform, columns=temp.columns)
    
    average = pd.DataFrame(subset.mean(axis=0)).T
    angry_test_post = pd.concat([angry_test_post, average], ignore_index=True)

angry_test_post = angry_test_post[:-1]
angry_test_post['Emotion'] = "Angry"
print(angry_test_post)

fear_train_post = pd.DataFrame()
for i in range(0, len(fear_train), 10):
    temp = fear_train.iloc[i:i+10]
    
    transform = scaler.fit_transform(temp)
    subset = pd.DataFrame(transform, columns=temp.columns)
    
    average = pd.DataFrame(subset.mean(axis=0)).T
    fear_train_post = pd.concat([fear_train_post, average], ignore_index=True)

fear_train_post = fear_train_post[:-1]
fear_train_post['Emotion'] = "Fear"
print(fear_train_post)

fear_test_post = pd.DataFrame()
for i in range(0, len(fear_test), 10):
    temp = fear_test.iloc[i:i+10]
    
    transform = scaler.fit_transform(temp)
    subset = pd.DataFrame(transform, columns=temp.columns)
    
    average = pd.DataFrame(subset.mean(axis=0)).T
    fear_test_post = pd.concat([fear_test_post, average], ignore_index=True)

fear_test_post = fear_test_post[:-1]
fear_test_post['Emotion'] = "Fear"
print(fear_test_post)

happy_train_post = pd.DataFrame()
for i in range(0, len(happy_train), 10):
    temp = happy_train.iloc[i:i+10]
    
    transform = scaler.fit_transform(temp)
    subset = pd.DataFrame(transform, columns=temp.columns)
    
    average = pd.DataFrame(subset.mean(axis=0)).T
    happy_train_post = pd.concat([happy_train_post, average], ignore_index=True)

happy_train_post = happy_train_post[:-1]
happy_train_post['Emotion'] = "Happy"
print(happy_train_post)

happy_test_post = pd.DataFrame()
for i in range(0, len(happy_test), 10):
    temp = happy_test.iloc[i:i+10]
    
    transform = scaler.fit_transform(temp)
    subset = pd.DataFrame(transform, columns=temp.columns)
    
    average = pd.DataFrame(subset.mean(axis=0)).T
    happy_test_post = pd.concat([happy_test_post, average], ignore_index=True)

happy_test_post = happy_test_post[:-1]
happy_test_post['Emotion'] = "Happy"
print(happy_test_post)

sad_train_post = pd.DataFrame()
for i in range(0, len(sad_train), 10):
    temp = sad_train.iloc[i:i+10]
    
    transform = scaler.fit_transform(temp)
    subset = pd.DataFrame(transform, columns=temp.columns)
    
    average = pd.DataFrame(subset.mean(axis=0)).T
    sad_train_post = pd.concat([sad_train_post, average], ignore_index=True)

sad_train_post = sad_train_post[:-1]
sad_train_post['Emotion'] = "Sad"
print(sad_train_post)

sad_test_post = pd.DataFrame()
for i in range(0, len(sad_test), 10):
    temp = sad_test.iloc[i:i+10]
    
    transform = scaler.fit_transform(temp)
    subset = pd.DataFrame(transform, columns=temp.columns)
    
    average = pd.DataFrame(subset.mean(axis=0)).T
    sad_test_post = pd.concat([sad_test_post, average], ignore_index=True)

sad_test_post = sad_test_post[:-1]
sad_test_post['Emotion'] = "Sad"
print(sad_test_post)


train_df = pd.concat([angry_train_post, fear_train_post, happy_train_post, sad_train_post], ignore_index=True)
test_df = pd.concat([angry_test_post, fear_test_post, happy_test_post, sad_test_post], ignore_index=True)
print(train_df)
print(test_df)

#Step 5: Build model 
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

svc = SVC()
nbc = GaussianNB()
kn = KNeighborsClassifier(n_neighbors=3)

svc_loud = SVC()
nbc_loud = GaussianNB()
kn_loud = KNeighborsClassifier(n_neighbors=3)

kn_mfcc = KNeighborsClassifier(n_neighbors=3)
kn_shim = KNeighborsClassifier(n_neighbors=3)

svc.fit(train_df.loc[:, train_df.columns != 'Emotion'], train_df['Emotion'])
nbc.fit(train_df.loc[:, train_df.columns != 'Emotion'], train_df['Emotion'])
kn.fit(train_df.loc[:, train_df.columns != 'Emotion'], train_df['Emotion'])

#Uses only the acoustic features associated with loudness, such as alpha ratio and the hammarberg index 
svc_loud.fit(train_df[['Loudness_sma3', 'alphaRatio_sma3', 'hammarbergIndex_sma3']], train_df['Emotion'])
nbc_loud.fit(train_df[['Loudness_sma3', 'alphaRatio_sma3', 'hammarbergIndex_sma3']], train_df['Emotion'])
kn_loud.fit(train_df[['Loudness_sma3', 'alphaRatio_sma3', 'hammarbergIndex_sma3']], train_df['Emotion'])

#Compares the k-nearest neighbors model performance with different accoustic features, such as MFCC and jitter/shimmer
kn_mfcc.fit(train_df[['mfcc1_sma3', 'mfcc2_sma3', 'mfcc3_sma3', 'mfcc4_sma3']], train_df['Emotion'])
kn_shim.fit(train_df[['jitterLocal_sma3nz', 'shimmerLocaldB_sma3nz']], train_df['Emotion'])

#Step 6: Evaluate models 
test_svc_predict = svc.predict(test_df.loc[:, train_df.columns != 'Emotion'])
test_nbc_predict = nbc.predict(test_df.loc[:, train_df.columns != 'Emotion'])
test_kn_predict = kn.predict(test_df.loc[:, train_df.columns != 'Emotion'])

test_svc_loud_pred = svc_loud.predict(test_df[['Loudness_sma3', 'alphaRatio_sma3', 'hammarbergIndex_sma3']])
test_nbc_loud_pred = nbc_loud.predict(test_df[['Loudness_sma3', 'alphaRatio_sma3', 'hammarbergIndex_sma3']])
test_kn_loud_pred = kn_loud.predict(test_df[['Loudness_sma3', 'alphaRatio_sma3', 'hammarbergIndex_sma3']])

test_kn_mfcc_pred = kn_mfcc.predict(test_df[['mfcc1_sma3', 'mfcc2_sma3', 'mfcc3_sma3', 'mfcc4_sma3']])
test_kn_shim_pred = kn_shim.predict(test_df[['jitterLocal_sma3nz', 'shimmerLocaldB_sma3nz']])


print("SVC Classification Results: \n")
print(classification_report(test_df['Emotion'], test_svc_predict))
print("Naive Bayes Classification Results:: \n")
print(classification_report(test_df['Emotion'], test_nbc_predict))
print("K-Nearest Neighbors Classification Results: \n")
print(classification_report(test_df['Emotion'], test_kn_predict))

print("SVC Loudness Acoustic Results: \n")
print(classification_report(test_df['Emotion'], test_svc_loud_pred))
print("Naive Bayes Loudness Acoustic Results: \n")
print(classification_report(test_df['Emotion'], test_nbc_loud_pred))
print("K-Nearest Neighbors Loudness Acoustic Results: \n")
print(classification_report(test_df['Emotion'], test_kn_loud_pred))

print("K-Nearest Neighbors MFCC Acoustic Results: \n")
print(classification_report(test_df['Emotion'], test_kn_mfcc_pred))
print("K-Nearest Neighbors Shimmer Acoustic Results: \n")
print(classification_report(test_df['Emotion'], test_kn_shim_pred))
