import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pygal
import lxml
import pycountry_convert
import pygal_maps_world
import seaborn as sns

#"""
#Reads .csv files and sorts them by Article No., appending author info after article info, but never on the same line 
df = pd.concat(map(pd.read_csv, ['articleinfo.csv', 'authorinfo.csv']))
df = df.sort_values('Article No.')
df = df.reset_index()
#Fills all empty cells with 0 value 
df.fillna(0, inplace=True)
#Removes additional index from data 
df = df[['Article No.', 'Title', 'Year', 'Author Number', 'Key Words', 'Citation', 'Source', 'Abstract', 'Type', 'Author Name', 'Author Affiliation', 'Country', 'h-index']]

#1. 
#Get # of articles posted in each year in descending order 
plot = df['Year'].value_counts()
#Remove any articles without year given 
plot = plot.drop(plot.index[0])
plot = plot.sort_index(ascending=False)

#Formats and plots values with matplotlib functions
x = plot.index.to_numpy()
y = plot.array.to_numpy()
plt.plot(x, y)
plt.title("yearly_publication")
plt.xlabel("year")
plt.ylabel("# of articles")
plt.legend(["y"])
plt.show()

#2.
#Same idea as Question 1, except the # of total citations need to be calculated 
plot = df[['Year', 'Citation']]
plot = plot.groupby('Year')['Citation'].sum()
plot = plot.drop(plot.index[0])
plot = plot.sort_index(ascending=False)

x = plot.index.to_numpy()
y = plot.array.to_numpy()
plt.plot(x, y)
plt.title("yearly_citation")
plt.xlabel("year")
plt.ylabel("# of articles")
plt.legend(["y"])
plt.show()

#3. 
#Correcting country naming errors existing in original data set 
df['Country'] = df['Country'].replace(['Spain '], 'Spain')
df['Country'] = df['Country'].replace(['Denamrk'], 'Denmark')
df['Country'] = df['Country'].replace(['Denmark '], 'Denmark')
df['Country'] = df['Country'].replace(['Chile '], 'Chile')
df['Country'] = df['Country'].replace(['Korea'], 'Korea, Republic of')
df['Country'] = df['Country'].replace(['Bristol'], 'United Kingdom')
df['Country'] = df['Country'].replace(['Chian'], 'China')
df['Country'] = df['Country'].replace(['Israel '], 'Israel')
plot = df['Country'].value_counts()

plot = plot.drop(plot.index[0])
countrytags = plot.index.to_list()
data = plot.to_list()

#Pygal_maps_world requires country data to be in country ID; must be formatted 
countryids = []
i = 0
while i < len(countrytags):
    countryids.insert(i, pycountry_convert.country_name_to_country_alpha2(countrytags[i]).lower())
    i = i + 1
    
newplot = pd.Series(data, countryids)
newData = newplot.to_dict()

#Creates interactive world map and displays it in browser 
worldmap = pygal.maps.world.World()
worldmap.title = 'Publications by Country'
worldmap.add('Publications', newData)
worldmap.render_in_browser()

#4. 
new_df = df.value_counts('Author Affiliation')
new_df = new_df.drop(new_df.index[0])
new_df = new_df.reset_index()
print(new_df.head(5))

#5. 
new_df = df.sort_values('h-index', ascending=False)
new_df = new_df[['Author Name', 'h-index']]
new_df = new_df.reset_index(drop=True)
print(new_df.head(5))
#"""

#Part 2 
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score,mean_squared_error
import statsmodels.api as sm

#"""
df = pd.read_csv('data.csv')
df = df[['Purchase', 'SUS', 'Duration', 'Gender', 'ASR_Error', 'Intent_Error']]

#Creating independent variables via encoding 
le = LabelEncoder()
le.fit(df['ASR_Error'])
df['ASR_Error'] = le.transform(df['ASR_Error'])

le.fit(df['Intent_Error'])
df['Intent_Error'] = le.transform(df['Intent_Error'])

le.fit(df['Duration'])
df['Duration'] = le.transform(df['Duration'])

le.fit(df['Gender'])
df['Gender'] = le.transform(df['Gender'])

le.fit(df['Purchase'])
df['Purchase'] = le.transform(df['Purchase'])

print(df.corr(method='pearson')['SUS'].sort_values())

y = df['SUS']
x = df[['ASR_Error', 'Intent_Error', 'Duration', 'Gender', 'Purchase']]
x = sm.add_constant(x)

model = sm.OLS(y, x).fit()
print(model.summary())

x_train, x_test, y_train, y_test = train_test_split(x, y)
lr = LinearRegression().fit(x_train, y_train)

y_train_pred = lr.predict(x_train)
y_test_pred = lr.predict(x_test)

print("R square score of linear regression: ", lr.score(x_test, y_test))
#"""

#Part 3
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

#""" 
#Initial check to see if any data values are missing in set 
print(df.isna().sum())

#Create labels
y = df['Purchase'].to_numpy()

x = df[['ASR_Error', 'Intent_Error', 'Duration', 'Gender']].to_numpy()

#Scales data (not necessary, but encouraged for extreme data ranges)
scale = StandardScaler()
scaled_x = scale.fit_transform(x)

#Splits data into test and training sections for eventual use by the models 
x_train, x_test, y_train, y_test = train_test_split(scaled_x, y, test_size = 0.5)

#Creates variables of each classification type we will be using 
lr = LogisticRegression()
neigh = KNeighborsClassifier(n_neighbors = 4)
nbc = GaussianNB()
rfc = RandomForestClassifier()

#Fits training data into classification models and trains them 
lr.fit(x_train, y_train)
neigh.fit(x_train, y_train)
nbc.fit(x_train, y_train)
rfc.fit(x_train, y_train)

#Once models are trained, test data is given to see how the models perform with unseen data 
y_lr_predicted = lr.predict(x_test)
y_lr_pred_probab = lr.predict_proba(x_test)

y_neigh_predicted = neigh.predict(x_test)
y_neigh_pred_probab = neigh.predict_proba(x_test)

y_nbc_predicted = nbc.predict(x_test)
y_nbc_pred_probab = nbc.predict_proba(x_test)

y_rfc_predicted = rfc.predict(x_test)
y_rfc_pred_probab = rfc.predict_proba(x_test)

#Results of unseen data tests are shown, comparing actual data to model performance 
print(classification_report(y_test, y_lr_predicted))
print(classification_report(y_test, y_neigh_predicted))
print(classification_report(y_test, y_nbc_predicted))
print(classification_report(y_test, y_rfc_predicted))
#"""