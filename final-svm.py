import pandas as pd
import csv
from sklearn import cross_validation
from sklearn import svm
import nltk
import os

''' BBC data set is in sparse matrix format.
The make-data.py converts it into dense matrix and saves the dataframe
into data.pkl
'''

# Getting data
if 'data.pkl' in os.listdir():
	df = pd.read_pickle("data.pkl")
else:
	data = pd.read_table("data/bbc/bbc.mtx",delimiter=' ', skiprows=2,names=['word','file','val'])

	rows= set(data['file'])
	cols = set(data['word'])

	df = pd.DataFrame(index=rows,columns=cols)
	df = df.fillna(0)

	for x in data.itertuples():
		df[x[1]][x[2]] = x[3]

	df.to_pickle("data.pkl")

labels = pd.read_table("data/bbc/bbc.classes",delimiter=' ',skiprows=4,names=['file','labels'])

#there is a mismatch in the indexes of df and labels
labels.index = list(range(1,2226))
y=labels['labels']

#df = df.drop('labels',1)

#df['labels'] = labels['labels']
x_train, x_test, y_train, y_test = cross_validation.train_test_split(df,y,test_size=0.1,random_state=42)

clf = svm.SVC(kernel='rbf',C=1000)
clf.fit(x_train,y_train)

def load_features(filename) :
	file = open(filename, "r")
	terms = list(csv.reader(file))
	features = []
	for row in terms :
		features.append(row[0])
	# print(features[0:5])
	return features

def get_frequency(features, article) :
	Dict = dict()
	for elem in features:
		Dict[elem] = 0
	# print(Dict)

	for elem in article :
		if elem in features :
			Dict[elem] = Dict[elem] + 1
	# print(Dict)

	frequency = []
	for elem in features :
		frequency.append(Dict[elem])
	return frequency	

# features
features = load_features("data/bbc/bbc.terms")

# article

def articulate(filename) :
	f = open(filename,'r')
	data = f.read()
	reg_token = nltk.tokenize.RegexpTokenizer(r'\w+')
	data_tokens = reg_token.tokenize(data)
	stemmer = nltk.stem.PorterStemmer()
	article_data = [stemmer.stem(token) for token in data_tokens]
	frequency = get_frequency(features, article_data)
	return frequency

# CROSS VALIDATING
pred = clf.predict(x_test)
accuracy = float(sum(pred==y_test))/len(pred)

print('accuracy:',accuracy)

print("cross validation : 5 FOLD")

scores = cross_validation.cross_val_score(clf,df,y,cv=5)
print("----scores----")
print(scores)
print("average:",scores.mean())


# TESTING
file_names = os.listdir('news')
news=[]
for file in file_names:
	news.append(articulate('news/'+file))

pred = clf.predict(news)
classes = ['business','entertainment','politics','sport','tech']
pred = [classes[x] for x in pred]
output = zip(file_names,pred)
for i in output:
	print(i)