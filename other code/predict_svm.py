import pandas as pd
import csv
from sklearn import cross_validation
from sklearn import svm
from sklearn.externals import joblib
import nltk
import os


''' BBC data set is in sparse matrix format.
The make-data.py converts it into dense matrix and saves the dataframe
into data.pkl
'''
clf = joblib.load('svm_model.pkl')

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

# TESTING
file_names = os.listdir('news')
news=[]
for file in file_names:
	news.append(articulate('news/'+file))

pred = clf.predict(news)
output = zip(file_names,pred)
for i in output:
	print(i)