import pandas as pd
from sklearn import cross_validation
from sklearn import naive_bayes

''' BBC data set is in sparse matrix format.
The make-data.py converts it into dense matrix and saves the dataframe
into data.pkl
'''

data = pd.read_table("data/bbc/bbc.mtx",delimiter=' ',
	skiprows=2,names=['word','file','val'])
print(data.head())

rows= set(data['file'])
cols = set(data['word'])

df = pd.DataFrame(index=rows,columns=cols)
df = df.fillna(0)

for x in data.itertuples():
	df[x[1]][x[2]] = x[3]
 
df.to_pickle("data.pkl")