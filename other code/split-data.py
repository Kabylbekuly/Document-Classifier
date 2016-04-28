import pandas as pd 

df = pd.read_pickle("data.pkl")
labels = pd.read_table("data/bbc/bbc.classes",delimiter=' ',
	skiprows=4,names=['file','labels'])

labels.index = list(range(1,2226))
y=labels['labels']

#df['labels'] = labels['labels']
x_train,x_test,y_train,y_test = cross_validation.train_test_split(df,y,test_size=0.3)

x_train.to_pickle('x_train.pkl')
