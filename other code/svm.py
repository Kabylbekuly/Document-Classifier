import pandas as pd
from sklearn import cross_validation
from sklearn import svm
from sklearn.externals import joblib


''' BBC data set is in sparse matrix format.
The make-data.py converts it into dense matrix and saves the dataframe
into data.pkl
'''

df = pd.read_pickle("data.pkl")
labels = pd.read_table("data/bbc/bbc.classes",delimiter=' ',
	skiprows=4,names=['file','labels'])

#there is a mismatch in the indexes of df and labels
labels.index = list(range(1,2226))
y=labels['labels']

#df = df.drop('labels',1)

#df['labels'] = labels['labels']
# x_train,x_test,y_train,y_test = cross_validation.train_test_split(df,y,test_size=0.3,random_state=42)



clf = svm.SVC(kernel='rbf',C=1000)
clf.fit(df,y)
# pred = clf.predict(x_test)
# accuracy = float(sum(pred==y_test))/len(pred)

# print('accuracy:',accuracy)
# # accuracy: 94


# print("cross validation : 5 FOLD")

# scores = cross_validation.cross_val_score(clf,df,y,cv=5)
# print("----scores----")
# print(scores)
# print("average:",scores.mean())
# #accuracy 92
joblib.dump(clf, 'svm_model.pkl', compress=9)

