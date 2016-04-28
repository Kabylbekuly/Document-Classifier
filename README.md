## Document-Classifier
Trains a Classifier on BBC data sets and give the accuracy on test data. For testing it on other news articles, add the articles as text files in the folder news

## Running the code
1. `python3 make-data.py` : Run this to generate the data. It will be used by final code for training.
2. `python3 final-nb.py` : Trains Naive-Bayes classifier and gives the accuracy and predictions on articles in news folder
3. `python3 final-svm.py` : Trains SVM classifier and gives the accuracy and predictions on articles in news folder