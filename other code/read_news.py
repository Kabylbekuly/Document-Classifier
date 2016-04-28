import nltk
f = open('news1.txt','r')
data = f.read()
reg_token = nltk.tokenize.RegexpTokenizer(r'\w+')
data_tokens = reg_token.tokenize(data)
stemmer = nltk.stem.PorterStemmer()
article_data = [stemmer.stem(token) for token in data_tokens]