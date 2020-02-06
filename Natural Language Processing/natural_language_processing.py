import pandas as pd
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv('Restaurant_Reviews.tsv',  delimiter = '\t', quoting = 3)
corpus = []
for i in range(len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()  
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

tp = cm[0,0]
tn = cm[1,1]
fp = cm[1,0]
fn = cm[0,1]

# Accuracy = (TP + TN) / (TP + TN + FP + FN)
accuracy = (tp + tn)/(tp+tn+fp+fn)

# Precision = TP / (TP + FP)
precision = tp/(tp+fp)

# Recall = TP / (TP + FN)
recall = tp/(tp + fn)

# F1 Score = 2 * Precision * Recall / (Precision + Recall)
f1_score = (2 * precision * recall)/(precision + recall)