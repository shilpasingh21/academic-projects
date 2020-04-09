#!/usr/bin/env python3
"""
@author: Yuhan Zeng
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sys

# Read in the dataset file
datafile = sys.argv[1] 

df = pd.read_csv(datafile, index_col=0)

# Get an array of the labels for each restaurant
df = df.dropna(subset=['labels'])
labels = df['labels']
labels = labels.values.reshape(-1, 1)
y = []
for i in range(len(labels)):
    y.append(labels[i][0].split(","))

# Vectorize the review text
vectorizer = CountVectorizer(min_df=0, lowercase=True,stop_words='english')
X = vectorizer.fit_transform(df.review_combined).toarray()

# Binarize the multiple labels
mlb = MultiLabelBinarizer()
y_enc = mlb.fit_transform(y)

train_x, test_x, train_y, test_y = train_test_split(X, y_enc, test_size=0.3)

# Train and test a SVC model using OneVsRest 
SVC_clf = OneVsRestClassifier(SVC(probability=True, gamma='auto'))
SVC_clf.fit(train_x, train_y)
SVC_predictions = SVC_clf.predict(test_x)

# Train and test a multinomial Naive Bayes model using OneVsRest 
NB_clf =  OneVsRestClassifier(MultinomialNB())
NB_clf.fit(train_x, train_y)
NB_predictions = NB_clf.predict(test_x)

# Train and test a Random Forest model using OneVsRest 
RF_clf =  OneVsRestClassifier(RandomForestClassifier(n_estimators=75))
RF_clf.fit(train_x, train_y)
RF_predictions = RF_clf.predict(test_x)

## Calculate evaluation metrics
def evaluate(test_df, predict_df):
    combined_df = test_df + predict_df
    intersect_count, union_count, actual_count= 0.0, 0.0, 0.0
    accuracy, precision= 0.0, 0.0
    n = test_df.shape[0]
    for i in range(n):
        intersect_count = (combined_df.iloc[i,:] == 2).sum()
        union_count = (combined_df.iloc[i,:] > 0).sum()
        actual_count = test_df.iloc[i,:].sum()
        
        accuracy += intersect_count/union_count
        precision += intersect_count/actual_count
        
    accuracy /= n
    precision /= n
    return (accuracy, precision)

(SVC_accu, SVC_prec) = evaluate(pd.DataFrame(test_y), pd.DataFrame(SVC_predictions))
(NB_accu, NB_prec) = evaluate(pd.DataFrame(test_y), pd.DataFrame(NB_predictions))
(RF_accu, RF_prec) = evaluate(pd.DataFrame(test_y), pd.DataFrame(RF_predictions))

print("Test result for SVC:")
print("Accuracy = ", SVC_accu*100, "%", sep='')
print("Precision = ", SVC_prec*100, "%", sep='')

print("Test result for MultinomialNB:")
print("Accuracy = ", NB_accu*100, "%", sep='')
print("Precision = ", NB_prec*100, "%", sep='')

print("Test result for Random Forest:")
print("Accuracy = ", RF_accu*100, "%", sep='')
print("Precision = ", RF_prec*100, "%", sep='')
