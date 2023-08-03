from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import numpy as np
from sklearn.model_selection import train_test_split



app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    data = pd.read_csv('kindle_reviews.csv')
    data.drop(['Unnamed: 0', 'asin', 'helpful','reviewTime','reviewerID','reviewerName','summary','unixReviewTime'], axis=1, inplace=True)
    g=data.overall>3
    data.loc[g,'reviewText']=data.loc[g,'reviewText'].fillna('Good')
    b=data.overall<=3
    data.loc[b,'reviewText']=data.loc[b,'reviewText'].fillna('Bad')
    data.overall=data.overall.replace([1,2,3],0)
    data.overall=data.overall.replace([4,5],1)
    data.loc[data['overall'] == 0, 'type'] = 'Negative Review'
    data.loc[data['overall'] == 1, 'type'] = 'Positive Review'
    X=data['reviewText']
    y=data['type']
    cv = CountVectorizer()
    X = cv.fit_transform(X) # Fit the Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    #Naive Bayes Classifier
    clf = MultinomialNB()
    clf.fit(X_train,y_train)
    clf.score(X_test,y_test)
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)
