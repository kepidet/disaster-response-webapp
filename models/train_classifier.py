import sys
import pandas as pd
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

import re
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import pickle

def load_data(database_filepath):
    '''
    function: loading data from sql database and create variables X, y
    input: database filepath
    output: model variables X,Y 
    '''
    # connect to database
    engine = create_engine('sqlite:///'+ database_filepath)
    # table named disaster_responses will be returned as a dataframe
    df = pd.read_sql_table('DisasterResponse', con=engine)
    
    # extract message column
    X = df['message']
    
    # classification labels
    Y = df.iloc[:, 4:]
    
    # category names for visualization
    category_names = Y.columns
    
    return X, Y, category_names



def tokenize(text):
    '''
    function: returning the root form of the words of messages
    input: message text(str)
    output: cleaned list of words of messages
    '''
    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) # normalizing text
    words = word_tokenize(text) # tokenizing text
    words = [w for w in words if w not in stopwords.words("english")] # removing stop words
    lemmatizer = WordNetLemmatizer() # initiating text
    
    # lemmatizing - iterate through each token
    clean_words = []
    for w in words:
        clean = lemmatizer.lemmatize(w)
        clean_words.append(clean)
    
    return clean_words


def build_model():
    '''
    function: building a model for classifing messages
    output: pipeline
    '''
    ### defining pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',  MultiOutputClassifier(RandomForestClassifier()))
    ])

    # changing parameters with grid search
    parameters = {
            'clf__estimator__n_estimators': [60]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test):
    '''
    function: testing function, print out the classification report
    input: predicted categories
    '''
    Y_pred = model.predict(X_test)
    
    i = 0
    for col in Y_test:
        print('Category {}: {}'.format(i+1, col))
        print(classification_report(Y_test[col], Y_pred[:, i]))
        i = i+1
    accuracy = (Y_pred == Y_test).mean()
    print('Accuracy: ', accuracy)
    sample_accuracy = accuracy.mean()
    print('Average accuracy: ', sample_accuracy)


def save_model(model, model_filepath):
    '''
    function: save the pickle file
    input: classification model
    output: path of pickle file
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()