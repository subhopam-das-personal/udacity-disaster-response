# Disaster Response Pipeline
# Author: Sidharth Kumar Mohanty

#import necessary libraries
import sys
import nltk
import re
nltk.download(['punkt', 'wordnet'])
import warnings
import pickle
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier


def load_data(database_filepath):
    """
    Load data from a SQLite database and return the features, labels, and category names.
    
    Parameters:
        database_filepath (str): The file path of the SQLite database.
        
    Returns:
        X (pandas.Series): The features (messages) as a pandas Series.
        y (pandas.DataFrame): The labels (categories) as a pandas DataFrame.
        category_names (list): The names of the categories.
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df.message
    y = df[df.columns[4:]]
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    """
    Tokenizes the input text by removing URLs, converting text messages into tokens,
    and lemmatizing the tokens.

    Args:
        text (str): The input text to be tokenized.

    Returns:
        list: A list of clean tokens.

    """
    # remove URL present in the messages
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # convert text messages into tokens
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens    


def build_model():
    """
    Build and return a machine learning model for classifying disaster response messages.
    
    Returns:
    model (GridSearchCV): A machine learning model that has been optimized using grid search.
    """
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [8, 15],
        'clf__estimator__min_samples_split': [2],
    
    }
    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=2, cv=3)
    return model


def evaluate_model(model, X_test, y_test, category_names):
    """
    Evaluate the performance of a machine learning model on the test data.

    Parameters:
        model (object): The trained machine learning model.
        X_test (array-like): The test data features.
        y_test (array-like): The true labels for the test data.
        category_names (list): The list of category names.

    Returns:
        None
    """

    y_pred = model.predict(X_test)
    y_pred[123].shape

    """ for i in range(36):
        print("=======================",y_test.columns[i],"======================")
        print(classification_report(y_test.iloc[:,i], y_pred[:,i])) """
    for i in range(y_pred.shape[1]):
        print("=======================",y_test.columns[i],"======================")
        print(classification_report(y_test.iloc[:,i], y_pred[:,i]))

def save_model(model, model_filepath):
    '''
    Save the trained model to a file.

    Parameters:
    model (object): The trained model object to be saved.
    model_filepath (str): The file path where the model should be saved.

    Returns:
    None
    '''
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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