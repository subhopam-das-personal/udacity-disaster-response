import sys
# import necessary libraries 
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])

import re
import numpy as np
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, classification_report
import os
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
import pickle

import warnings
warnings.filterwarnings('ignore')

def load_data(database_filepath):
    """
    Load data from a SQLite database.

    Parameters:
    database_filepath (str): The file path of the SQLite database.

    Returns:
    X (pandas.Series): The input messages.
    Y (pandas.DataFrame): The target categories.
    category_names (list): The list of category names.
    """
    print(f"database_filename:{database_filepath}")
    engine = create_engine('sqlite:///' + os.path.join(os.getcwd(), database_filepath))
    df = pd.read_sql_table('t_diaster_data', engine)
    X = df.message
    Y = df.iloc[:,4:]
    row = df.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_names = row.tolist()
    return X, Y, category_names


def tokenize(text):
    """
    Tokenizes the input text by performing the following steps:
    1. Detects and replaces URLs with "urlplaceholder".
    2. Tokenizes the text into individual words.
    3. Lemmatizes each word to its base form.
    4. Converts each word to lowercase and removes leading/trailing whitespaces.

    Args:
        text (str): The input text to be tokenized.

    Returns:
        list: A list of clean tokens.

    Example:
        >>> tokenize("This is an example sentence.")
        ['this', 'is', 'an', 'example', 'sentence']
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # detect all URL present in the messages
    detected_urls = re.findall(url_regex, text)
    # replace URL with "urlplaceholder"
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Build and return a machine learning model pipeline for classifying disaster response messages.

    Returns:
    cv (GridSearchCV): A GridSearchCV object that performs hyperparameter tuning on the pipeline.
    """

    pipeline = Pipeline([
        ('cvect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=-1)))
    ])

    parameters = {
        'clf__estimator__n_estimators': [100, 150],
        'clf__estimator__min_samples_split': [2, 4],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the performance of a machine learning model on the test data.

    Parameters:
    model (object): The trained machine learning model.
    X_test (array-like): The input features for the test data.
    Y_test (array-like): The true labels for the test data.
    category_names (list): The list of category names.

    Returns:
    None
    """
    Y_pred = model.predict(X_test)
    report = classification_report(Y_test, Y_pred, target_names=category_names)

    # Print the classification report
    print("Classification Report:\n", report)


def save_model(model, model_filepath):
    """
    Save the trained model to a file.

    Parameters:
    model (object): The trained model object to be saved.
    model_filepath (str): The file path where the model should be saved.

    Returns:
    None
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


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
        evaluate_model(model, X_test, Y_test, category_names)

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