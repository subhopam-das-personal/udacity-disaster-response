import sys
import pandas as pd
import sqlalchemy 
from sqlalchemy import create_engine
import os


def load_data(messages_filepath, categories_filepath):
    """
    Load and merge the messages and categories data from the given filepaths.

    Args:
        messages_filepath (str): The filepath of the messages data file.
        categories_filepath (str): The filepath of the categories data file.

    Returns:
        pandas.DataFrame: The merged dataframe containing messages and categories data.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    """
    Clean the data by splitting the 'categories' column into individual category columns,
    converting the values to numeric, and dropping duplicates.

    Args:
        df (pandas.DataFrame): The input dataframe containing the 'categories' column.

    Returns:
        pandas.DataFrame: The cleaned dataframe with individual category columns.

    """
    print(df.head())
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.tolist()
    print(f"category_columns:{category_colnames}")
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # drop the original categories column from `df`

    df = df.drop('categories', axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()
    print(f'Number of Duplicates:{df.duplicated().sum()}')
    return df


def save_data(df, database_filename):
    """
    Save the given DataFrame to a SQLite database.

    Args:
        df (pandas.DataFrame): The DataFrame to be saved.
        database_filename (str): The filename of the SQLite database.

    Returns:
        None
    """
    print(f"database_filename:{database_filename}")
    engine = create_engine('sqlite:///' + os.path.join(os.getcwd(), database_filename))

    df.to_sql('t_diaster_data', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()