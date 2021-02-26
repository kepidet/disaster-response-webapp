import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    funtion: load and merge messages and categories datasets
    input: filepaths of the two datasets
    output: merged dataframe
    '''
    # load messages
    messages = pd.read_csv(messages_filepath)
    # load categories
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on="id")
    return df


def clean_data(df):
    '''
    function: it splits the categories column into separate, names columns, converts values to binary, and drops duplicates
    input: dataframe created in load_data function
    output: cleaned data frame
    '''
    # Spliting the values in the categories column on the ; character so that each value becomes a separate column
    categories = df['categories'].str.split(pat=';', expand=True)
    # selecting the first row of the categories dataframe
    row = categories.iloc[[1]]
    # Using the first row of categories dataframe to create column names for the categories data
    category_colnames = [category_name.split('-')[0] for category_name in row.values[0]]
    # Renaming columns of categories with new column names
    categories.columns = category_colnames

    # Converting category values to just numbers 0 or 1
    for column in categories:
        # setting each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1:]
        # converting column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # Replacing categories column in df with new category columns
    # droping the original categories column from `df`
    df = df.drop(['categories'], axis=1)
    # concatenating the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], join='inner', axis=1)

    # Removing duplicates
    df.drop_duplicates(inplace=True)
    
    #Removing child alone as it has zero values
    df = df.drop(['child_alone'], axis=1)
    
    # Given value 2 in the related field are neglible so it could be error. Replacing 2 with 1 to consider it a valid response.
    df['related']=df['related'].map(lambda x: 1 if x == 2 else x)
    
    # encoding genre
    encoding_genre = {'genre': {'news': 1, 'direct':2, 'social':3}}
    df = df.replace(encoding_genre)

    return df


def save_data(df, database_filename):
    '''
    function: stores the clean data into a SQLite database in the specified database file path
    input: cleaned data frame
    output: SQLite database 
    ''' 

    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')


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