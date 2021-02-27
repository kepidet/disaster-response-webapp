# Disaster Response Pipeline Project

### Table of Contents

1. [Installation](#installation)
2. [Project Overview](#overview)
3. [Project Components](#components)
4. [Dataset](#dataset)
5. [Instructions](#instructions)
6. [Licensing](#licensing)

## Installation <a name="installation"></a>

There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python. The code should run with no issues using Python versions 3.*.


## Project Overview <a name="overview"></a>

In this project, I'm analyzing disaster data from [Appen](https://appen.com/) to build a model for an API that classifies disaster messages. Data sets containe real messages that were sent during disaster events. The task is to create a machine learning pipeline to categorize these events so that we can send the messages to an appropriate disaster relief agency. The project includes a web app where an emergency worker can input a new message and get classification results in several categories.

![image](https://github.com/kepidet/disaster-response-webapp/blob/main/prtsc/Disaster%20Response%20Project%20web%20prtsc.PNG)


## Project Components <a name="components"></a>

1. ETL Pipeline - data cleaning: process.py (data folder)
    - loading messages and categories datasets
    - merging datasets
    - cleaning data
    - storing in a SQLite database

![process_data](https://github.com/kepidet/disaster-response-webapp/blob/main/prtsc/Disaster%20Response%20Project%20process_data%20prtsc.PNG)

2. ML Pipeline - machine learning: train_classifier.py (models folder)
    - loading data from the SQLite database
    - slipts dataset into training and test sets
    - building text processing and machine learning pipeline
    - training and tuning model using GridSearchCV
    - outputing results on the test set
    - exporting the final model as a pickle file

![classification](https://github.com/kepidet/disaster-response-webapp/blob/main/prtsc/Disaster%20Response%20Project%20classification%20prtsc.PNG)

![results](https://github.com/kepidet/disaster-response-webapp/blob/main/prtsc/Disaster%20Response%20Project%20evaulation%20total%20prtsc.PNG)

3. Flask Web App - run.py (app folder)
    - data visualization using Plotly 


![chart](https://github.com/kepidet/disaster-response-webapp/blob/main/prtsc/Disaster%20Response%20Project%20charts%20prtsc.PNG)


## Dataset <a name="dataset"></a>

- disaster_messages.csv
- disaster_categories.csv


## Instructions <a name="instructions"></a>
To execute the app follow the instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## Licensing <a name="licensing"></a>

Must give credit to Appen for the data. This app was completed as part of the Udacity Data Scientist Nanodegree.
