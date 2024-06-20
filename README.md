# Disaster Response Pipeline Project
In this project, the data engineering skills will be applied to analyze disaster data from Appen (formerly Figure 8) to build a model for an API that classifies disaster messages.

In the Project Workspace, a data set containing real messages sent during disaster events will be found. A machine learning pipeline will be created to categorize these events so that the messages can be sent to an appropriate disaster relief agency.

A project will be created that includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. This project will showcase your software skills, including your ability to create basic data pipelines and write clean, organized code.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Project Structure
- app
    - template
    - master.html  # main page of web app   
    - go.html  # classification result page of web app
    - run.py  # Flask file that runs app

- data
  - disaster_categories.csv  # data to process 
  - disaster_messages.csv  # data to process
  - process_data.py  # performs ETL process
  - DisasterResponse.db   # database to save clean data to

- models
  - train_classifier.py  # trains the classifier
  - classifier.pkl  # saved model 
