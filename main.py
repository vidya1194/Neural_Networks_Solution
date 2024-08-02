import pandas as pd
from sklearn.model_selection import train_test_split
from src.features.build_features import  engineer_features
from src.partition.data_partition import data_partitioning
from src.data.data_loading import load_data
from src.models.train_model import train_model, save_model
from src.models.evaluate_model import evaluate_model
from src.utils.utils import setup_logging

def main():
    logger = setup_logging()
    logger.info('Starting Neural Networks Solution')

    try:
        
        # Load data
        data = load_data('data/Admission.csv')
        logger.info("Data loaded")

        # Feature Engineering
        data = engineer_features(data)
        logger.info("Feature Engineering Completed")
        
        # Define features and target
        X = data.drop(['Admit_Chance'], axis=1)
        y = data['Admit_Chance']

        # Split the data
        Xtrain, Xtest, xtrain, xtest, ytrain, ytest = data_partitioning(X,y)
        logger.info('Data Partitioned')
        
        model, yprediction = train_model(X, y, Xtrain, Xtest, ytrain)
        
        mlp_accuracy, mlp_conf_matrix = evaluate_model(model, Xtest, ytest, yprediction)
        
        logger.info(f'MLPClassifier Accuracy: {mlp_accuracy}')
        logger.info(f'MLPClassifier Confusion Matrix: {mlp_conf_matrix}')
        
               
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()
