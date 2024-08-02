from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from src.utils.utils import setup_logging

def data_partitioning(X, y):
        
    logger = setup_logging()
    
    try:
        
        # Splitting the dataset into train and test data
        xtrain, xtest, ytrain, ytest =  train_test_split(X, y, test_size=0.2, random_state=123)
        
        # fit calculates the mean and standard deviation
        scaler = MinMaxScaler()
        scaler.fit(xtrain)
    
        # Now transform xtrain and xtest
        Xtrain = scaler.transform(xtrain)
        Xtest = scaler.transform(xtest)
        
        return  Xtrain, Xtest, xtrain, xtest, ytrain, ytest

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")