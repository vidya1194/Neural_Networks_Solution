import pandas as pd
from src.utils.utils import setup_logging

def engineer_features(data):
    """
    Engineer features from the dataset.
    """
    logger = setup_logging()
    try:
        # Converting the target variable into a categorical variable
        data['Admit_Chance']=(data['Admit_Chance'] >=0.8).astype(int)
        
        # Dropping columns
        data = data.drop(['Serial_No'], axis=1)
        
        # Create dummy variables for all 'object' type variables except 'Loan_Status'
        data = pd.get_dummies(data, columns=['University_Rating','Research'])
        
        return data
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
