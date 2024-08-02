from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from src.utils.utils import setup_logging

logger = setup_logging()

def evaluate_model(model, X_test, y_test, yprediction):
    try:
        conf_matrix = confusion_matrix(y_test, yprediction)
        accuracy = accuracy_score(y_test, yprediction)
        
        return accuracy, conf_matrix

    except Exception as e:
        logger.error("Error during model evaluation: %s", str(e))
        raise
