from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from src.utils.utils import setup_logging


def train_model(X, y, Xtrain, Xtest, ytrain):
    logger = setup_logging()
    try:
        # fit/train the model. Check batch size.
        MLP = MLPClassifier( batch_size=50, max_iter=100, random_state=123)
        MLP.fit(Xtrain,ytrain)
        
        # make Predictions
        prediction = MLP.predict(Xtest)
        
        #Grid Search CSV
        params = {'batch_size':[20, 30, 40, 50],
                    'hidden_layer_sizes':[[0,0],(2,),(3,),(3,2)],
                    'max_iter':[50, 70, 100]}
        grid = GridSearchCV(MLP, params, cv=10, scoring='accuracy')
        grid.fit(X, y)
        
        logger.info(f'grid.best_score_: {grid.best_score_}')
        logger.info(f'grid.best_params_: {grid.best_params_}')
             
        return MLP, prediction
        
    except Exception as e:
        logger.error("Error during model training: %s", str(e))
        raise
    
    

def save_model(model, filename):
    logger = setup_logging()
    try:
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")