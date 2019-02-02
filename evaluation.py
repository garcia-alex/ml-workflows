import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

from logger import logger, LOGGING_INFO


class Evaluation(object):
    def __init__(self, regressor, *args, **kwargs):    
        self.regressor = regressor
        self.name = regressor.__name__
    
        self.model = regressor(*args, **kwargs)

    def mse(self, labels, predictions):
        mse = mean_squared_error(labels, predictions)
        rmse = np.sqrt(mse)
        
        return rmse

    def score(self, train, labels):
        scores = cross_val_score(
            self.model, train, labels, 
            scoring="neg_mean_squared_error", 
            cv=10
        )

        scores = np.sqrt(-scores)

        return scores

    def run(self, train, labels):
        def _fmt(d):
            return f'{d:.2f}'

        self.model.fit(train, labels)

        predictions = self.model.predict(train)

        mse = self.mse(labels, predictions)
        scores = self.score(train, labels)

        logger.log(LOGGING_INFO, self.name)
        logger.log(LOGGING_INFO, '----------')
        logger.log(LOGGING_INFO, ('mse:', _fmt(mse)))
        logger.log(LOGGING_INFO, '==========')
        logger.log(LOGGING_INFO, ('Scores:', ', '.join(map(_fmt, scores))))
        logger.log(LOGGING_INFO, ('Mean:', _fmt(scores.mean())))
        logger.log(LOGGING_INFO, ('StdDev', _fmt(scores.std())))

        logger.log(LOGGING_INFO, '--------\n\n')
