import numpy as np

from sklearn.metrics import (
    mean_squared_error, confusion_matrix,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import cross_val_score, cross_val_predict

from logger import logger, LOGGING_INFO


class EvaluateRegression(object):
    def __init__(self, regressor, *args, **kwargs):
        self.regressor = regressor
        self.name = regressor.__name__

        self.model = regressor(*args, **kwargs)

    def metric(self, labels, predictions):
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

        metric = self.metric(labels, predictions)
        scores = self.score(train, labels)

        logger.log(LOGGING_INFO, self.name)
        logger.log(LOGGING_INFO, '----------')
        logger.log(LOGGING_INFO, ('metric:', _fmt(metric)))
        logger.log(LOGGING_INFO, '==========')
        logger.log(LOGGING_INFO, ('Scores:', ', '.join(map(_fmt, scores))))
        logger.log(LOGGING_INFO, ('Mean:', _fmt(scores.mean())))
        logger.log(LOGGING_INFO, ('StdDev', _fmt(scores.std())))

        logger.log(LOGGING_INFO, '--------\n\n')


class EvaluateClassifier(object):
    def __init__(self, classifier, *args, **kwargs):
        self.classifier = classifier
        self.name = classifier.__name__

        self.model = classifier(*args, **kwargs)

    def score(self, labels, predictions, manual=True):
        matrix = confusion_matrix(labels, predictions)

        index = [(0, 0), (1, 1), (1, 0), (0, 1)]
        tn, tp, fn, fp = [matrix[r][c] for r, c in index]

        p_predicted = tp + fp
        p_actual = tp + fn

        if manual:
            precision = tp / p_predicted
            recall = tp / p_actual
            f1 = 2 / (1 / precision + 1 / recall)

        else:
            precision = precision_score(labels, predictions)
            recall = recall_score(labels, predictions)
            f1 = f1_score(labels, predictions)

        return (matrix, precision, recall, f1)

    def run(self, train, labels, *args, **kwargs):
        def _fmt(d):
            return f'{d:.2f}'

        self.model.fit(train, labels)

        predictions = cross_val_predict(self.model, train, labels, *args, **kwargs)

        matrix, precision, recall, f1 = self.score(labels, predictions)

        logger.log(LOGGING_INFO, self.name)
        logger.log(LOGGING_INFO, '==========')
        logger.log(LOGGING_INFO, ('Confusion Matrix:', matrix))
        logger.log(LOGGING_INFO, ('P/R', list(map(_fmt, [precision, recall, f1]))))
        logger.log(LOGGING_INFO, '--------\n\n')
