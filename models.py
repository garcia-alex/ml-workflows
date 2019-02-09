import numpy as np

from sklearn.metrics import (
    mean_squared_error, confusion_matrix,
    precision_recall_curve, roc_curve, roc_auc_score
)
from sklearn.model_selection import cross_val_score, cross_val_predict


class GenericModel(object):
    def __init__(self, algorithm, *args, **kwargs):
        self.algorithm = algorithm
        self.name = algorithm.__name__

        self.model = algorithm(*args, **kwargs)

    def train(self, features, labels):
        self.features = features
        self.labels = labels

        self.model.fit(features, labels)

    def __repr__(self):
        representation = '<{0}.{1} object at {2}>: {3}'.format(
            self.__module__,
            type(self).__name__,
            hex(id(self)),
            self.name
        )

        return representation

    def __str__(self):
        return f'Generic model with algorithm: {self.name}'


class RegressionModel(GenericModel):
    def __init__(self, model, *args, **kwargs):
        super(RegressionModel, self).__init__(model, *args, **kwargs)

    def evaluate(self, features, labels):
        self.train(features, labels)

        self.predictions = self.model.predict(features)

        self.mse = mean_squared_error(labels, self.predictions)
        self.rmse = np.sqrt(self.mse)

        scores = cross_val_score(
            self.model, features, labels,
            scoring="neg_mean_squared_error",
            cv=10
        )

        self.scores = np.sqrt(-scores)

        self.evaluated = True

    def __repr__(self):
        representation = super(RegressionModel, self).__repr__()

        if not self.evaluated:
            return f'{representation}: not applied'

        representation = '{0},\nsize: {1:.0f}, rmse: {2:.2f}, mean: {3:.2f}, std: {4:.2f}'.format(
            representation,
            len(self.labels),
            self.rmse,
            self.scores.mean(),
            self.scores.std()
        )

        return representation


class BinaryClassifierModel(GenericModel):
    def __init__(self, model, *args, **kwargs):
        super(BinaryClassifierModel, self).__init__(model, *args, **kwargs)

    def evaluate(self, features, labels, *args, **kwargs):
        self.train(features, labels)

        self.predictions = cross_val_predict(self.model, features, labels, *args, **kwargs)

        self.matrix = confusion_matrix(labels, self.predictions)

        self.tn = self.matrix[0][0]
        self.fp = self.matrix[0][1]
        self.fn = self.matrix[1][0]
        self.tp = self.matrix[1][1]

        self.pp = self.tp + self.fp
        self.pn = self.tn + self.fn
        self.ap = self.tp + self.fn
        self.an = self.tn + self.fp

        self.sensitivity = self.tp / self.ap
        self.specificity = self.tn / self.an

        self.recall = self.tp / self.ap  # identical to sensitivity
        self.precision = self.tp / self.pp
        self.f1 = 2 / (1 / self.precision + 1 / self.recall)

        self.tpr = self.sensitivity
        self.fnr = 1 - self.tpr
        self.tnr = self.specificity
        self.fpr = 1 - self.specificity

        self.auc = roc_auc_score(labels, self.predictions)

        self.evaluated = True

    def pc_curve(self, *args, **kwargs):
        predictions = cross_val_predict(self.model, self.features, self.labels, *args, **kwargs)

        precision, recall, thresholds = precision_recall_curve(self.labels, predictions)

        return (precision, recall, thresholds, predictions)

    def roc_curve(self, *args, **kwargs):
        predictions = cross_val_predict(self.model, self.features, self.labels, *args, **kwargs)

        fpr, tpr, thresholds = roc_curve(self.labels, predictions)

        return (fpr, tpr, thresholds, predictions)

    def __repr__(self):
        representation = super(BinaryClassifierModel, self).__repr__()

        if not self.evaluated:
            return f'{representation}: not applied'

        scores = 'auc: {:.2f}, sensitivity: {:.2f}, specificity: {:.2f}, precision: {:.2f}'.format(
            self.auc,
            self.sensitivity,
            self.specificity,
            self.precision
        )
        representation = '{0},\nsize: {1:.02f}, {2}'.format(
            representation,
            len(self.labels),
            scores
        )

        return representation
