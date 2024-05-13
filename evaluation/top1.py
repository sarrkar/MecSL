from sklearn.metrics import accuracy_score


class Top1:
    def __init__(self):
        self.y_true = []
        self.y_pred = []

    def add(self, y_true, y_pred):
        self.y_true.extend(y_true)
        self.y_pred.extend(y_pred)

    def evaluate(self):
        return accuracy_score(self.y_true, self.y_pred)

    def reset(self):
        self.y_true = []
        self.y_pred = []
