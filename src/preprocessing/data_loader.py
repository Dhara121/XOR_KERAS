import pandas as pd

class DataLoader:
    def __init__(self):
        self.training_data = pd.DataFrame(data={
            "X1": [0, 0, 1, 1],
            "X2": [0, 1, 0, 1],
            "Y": [0, 1, 1, 0]
        })

    def load_data(self):
        X_train = self.training_data.iloc[:, 0:2]
        Y_train = self.training_data["Y"]
        return X_train, Y_train


def load_data():
    training_data = pd.DataFrame(data={
        "X1": [0, 0, 1, 1],
        "X2": [0, 1, 0, 1],
        "Y": [0, 1, 1, 0]
    })
    X_train = training_data.iloc[:, 0:2]
    Y_train = training_data["Y"]
    return X_train, Y_train
