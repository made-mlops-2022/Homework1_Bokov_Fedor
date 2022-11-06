import numpy as np
import pandas as pd


class transformer:
    def __init__(self):
        self.means = np.zeros(1)
        self.vars = np.ones(1)
        self.fitted = False

    def fit(self, X):
        arr = X.values if isinstance(X, pd.DataFrame) else X
        self.means = arr.mean(axis=0)
        self.vars = arr.var(axis=0)
        self.vars[self.vars == 0] = 1
        self.fitted = True
    
    def transform(self, X):
        if X is None or not self.fitted:
            raise ValueError("Transformer is not fitted")
        arr = X.values if isinstance(X, pd.DataFrame) else X
        res = (arr - self.means) / self.vars
        self.fitted = True
        return res

if __name__ =="__main__":
    pass
