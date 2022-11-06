from src.transformer import transformer
import numpy as np

def test_trans():
    X = np.array([[1,2,3], [2,3,4], [3,4,5]])
    
    scaler = transformer()
    scaler.fit(X)
    res = scaler.transform(X)
    print(res)
    assert (res == np.array([[-1.5, -1.5, -1.5], [ 0,  0, 0], [ 1.5,  1.5,  1.5]])).all()
