import numpy as np  
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class MeanEstimator():

    def __init__(self, one_parameter=True):  
        self.one_parameter = one_parameter        

    def fit(self, X, y):
        if len(X.shape)==1:
            self.n_features_in_ = X.shape[0]
        else:
            self.n_features_in_ = X.shape[1]
        X,y = check_X_y(X,y)
        self.k_ = np.mean(y)
        return self
    
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return np.ones(len(X))*self.k_
    
    def score(self, X, y):
        return np.mean(np.abs(np.ones(len(X))*self.k_-y))
    
    def get_params(self, deep=True):
       return {"one_parameter":self.one_parameter}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
if __name__=='__main__':

    check_estimator(MeanEstimator())