from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import scipy.sparse as sparse

class EsnClassifier(BaseEstimator, ClassifierMixin):
    """Echo state network classifier"""
    def __init__(self, \
                 density=1, reservoirSize=100, outputleakingRate=1, \
                 inputSize=1, outputSize=1, leakingRate = 1, \
                 randomState=None, regularizationCoefficient=10e-6,
                 alpha=None):
        """
        Called when initializing the classifier
        """
        self.density = density
        self.reservoirSize = reservoirSize
        self.leakingRate = leakingRate
        self.randomState = randomState
        self.outputSize = outputSize
        self.inputSize = inputSize
        self.alpha = alpha
        self.regularizationCoefficient = regularizationCoefficient
    
    def fit(self, X, y=None):
        """
        """
        # FIXME: add Asserts or try/catch
        
        examples, sequenceLength = X.shape
        self.Win_, self.W_ = self.build_reservoirs()
    
        bias = np.ones((1, examples))
        
        # run the reservoir with the data and collect X
        x = np.zeros((self.reservoirSize,examples))
        for pic in range(sequenceLength):
            u = X[:, pic]
            x = (1-self.leakingRate)*x + self.leakingRate*np.tanh( np.dot( self.Win_, np.vstack((bias,u)) ) + np.dot( self.W_, x ) )
            print(pic, end="\r")
        
        # Reservoir values
        self.X = np.vstack((bias,x))
        self.y = y
        # Fit linear regression
        self.refit(self.regularizationCoefficient)
        return self
    
    def refit(self, regularizationCoefficient):
        """
        Fit regression with parameter regularizationCoefficient
        """
        self.Wout_ = np.dot( np.dot(self.y.T,self.X.T), np.linalg.inv( np.dot(self.X,self.X.T) + \
            regularizationCoefficient*np.eye(1+self.reservoirSize) ) ) 
        return self
    
    def predict(self, X, y=None):
        '''
        '''
        examples, sequenceLength = X.shape
        x = np.zeros((self.reservoirSize,examples))
        bias = np.ones((1, examples))
        for pix in range(sequenceLength):
            u = X[:, pix]
            x = (1-self.leakingRate)*x + self.leakingRate*np.tanh( np.dot( self.Win_, np.vstack((bias,u)) ) + np.dot( self.W_, x ) )
            print(pix, end="\r")
            
        y = np.dot( self.Wout_, np.vstack((bias,x)) ).T 
        return np.array(np.argmax(y, axis=1))
    
    
    # Helpers to build reservoir
    def __spectral_radius(self, matrix):
        '''
        Calculate spectral radius of matrix. 
        Spectral radius is max absolute eigenvalue.
        '''
        # FIXME: remove code below
        inner = matrix
        eigenvalues = np.linalg.eig(inner)[0]
        return max(abs(eigenvalues))
    
    def build_reservoirs(self):
        '''
        Generate reservoirs
        '''
        # FIXME: move to spartial
        
        # include bias term
#         if self.alpha is None:
        Win =  sparse.rand(self.reservoirSize, self.inputSize + 1, density=self.density, random_state=self.randomState)
        Win -= (Win.sign()*0.5)
        Win = Win.toarray() * self.alpha
#         Win = Win.toarray() * self.alpha
#         else:
#             Win = np.ones((self.reservoirSize, self.inputSize + 1)) * self.alpha

        W = sparse.rand(self.reservoirSize, self.reservoirSize, density=self.density, random_state=self.randomState)
        W -= W.sign()*0.5
        W *= 1.25/self.__spectral_radius(W.toarray())
        return (Win, W.toarray())