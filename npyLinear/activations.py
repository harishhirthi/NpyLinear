import numpy as np


class Activations:

    def __init__(self):
        pass

    def sigmoid(self, x: np.ndarray) -> tuple:
        r"""
        Sigmoid activation.
        
        Parameters:
            x (np.ndarray): Input array
        
        Returns:
            out (tuple): Activated array, activation string
        """
        return 1 / (1 + np.exp(-x)), 'sigmoid'
    
    def tanh(self, x: np.ndarray) -> tuple:
        r"""
        TanH activation.
        
        Parameters:
            x (np.ndarray): Input array
        
        Returns:
            out (tuple): Activated array, activation string
        """
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)), 'tanh'
    
    def relu(self, x: np.ndarray) -> tuple:
        r"""
        ReLU activation.
        
        Parameters:
            x (np.ndarray): Input array
        
        Returns:
            out (tuple): Activated array, activation string
        """
        return np.maximum(0, x), 'relu'
    
    def softmax(self, x: np.ndarray) -> tuple:
        r"""
        Softmax activation.
        
        Parameters:
            x (np.ndarray): Input array
        
        Returns:
            out (tuple): Activated array, activation string
        """
        out = np.exp(x - np.max(x, axis = 1, keepdims = True))  # Subtraction is performed to make the computation more stable and to prevent overflow or underflow issues
        return out / np.sum(out, axis = 1, keepdims = True), 'softmax'
    
    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        r"""
        Derivative of Sigmoid activation.
        
        Parameters:
            x (np.ndarray): Input array
        
        Returns:
            out (np.ndarray): Out array
        """
        return self.sigmoid(x)[0] * (1 - self.sigmoid(x)[0]) 

    def tanh_derivative(self, x: np.ndarray) -> np.ndarray:
        r"""
        Derivative of TanH activation.
        
        Parameters:
            x (np.ndarray): Input array
        
        Returns:
            out (np.ndarray): Out array
        """
        return 1 - (self.tanh(x)[0] ** 2)
    
    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        r"""
        Derivative of ReLU activation.
        
        Parameters:
            x (np.ndarray): Input array
        
        Returns:
            out (np.ndarray): Out array
        """
        return x > 0