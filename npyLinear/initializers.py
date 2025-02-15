import numpy as np
from typing import Optional


class Initializers:

    def __init__(self):
        pass

    def he_normal(self, W: np.ndarray, fan_in: int, bias: Optional[np.ndarray] = None) -> tuple:
        r"""
        He normal initializer.
        
        Parameters:
            W (np.ndarray): Weight array
            fan_in (int): Input size of neuron
            bias (Optional[np.ndarray]): Bias array
        
        Returns:
            out (tuple): Initialized weight and bias  
        """
        W = np.random.normal(0, np.sqrt(2 / fan_in), (W.shape)).astype(np.float32)
        if bias is not None:
            bias = np.random.normal(0, np.sqrt(2 / fan_in), (bias.shape)).astype(np.float32)
        return W, bias

    def he_uniform(self, W: np.ndarray, fan_in: int, bias: Optional[np.ndarray] = None) -> tuple:
        r"""
        He uniform initializer.
        
        Parameters:
            W (np.ndarray): Weight array
            fan_in (int): Input size of neuron
            bias (Optional[np.ndarray]): Bias array
        
        Returns:
            out (tuple): Initialized weight and bias  
        """
        W = np.random.uniform(-(np.sqrt(6 / fan_in)), np.sqrt(6 / fan_in), (W.shape)).astype(np.float32)
        if bias is not None:
            bias = np.random.uniform(-(np.sqrt(6 / fan_in)), np.sqrt(6 / fan_in), (bias.shape)).astype(np.float32)
        return W, bias

    def xavier_normal(self, W: np.ndarray, bias: Optional[np.ndarray] = None) -> tuple:
        r"""
        Xavier normal initializer.
        
        Parameters:
            W (np.ndarray): Weight array
            bias (Optional[np.ndarray]): Bias array
        
        Returns:
            out (tuple): Initialized weight and bias  
        """
        W = np.random.normal(0, np.sqrt(2 / (W.shape[0] + W.shape[1])), (W.shape)).astype(np.float32) # fan-in -> W.shape[0], fan-out -> W.shape[1]
        if bias is not None:
            bias = np.random.normal(0, np.sqrt(2 / (W.shape[0] + W.shape[1])), (bias.shape)).astype(np.float32)
        return W, bias

    def xavier_uniform(self, W: np.ndarray, bias: Optional[np.ndarray] = None) -> tuple:
        r"""
        Xavier uniform initializer.
        
        Parameters:
            W (np.ndarray): Weight array
            bias (Optional[np.ndarray]): Bias array
        
        Returns:
            out (tuple): Initialized weight and bias  
        """
        W = np.random.uniform(-(np.sqrt(6 / (W.shape[0] + W.shape[1]))), np.sqrt(6 / (W.shape[0] + W.shape[1])), (W.shape)).astype(np.float32)
        if bias is not None:
            bias = np.random.uniform(-(np.sqrt(6 / (W.shape[0] + W.shape[1]))), np.sqrt(6 / (W.shape[0] + W.shape[1])), (bias.shape)).astype(np.float32)
        return W, bias