import numpy as np
from typing import Optional
from .initializers import Initializers

class Linear:

    def __init__(self, in_feat: int, out_feat: int, bias: Optional[bool] = False, initializer: Optional[str] = None):
        r"""
        Primary linear layer that initializes weights and biases and performs matrix multiplication with input. It can also be 
        intialized with various available intializers.

        Args:
            in_feat (int): Input size of weight
            out_feat (int): Output size of weight
            bias (Optional[bool]): To include bias
            initializer (Optional[str], [he_normal, he_uniform, xavier_normal, xavier_uniform]): To initialize weights and bias using pre-defined initializers 
           
        """    
        self.in_ = in_feat
        self.out_ = out_feat
        self.W = np.random.randn(self.in_, self.out_).astype(np.float32)
        self.bias_flag = bias
        self.bias = None
        if self.bias_flag:
            self.bias = np.random.randn(1, self.out_).astype(np.float32)
        if initializer is not None:
            self.initialize(initializer)
    
    def __sizeof__(self):
        if self.bias_flag:
            return self.W.__sizeof__() + self.bias.__sizeof__()
        else:
            return self.W.__sizeof__()
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        r"""
        Performs forward pass of weights and bias with input x.
        
        Parameters:
            x (np.ndarray): Input array
        
        Returns:
            out (np.ndarray): Output array of forward pass
        """
        if self.bias_flag:
            out = x @ self.W + self.bias
        else:
            out = x @ self.W
        return out
    
    def initialize(self, init_str: str) -> None:
        r"""
        Initialize weights and bias with pre-defined initializers.
        
        Parameters:
            init_string (str): Name of the initializer 
        """
        init = Initializers()
        if init_str == "he_normal":
            if self.bias_flag:
                self.W, self.bias = init.he_normal(self.W, self.W.shape[0], self.bias)
            else:
                self.W, _ = init.he_normal(self.W, self.W.shape[0])
        if init_str == "he_uniform":
            if self.bias_flag:
                self.W, self.bias = init.he_normal(self.W, self.W.shape[0], self.bias)
            else:
                self.W, _ = init.he_normal(self.W, self.W.shape[0])
        if init_str == "xavier_normal":
            if self.bias_flag:
                self.W, self.bias = init.xavier_normal(self.W, self.bias)
            else:
                self.W, _ = init.xavier_normal(self.W)
        if init_str == "xavier_uniform":
            if self.bias_flag:
                self.W, self.bias = init.xavier_uniform(self.W, self.bias)
            else:
                self.W, _ = init.xavier_uniform(self.W)

    def update(self, W: np.ndarray, bias: Optional[np.ndarray] = None) -> None:
        r"""
        Update old weights and bias with optimized values.
        
        Parameters:
            W (np.ndarray): Optimized weights
            B (np.ndarray): Opimized bias  
        """
        self.W = W
        if self.bias_flag:
            self.bias = bias

    def params(self) -> int:
        r"""
        Returns the total number of params.
        
        Returns:
            n_params (int): Total number of params  
        """
        n_params = self.W.shape[0] * self.W.shape[1]
        if self.bias_flag:
            n_params += self.bias.shape[1]
        return n_params