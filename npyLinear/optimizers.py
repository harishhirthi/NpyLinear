import numpy as np
from typing import Optional


class Optimizers:
    
    def __init__(self):
        pass

    def update(self, W: np.ndarray, eta: float, dW: np.ndarray) -> np.ndarray:
        W -= eta * dW
        return W
    
class Adam(Optimizers):
    
    def __init__(self, beta_1: float, beta_2: float, lr: float):
        r"""
        Optimization of weights and bias using Adam optimizer.
        
        Args:
            beta_1 (float): Beta1 value
            beta_2 (float): Beat2 value
            lr (float): Learning rate
        """
        super(Adam, self).__init__()
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = 1e-8
        self.W_m_cache = {}
        self.B_m_cache = {}
        self.W = {}
        self.B = {}
        self.lr = lr

    def calc(self, t: int, dW: np.ndarray) -> np.ndarray:
        r"""
        Calculation of first order and second order moments.
        
        Parameters:
            t (int): Iteration number
            dW (np.ndarray): Gradient array
        
        Returns:
            out (np.ndarray): out array  
        """
        self.m_t = (self.beta_1 * self.m_t) + ((1 - self.beta_1) * dW)
        self.v_t = (self.beta_2 * self.v_t) + ((1 - self.beta_2) * (dW ** 2))
        m_hat = self.m_t / (1 - (self.beta_1 ** t))
        v_hat = self.v_t / (1 - (self.beta_2 ** t))
        out = m_hat / (np.sqrt(v_hat) + self.eps)
        return out
    
    def step(self, t: int, dW: dict, cache: dict, dB: Optional[dict] = None) -> dict:
        r"""
        Step function.
        
        Parameters:
            t (int): Iteration number
            dW (dict): Dictionary of weights' gradients
            cache (dict): Dictionary of intermediate results of forward pass
            dB (Optional[dict]): Dictionary of bias gradients
        
        Returns:
            W (dict): Dictionary of optimized weights' 
            B Optional[dict]: Dictionary of optimized bias
        """
        for name, array in dW.items():
            if t == 1: # Initialize zeros for m(t) and v(t) at first iteration.
                self.m_t = np.zeros_like(array)
                self.v_t = np.zeros_like(array)
            else:
                self.m_t = self.W_m_cache[name]['m_t']
                self.v_t = self.W_m_cache[name]['v_t']
            eta = self.calc(t, array)
            self.W[name] = self.update(W = cache[name]["W"], eta = self.lr, dW = eta)   
            self.W_m_cache[name] = {'m_t': self.m_t, 'v_t': self.v_t}
        
        if dB is not None:
            for name, array in dB.items():
                if t == 1:
                    self.m_t = np.zeros_like(array)
                    self.v_t = np.zeros_like(array)
                else:
                    self.m_t = self.B_m_cache[name]['m_t']
                    self.v_t = self.B_m_cache[name]['v_t']
                eta = self.calc(t, array)
                self.B[name] = self.update(cache[name]["B"], self.lr, eta)   
                self.B_m_cache[name] = {'m_t': self.m_t, 'v_t': self.v_t}
            return self.W, self.B
        
        else:
            return self.W, None

    
class RMS_prop(Optimizers):

    def __init__(self, gamma: float, lr: float):
        r"""
        Optimization of weights and bias using RMS-Prop optimizer.
        
        Args:
            gamma (float): Gamma value
            lr (float): Learning rate
        """
        super(RMS_prop, self).__init__()
        self.gamma = gamma
        self.eps = 1e-3
        self.W_E_cache = {}
        self.B_E_cache = {}
        self.W = {}
        self.B = {}
        self.lr = lr

    def calc(self, dW: np.ndarray) -> np.ndarray:
        r"""
        Calculation of E[g].
        
        Parameters:
            dW (np.ndarray): Gradient array
        
        Returns:
            out (np.ndarray): out array  
        """
        self.E_g = (self.gamma * self.E_g) + ((1 - self.gamma) * (dW ** 2))
        out = 1 / (np.sqrt(self.E_g + self.eps))
        return out
    
    def step(self, t: int, dW: dict, cache: dict, dB: Optional[dict] = None) -> dict:
        r"""
        Step function.
        
        Parameters:
            t (int): Iteration number
            dW (dict): Dictionary of weights' gradients
            cache (dict): Dictionary of intermediate results of forward pass
            dB (Optional[dict]): Dictionary of bias gradients
        
        Returns:
            W (dict): Dictionary of optimized weights' 
            B Optional[dict]: Dictionary of optimized bias
        """
        for name, array in dW.items():
            if t == 1: # Initialize zeros for E[g] at first iteration.
                self.E_g = np.zeros_like(array)
            else:
                self.E_g = self.W_E_cache[name]['E_g']
            eta = self.calc(array)
            self.W[name] = self.update(cache[name]["W"], self.lr * eta, array)   
            self.W_E_cache[name] = {'E_g': self.E_g}
        
        if dB is not None:
            for name, array in dB.items():
                if t == 1:
                    self.E_g = np.zeros_like(array)
                else:
                    self.E_g = self.B_E_cache[name]['E_g']
                eta = self.calc(array)
                self.B[name] = self.update(cache[name]["B"], self.lr * eta, array)   
                self.B_E_cache[name] = {'E_g': self.E_g}
            return self.W, self.B
        else:
            return self.W, None

    
class Adagrad(Optimizers):

    def __init__(self, lr: float):
        r"""
        Optimization of weights and bias using Adagrad optimizer.
        
        Args:
            lr (float): Learning rate
        """
        super(Adagrad, self).__init__()
        self.eps = 1e-8
        self.W_a_cache = {}
        self.B_a_cache = {}
        self.lr = lr
        self.W = {}
        self.B = {}

    def calc(self, dW: np.ndarray) -> np.ndarray:
        r"""
        Calculation of alpha.
        
        Parameters:
            dW (np.ndarray): Gradient array
        
        Returns:
            out (np.ndarray): out array  
        """
        self.alpha += (dW ** 2)
        out = 1 / np.sqrt(self.alpha + self.eps)
        return out

    def step(self, t: int, dW: dict, cache: dict, dB: Optional[dict] = None) -> dict:
        r"""
        Step function.
        
        Parameters:
            t (int): Iteration number
            dW (dict): Dictionary of weights' gradients
            cache (dict): Dictionary of intermediate results of forward pass
            dB (Optional[dict]): Dictionary of bias gradients
        
        Returns:
            W (dict): Dictionary of optimized weights' 
            B Optional[dict]: Dictionary of optimized bias
        """
        for name, array in dW.items():
            if t == 1: # Initialize zeros for alpha at first iteration.
                self.alpha = np.zeros_like(array)
            else:
                self.alpha = self.W_a_cache[name]['alpha']
            eta = self.calc(array)
            self.W[name] = self.update(cache[name]["W"], self.lr * eta, array)   
            self.W_a_cache[name] = {'alpha': self.alpha}

        if dB is not None:
            for name, array in dB.items():
                if t == 1:
                    self.alpha = np.zeros_like(array)
                else:
                    self.alpha = self.B_a_cache[name]['alpha']
                eta = self.calc(array)
                self.B[name] = self.update(cache[name]["B"], self.lr * eta, array)   
                self.B_a_cache[name] = {'alpha': self.alpha}
            return self.W, self.B
        else:
            return self.W, None
    
