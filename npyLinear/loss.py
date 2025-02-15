import numpy as np
from typing import Optional
from typing import Callable


class Loss:
    
    def __init__(self):
        pass

    @staticmethod
    def BCE_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
       r"""
        Binary Cross-Entropy loss for binary classification.
        
        Parameters:
            y_true (np.ndarray): True labels 
            y_pred (np.ndarray): Predicted labels

        Returns:
            value (float): Calculated loss
        """
       out = -((y_true * np.log(y_pred + 1e-9)) + ((1 - y_true) * np.log(1 - (y_pred + 1e-9)))) 
       return np.mean(out)

    @staticmethod
    def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        r"""
        Cross-Entropy loss for multi-class classification.
        
        Parameters:
            y_true (np.ndarray): One-hot encoded true labels 
            y_pred (np.ndarray): Predicted labels

        Returns:
            value (float): Calculated loss
        """
        out = np.sum(y_true * np.log(y_pred + 1e-9), axis = 1)
        return -np.mean(out)
    
    def loss_derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        r"""
        Derivative of entropy loss.
        
        Parameters:
            y_true (np.ndarray): True labels 
            y_pred (np.ndarray): Predicted labels

        Note:
            y_true should be one-hot encoded incase of multi-class classification.

        Returns:
            value (float): Calculated loss
        """
        return y_pred - y_true
    
    def choose_derivative(self, act_obj: object, activation: str) -> Callable:
        r"""
        Function to choose the derivative of activation used in the respective layer.
        
        Parameters:
            act_obj (Object): Object of activation class 
            activation (str): Name of the activation

        Returns:
            func (fuction): Derivative function of respective activation
        """
        if activation == 'relu':
            return act_obj.relu_derivative
        if activation == 'tanh':
            return act_obj.tanh_derivative
        if activation == 'sigmoid':
            return act_obj.sigmoid_derivative

    
    def backward_W(self, cache: dict, y_true: np.ndarray, act: object, input: np.ndarray, lambda_reg: Optional[float] = 0.0) -> dict: 
        r"""
        Backward pass of Weights.
        
        Parameters:
            cache (dict): Dictionary of intermediate results of forward pass 
            y_true (np.ndarray): True labels
            act (object): Object of activation class
            input (nd.array): Input array of first neuron
            lambda_reg (Optional[float]): Regularization strength value

        Note:
            y_true should be one-hot encoded incase of multi-class classification.

        Returns:
            dW (dict): Dictionary of gradients of respective weights'
        """
        # N = input.shape[0]
        layers = list(cache.keys())
        y_pred = cache[layers[-1]]['a_out']
        dL_dyhat = self.loss_derivative(y_true, y_pred)
        dW = {}
        # Gradient of weight of last layer
        dW[layers[-1]] = np.dot(cache[layers[-2]]['a_out'].T, dL_dyhat) +  (lambda_reg * cache[layers[-1]]['W']) # (Loss derivative * Activation output) + Regularization derivative
        if len(cache) == 1:
            return dW
        else:
            act_der_func = self.choose_derivative(act, cache[layers[-2]]['activation'])
            calc = lambda dl, W, act_der: np.dot(dl, W.T) * act_der_func(act_der)
            initial = calc(dL_dyhat, cache[layers[-1]]['W'], cache[layers[-2]]['l_out']) # Loss derivative * W * Activation derivative
            rem_layers = layers[:-1][::-1]
            for i, layer in enumerate(rem_layers):
                if i == len(rem_layers) - 1: # Gradient of weight of first layer
                    out = np.dot(input.T, initial)
                    dW[layer] = out + (lambda_reg * cache[rem_layers[i]]['W'])
                else: # Gradient of weights of hidden layers [Multiplication of chain of gradients]
                    out = np.dot(cache[rem_layers[i + 1]]["a_out"].T, initial)
                    act_der_func = self.choose_derivative(act, cache[rem_layers[i + 1]]['activation'])
                    initial = calc(initial, cache[rem_layers[i]]['W'], cache[rem_layers[i + 1]]['l_out'])
                    dW[layer] = out + (lambda_reg * cache[rem_layers[i]]['W'])
            return dW
        
    def backward_b(self, cache: dict, y_true: np.ndarray, act: object, lambda_reg: Optional[float] = 0.0) -> dict:
        r"""
        Backward pass of Bias.
        
        Parameters:
            cache (dict): Dictionary of intermediate results of forward pass 
            y_true (np.ndarray): True labels
            act (object): Object of activation class
            lambda_reg (Optional[float]): Regularization strength value

        Note:
            y_true should be one-hot encoded incase of multi-class classification.

        Returns:
            dB (dict): Dictionary of gradients of respective biases
        """
        layers = list(cache.keys())
        y_pred = cache[layers[-1]]['a_out']
        dL_dyhat = self.loss_derivative(y_true, y_pred)
        dB = {}
        # Gradient of bias of last layer
        dB[layers[-1]] = np.sum(dL_dyhat, axis = 0, keepdims = True) + (lambda_reg * cache[layers[-1]]['B']) # Loss derivative + Regularization derivative
        if len(cache) == 1:
            return dB
        else:
            act_der_func = self.choose_derivative(act, cache[layers[-2]]['activation'])
            calc = lambda dl, W, act_der: np.dot(dl, W.T) * act_der_func(act_der)
            initial = calc(dL_dyhat, cache[layers[-1]]['W'], cache[layers[-2]]['l_out']) # Loss derivative * W * Activation derivative
            rem_layers = layers[:-1][::-1]
            for i, layer in enumerate(rem_layers): 
                if i == len(rem_layers) - 1: # Gradient of bias of first layer
                    out = initial
                    dB[layer] = np.sum(out, axis = 0, keepdims = True) + (lambda_reg * cache[rem_layers[i]]['B'])
                else: # Gradient of bias of hidden layers [Multiplication of chain of gradients]
                    out = initial
                    act_der_func = self.choose_derivative(act, cache[rem_layers[i + 1]]['activation'])
                    initial = calc(initial, cache[rem_layers[i]]['W'], cache[rem_layers[i + 1]]['l_out'])
                    dB[layer] = np.sum(out, axis = 0, keepdims = True) + (lambda_reg * cache[rem_layers[i]]['B'])
            return dB
        
    def backward(self, cache: dict, y_true: np.ndarray, act: object, input: np.ndarray, lambda_reg: Optional[float] = 0.0, bias_flag: Optional[bool] = False) -> dict:
        r"""
        Backward pass of Weights and Biases.
        
        Parameters:
            cache (dict): Dictionary of intermediate results of forward pass 
            y_true (np.ndarray): True labels
            act (object): Object of activation class
            input (nd.array): Input array of first neuron
            lambda_reg (Optional[float]): Regularization strength value
            bias_flag (Optional[bool]): To include backward pass for bias

        Note:
            y_true should be one-hot encoded incase of multi-class classification.

        Returns:
            dW (dict): Dictionary of gradients of respective weights' 
            dB Optional[dict]: Dictionary of gradients of respective biases
        """
        dW = self.backward_W(cache, y_true, act, input, lambda_reg)
        if bias_flag:
            dB = self.backward_b(cache, y_true, act, lambda_reg)
            return dW, dB
        else:
            return dW, None
