import numpy as np
from npyLinear import (Linear, Activations)
from collections import defaultdict
from typing import Optional
import os
import pickle


class Model:

    def __init__(self, bias: Optional[bool] = True) -> None:
        self.bias_flag = bias
        self.l1 = Linear(784, 200, bias = self.bias_flag, initializer = "he_normal")
        self.l2 = Linear(200, 150, bias = self.bias_flag, initializer = "he_normal")
        self.l3 = Linear(150, 100, bias = self.bias_flag, initializer = "he_normal")
        self.l4 = Linear(100, 10, bias = self.bias_flag, initializer = "xavier_normal")
        self.act = Activations()
        self.cache = defaultdict(dict)
    
    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        l1_out = self.l1.forward(x)
        out, act_str = self.act.relu(l1_out)
        self.cache["Linear 1"] = {"W": self.l1.W, "B": self.l1.bias, "l_out": l1_out, "a_out": out, "activation": act_str}

        l2_out = self.l2.forward(out)
        out, act_str = self.act.relu(l2_out)
        self.cache["Linear 2"] = {"W": self.l2.W, "B": self.l2.bias, "l_out": l2_out, "a_out": out, "activation": act_str}

        l3_out = self.l3.forward(out)
        out, act_str = self.act.relu(l3_out)
        self.cache["Linear 3"] = {"W": self.l3.W, "B": self.l3.bias, "l_out": l3_out, "a_out": out, "activation": act_str}

        l4_out = self.l4.forward(out)
        out, act_str = self.act.softmax(l4_out)
        self.cache["Linear 4"] = {"W": self.l4.W, "B": self.l4.bias, "l_out": l4_out, "a_out": out, "activation": act_str}

        return out, self.cache
    
    def update(self, W: dict, B: Optional[dict] = None) -> None:
        if self.bias_flag:
            self.l1.update(W["Linear 1"], B["Linear 1"])
            self.l2.update(W["Linear 2"], B["Linear 2"])
            self.l3.update(W["Linear 3"], B["Linear 3"])
            self.l4.update(W["Linear 4"], B["Linear 4"])
        else:
            self.l1.update(W["Linear 1"])
            self.l2.update(W["Linear 2"])
            self.l3.update(W["Linear 3"])
            self.l4.update(W["Linear 4"])

    def __sizeof__(self) -> int:
        return self.l1.__sizeof__() + self.l2.__sizeof__() + self.l3.__sizeof__() + self.l4.__sizeof__()
    
    def params(self) -> int:
        return self.l1.params() + self.l2.params() + self.l3.params() + self.l4.params()
    
    def accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.argmax(y_true, axis = 1) == np.argmax(y_pred, axis = 1))
    
    def save(self, filename: os.path) -> None:
        if self.bias_flag:
            all_values = {'Linear 1': {'W': self.l1.W, 'B': self.l1.bias}, 'Linear 2': {'W': self.l2.W, 'B': self.l2.bias},
                          'Linear 3': {'W': self.l3.W, 'B': self.l3.bias}, 'Linear 4': {'W': self.l4.W, 'B': self.l4.bias}}
        else:
            all_values = {'Linear 1': self.l1.W, 'Linear 2': self.l2.W, 'Linear 3': self.l3.W, 'Linear 4': self.l4.W}
        with open(f'{filename}', 'wb') as fp:
            pickle.dump(all_values, fp)
            print(f"All parameters are saved in {filename}")
            print()
    
    def load_dict(self, filename: os.path) -> None:
        with open(f'{filename}', 'rb') as fp:
            all_values = pickle.load(fp)
            if self.bias_flag:
                self.l1.update(all_values['Linear 1']['W'], all_values['Linear 1']['B'])
                self.l2.update(all_values['Linear 2']['W'], all_values['Linear 2']['B'])
                self.l3.update(all_values['Linear 3']['W'], all_values['Linear 3']['B'])
                self.l4.update(all_values['Linear 4']['W'], all_values['Linear 4']['B'])
                print("All Weights and Bias loaded successfully")
            else:
                self.l1.update(all_values['Linear 1']['W'])
                self.l2.update(all_values['Linear 2']['W'])
                self.l3.update(all_values['Linear 3']['W'])
                self.l4.update(all_values['Linear 4']['W'])
                print("All Weights loaded successfully")


if __name__ == '__main__':

    model = Model()
    print(f"Number of parameters: {model.params() / 1e3}K")