import os
import json
from datetime import datetime
import torch
from torch import nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Logger():
    def __init__(self,
                 exp_name: str='./runs',
                 filename: str=None):
        self.exp_name=exp_name
        self.cache={}
        if not os.path.exists(exp_name):
            os.makedirs(exp_name, exist_ok=True)
        self.date=datetime.today().strftime("%B_%d_%Y_%I_%M%p")
        if filename is None:
            self.filename=self.date
        else:
            self.filename="_".join([self.date, filename])
        fpath = f"{self.exp_name}/{self.filename}.json"
        with open(fpath, 'w') as f:
            data = json.dumps(self.cache)
            f.write(data)
        
    def add_scalar(self, key: str, value: float, t: int):
        if key in self.cache:
            self.cache[key][t] = value
        else:
            self.cache[key] = {t:value}
        self.update()
        return None
    
    def save_weights(self, state_dict, model_name: str='model'):
        fpath = f"{self.exp_name}/{model_name}.pt"
        torch.save(state_dict, fpath)
        return None
    
    def update(self,):
        fpath = f"{self.exp_name}/{self.filename}.json"
        with open(fpath, 'w') as f:
            data = json.dumps(self.cache)
            f.write(data)
        return None
    
    def close(self,):
        fpath = f"{self.exp_name}/{self.filename}.json"
        with open(fpath, 'w') as f:
            data = json.dumps(self.cache)
            f.write(data)
        self.cache={}
        return None