

import numpy as np

class svr(object):
    def __init__(self,params: dict, **kwargs) -> None:
        print("soy el modelo svr")      
        self.layers = params.get("layers",[10,10])
    def train(self):
        pass
    def create_model(self):
        pass
    def predict(self):
        pass



# Unit testing
if __name__ == "__main__":
    import ipdb
    pass