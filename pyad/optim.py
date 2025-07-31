import numpy as np


class Optimizer:
    def step(self):
        pass
    

class SGD(Optimizer):
    def __init__(self, parameters, lr, momentum=0, weight_decay=0, damp=0, nesterov=False):
        # parameters is a List of Tensors with data and gradient
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = damp
        self.nesterov = nesterov
        self.v = []
        for p in parameters:
            self.v.append(np.zeros_like(p.data))
        self.t = 0
        
    def step(self):
        self.t += 1
        for i,p in enumerate(self.parameters):
            # gt is p.grad
            # thetat is p.data
            
            if self.weight_decay != 0:
                p.grad = p.grad + self.weight_decay*p.data
            if self.momentum != 0:
                if self.t > 1:
                    self.v[i] = self.momentum*self.v[i] + (1-self.dampening)*p.grad
                else:
                    self.v[i] = p.grad
                if self.nesterov:
                    p.grad = p.grad + self.momentum*self.v[i]
                else:
                    p.grad = self.v[i]
            
            # this is wrong!!!!!!!!!!!!
            # wrong shape gets broadcasted
            # appears in batchnorm and layernorm
            # pretty bad -> always have to use same batch size in testing and training
            # dont know how to fix it properly
            p.data = p.data - self.lr * p.grad
            # p.data = p.data - self.lr * p.grad.sum(axis=0)
            # would fix it but feel like it has significant impact on performance

class Adam(Optimizer):
    
    def __init__(self, parameters, alpha=0.001, beta1=0.9, beta2=0.999):
        self.parameters = parameters
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        
        self.m = []
        self.v = []
        for p in parameters:
            self.m.append(np.zeros_like(p.data))
            self.v.append(np.zeros_like(p.data))
        self.t = 0
        
    def step(self):
        self.t += 1
        
        for i,p in enumerate(self.parameters):
            self.m[i] = self.beta1 * self.m[i] + (1-self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1-self.beta2) * (p.grad**2)
            
            m_hat = self.m[i] / (1-(self.beta1**self.t))
            v_hat = self.v[i] / (1-(self.beta2**self.t))
            
            p.data = p.data - self.alpha * m_hat/(np.sqrt(v_hat) + 10e-8)
            
class AdamW(Optimizer):
    
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01, amsgrad=False):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        
        self.m = []
        self.v = []
        self.v_max = []
        for p in parameters:
            self.m.append(np.zeros_like(p.data))
            self.v.append(np.zeros_like(p.data))
            self.v_max.append(np.zeros_like(p.data))
        self.t = 0
        
    def step(self):
        self.t += 1
        
        for i,p in enumerate(self.parameters):
            
            p.data = p.data - self.lr*self.weight_decay*p.data
            
            self.m[i] = self.beta1 * self.m[i] + (1-self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1-self.beta2) * (p.grad**2)
            
            m_hat = self.m[i] / (1-(self.beta1**self.t))
            v_hat = self.v[i] / (1-(self.beta2**self.t))
            
            if self.amsgrad:
                self.v_max[i] = np.maximum(self.v_max[i],v_hat)
                p.data = p.data - self.lr*m_hat/(np.sqrt(self.v_max[i]) + 10e-8)    
            else:
                p.data = p.data - self.lr * m_hat/(np.sqrt(v_hat) + 10e-8)            

            
class AdaMax(Optimizer):
    
    def __init__(self, parameters, alpha=0.002, beta1=0.9, beta2=0.999):
        self.parameters = parameters
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        
        self.m = []
        self.u = []
        for p in parameters:
            self.m.append(np.zeros_like(p.data))
            self.u.append(np.zeros_like(p.data))
        self.t = 0
        
    def step(self):
        self.t += 1
        
        for i,p in enumerate(self.parameters):
            self.m[i] = self.beta1 * self.m[i] + (1-self.beta1) * p.grad
            self.u[i] = np.maximum(self.beta2 * self.u[i], np.abs(p.grad) + 10e-8)
            
            p.data = p.data - (self.alpha/(1-(self.beta1**self.t))) * (self.m[i]/(self.u[i]))