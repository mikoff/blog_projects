from abc import ABC, abstractmethod

class Operation(ABC):
    """Base class for node operations.
    
    On forward pass the inherited class calculates the output of the 
    node and nulls the local gradient for future calculations.
    
    """
    
    def __init__(self, inputs):
        self.inputs = inputs
        self.output = None
        self.local_grad = 0.
        
    def forward(self, *args):
        """Calculates the node output given inputs"""
        self.local_grad = 0.
        self._forward(*args)
        
    @abstractmethod
    def _forward(self, *inputs):
        pass
    
    @abstractmethod
    def backward(self, dz):
        """Returns the local gradient."""
        pass


class Multiply(Operation):
    def _forward(self, x, y):
        self.x = x
        self.y = y
        self.output = x * y
        return self.output
    
    def backward(self, dz):
        dx = self.y * dz
        dy = self.x * dz
        return [dx, dy]


class Add(Operation):
    def _forward(self, x, y):
        self.output = x + y
        return self.output
    
    def backward(self, dz):
        return [dz, dz]


class Sub(Operation):
    def _forward(self, x, y):
        self.output = x - y
        return self.output
    
    def backward(self, dz):
        return [dz, -dz]


class Square(Operation):
    def _forward(self, x):
        self.x = x
        self.output = x * x
        return self.output
    
    def backward(self, dz):
        return [2 * self.x * dz]


class Sigmoid(Operation):
    def _forward(self, x):
        from numpy import exp
        self.output = 1. / (1. + exp(-x))
        return self.output
    
    def backward(self, dz):
        return [(1 - self.output) * self.output * dz]


class Div(Operation):
    def _forward(self, a, b):
        self.a = a
        self.b = b
        self.output = a / b
        return self.output
    
    def backward(self, dz):
        return [1 / self.b * dz, - self.a / self.b**2 * dz]


class Inv(Operation):
    def _forward(self, a):
        self.a = a
        self.output = 1 / a
        return self.output
    def backward(self, dz):
        return [-1. / self.a**2 * dz]


class Input(object):
    """Class for input values"""
    def __init__(self, value):
        self.output = value
        self.local_grad = 0.0
        

class ComputationalGraph:
    """Computational graph: stores computing nodes, gradients and inputs.
    
    The computational graph is used to calculate the output of each node
    on forward pass and to calculate the local gradients of each node with
    respect to the node inputs.
    """
    
    def __init__(self):
        self.operations = {}
        self.inputs = {}
        self.nodes = {}
        
    def add_inputs(self, **kwargs):
        for name, value in kwargs.items():
            self.inputs[name] = Input(value)
            self.nodes[name] = self.inputs[name]
            
    def add_computation_node(self, name, operation, inputs):
        self.operations[name] = operation(inputs)
        self.nodes[name] = self.operations[name]
        
    def _get_traverse_order(self, root):
        route = []
        queue = [root]
        while len(queue):
            prev = queue.pop(0)
            if prev not in self.inputs:
                route.append(prev)
            if prev in self.operations:
                queue.extend(self.operations[prev].inputs)
        return route
    
    def forward(self, root):
        for name in reversed(self._get_traverse_order(root)):
            node = self.nodes[name]
            node.forward(*[self.nodes[i].output for i in node.inputs])
            
        for inp in self.inputs:
            self.inputs[inp].local_grad = 0.0
            
    def backward(self, root):
        self.operations[root].local_grad = 1.
        for name in self._get_traverse_order(root):
            local_grad = self.nodes[name].local_grad
            inputs = self.nodes[name].inputs
            gradients_wrt_input = self.nodes[name].backward(local_grad)
            # roll back to inputs
            for parent, g in zip(inputs, gradients_wrt_input):
                self.nodes[parent].local_grad += g
                
    def get_gradient(self, node):
        if node in self.nodes:
            return self.nodes[node].local_grad
