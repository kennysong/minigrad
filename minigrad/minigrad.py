class Scalar:
    def __init__(self, value, parents=[], parent_op=None):
        self.value = value
        self.parents = parents
        self.parent_op = parent_op
        self.grad = 0           # This stores the ∂output/∂self value
        self.grad_wrt = dict()  # This stores all ∂self/∂parent values
                                # (only populated if self has parents, i.e. self was created by an arthimetic op)

    def __repr__(self):
        return f'Scalar(value={self.value:.2f}, grad={self.grad:.2f})' 
    
    # Called by: self + other
    def __add__(self, other):
        if not isinstance(other, Scalar): other = Scalar(other)
        output = Scalar(self.value + other.value, [self, other], '+')
        
        output.grad_wrt[self] = 1
        output.grad_wrt[other] = 1
        
        return output
    
    # Called by: other + self
    def __radd__(self, other):
        return self.__add__(other)
    
    # Called by: self - other
    def __sub__(self, other):
        if not isinstance(other, Scalar): other = Scalar(other)
        output = Scalar(self.value - other.value, [self, other], '-')
            
        output.grad_wrt[self] = 1
        output.grad_wrt[other] = -1 
        
        return output
    
    # Called by: other - self
    def __rsub__(self, other):
        if not isinstance(other, Scalar): other = Scalar(other)
        output = Scalar(other.value - self.value, [self, other], '-')
        
        output.grad_wrt[self] = -1
        output.grad_wrt[other] = 1
            
        return output
    
    # Called by: self * other
    def __mul__(self, other):
        if not isinstance(other, Scalar): other = Scalar(other)
        output = Scalar(self.value * other.value, [self, other], '*')
        
        output.grad_wrt[self] = other.value
        output.grad_wrt[other] = self.value

        return output
    
    # Called by: other * self
    def __rmul__(self, other):
        return self.__mul__(other)

    # Called by: self / other
    def __truediv__(self, other):
        if not isinstance(other, Scalar): other = Scalar(other)
        output = Scalar(self.value / other.value, [self, other], '/')
        
        output.grad_wrt[self] = 1 / other.value
        output.grad_wrt[other] = -self.value / other.value**2
        
        return output
    
    # Called by: other / self
    def __rtruediv__(self, other):
        if not isinstance(other, Scalar): other = Scalar(other)
        output = Scalar(other.value / self.value, [self, other], '/')
        
        output.grad_wrt[self] = -other.value / self.value**2
        output.grad_wrt[other] = 1 / self.value
            
        return output
  
    # Called by: self**other
    def __pow__(self, other):
        assert isinstance(other, (int, float)), 'minigrad does not support a Scalar in the exponent'
        output = Scalar(self.value ** other, [self], f'^{other}')
        
        output.grad_wrt[self] = other * self.value**(other - 1)

        return output
    
    # Called by: -self
    def __neg__(self):
        return self.__mul__(-1)
    
    # Called by: self.relu()
    def relu(self):
        output = Scalar(max(self.value, 0), [self], 'relu')
        
        output.grad_wrt[self] = int(self.value > 0)
        
        return output
    
    def backward(self):
        '''Compute ∂self/∂node, i.e. ∂output/∂node, for each node in self's dependency graph.'''

        # Note: To properly do reverse-mode autodiff, we need to traverse the DAG 
        # exactly in order from output to input, hitting each node once, and completing
        # the gradient computation (see _compute_grad_of_parents()) in that single step. 
        # A (reversed) topological sort will give us that DAG ordering.
        # Originally, I recursed through the DAG depth-first, which led to wrong
        # gradient calculations when inputs were computed before their outputs.
        def _topological_order():
            '''Returns the topological ordering of self's dependencies.'''
            def _add_parents(node):
                if node not in visited:
                    visited.add(node)
                    for parent in node.parents:
                        _add_parents(parent)
                    ordered.append(node)

            ordered, visited = [], set()
            _add_parents(self)
            return ordered

        def _compute_grad_of_parents(node):
            '''Given a node, compute its parents' gradients: ∂output/∂parent = ∂output/∂node * ∂node/∂parent.'''
            for parent in node.parents:
                # By the time _backward() is called on node, we've already computed ∂output_∂node
                Δoutput_Δnode = node.grad  # Python doesn't support ∂ in variable names :(
                
                # We've also already computed ∂node_∂parent, when node was created as the output of an arithmetic operation
                Δnode_Δparent = node.grad_wrt[parent]
                
                # Last, compute and store the value of ∂output/∂parent = ∂output/∂node * ∂node/∂parent
                #
                # It's actually += here, since a node can be a parent of multiple downstream nodes,
                # and we need to properly accumulate all of their gradients. That is,
                # ∂output/∂parent = Σ_i ∂output/∂node_i * ∂node_i/∂parent
                #                    ^ for all node_i that are downstream of parent
                parent.grad += Δoutput_Δnode * Δnode_Δparent
                
        
        # ∂output/∂output = 1; this bootstraps the backpropagation
        self.grad = 1
        
        # Now traverse through the graph in order from output to input, and compute gradients!
        ordered = reversed(_topological_order())
        for node in ordered:
            _compute_grad_of_parents(node)
