#!/usr/bin/env python

import math
import random

class LSTM:
    def __init__(self, input_dimension):
        self.input_dimension = input_dimension
        self.input_weights = [.5] * self.input_dimension
        self.input_gate_weights = [.5] * self.input_dimension
        self.persist_gate_weights = [.5] * self.input_dimension
        self.output_gate_weights = [.5] * self.input_dimension
        self.sum_node_weights = [.5] * 2
        
        self.input_node = 0
        self.input_gate = 0
        self.persist_gate = 0
        self.output_gate = 0

        self.sum_node = 0
        self.mult_output_node = 0
        self.mult_loop_node = 0
        self.mult_input_node = 0
        
    def dp(self, x, y):
        # dot product
        assert(len(x) == len(y))
        sum = 0
        for i in range(len(x)):
            sum += x[i] * y[i]
        return sum

    def sigm(self, x):
        return (1.0 / (1.0 + math.exp(-x)))

    def timeStep(self, x):
        # assume x is an array of dimension input_dimension
        
        # Calculate backwards? This is just a guess on how this is implemented
        output = self.mult_output_node
        self.mult_output_node = self.sum_node * self.output_gate
        self.output_gate = self.sigm( self.dp( x, self.output_gate_weights ))
        old_sum_node = self.sum_node
        self.sum_node = self.sum_node_weights[0] * self.mult_input_node + self.sum_node_weights[1] * self.mult_loop_node
        self.mult_loop_node = old_sum_node * self.persist_gate
        self.output_gate = self.sigm( self.dp( x, self.persist_gate_weights ))
        self.mult_input_node = self.input_node * self.input_gate
        self.input_node = self.sigm( self.dp( x, self.input_weights ))
        self.input_gate = self.sigm( self.dp( x, self.input_gate_weights))
        return output 

    def runSequence(self, sequence):
        for input in sequence:
            print self.timeStep(input)

    def trainBP(self):
        # Back propagation through time
        pass

    def trainHF(self):
        # Hessian free optimization
        pass

def randomVector(dimension):
    vector = []
    for i in range(dimension):
        vector.append( 2 * (random.random() - .5) )
    return vector 

def randomSequence(length, dimension):
    sequence = []
    for i in range(length):
        sequence.append( randomVector(dimension) )
    return sequence
        
def main():
    lstm = LSTM(input_dimension=3)
    lstm.runSequence( randomSequence(100, 3) )  
    
    # [[1, -1, 0], [0, .5, .2], [.5, .8, -.4], [.2, .5, .1], [-.4, -.4, -.9], [-.1, -.3, -.1] ])
    
if __name__ == "__main__":
    main()


