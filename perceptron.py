import numpy as np
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_delta(x):
    return math.exp(-x) / ((1+math.exp(-x))**x)

class PerceptronLayer(object):

    # -1 for inputs implies one to one connection of inputs to neurons
    def __init__(self, neurons, inputs):
        self.neurons = neurons
        if inputs is -1 :
            self.inputLayer = True
            self.weights = np.random.rand(neurons)
            self.threshold = np.random.rand(neurons)
        else:
            self.inputLayer = False
            self.weights = np.random.rand(inputs,neurons)
            self.threshold = np.random.rand(neurons)
        self.sigmoid = np.vectorize(sigmoid, otypes=[np.float])
        self.sigmoid_delta = np.vectorize(sigmoid_delta, otypes=[np.float])

    # input is 1 x N
    def forwardProp(self, input):
        self.input = input
        if(self.inputLayer): self.Z = (input*self.weights)+self.threshold
        else: self.Z = np.dot(input,self.weights) + self.threshold
        self.A = self.sigmoid(self.Z)
        return self.A

    def backprop(self, dE):
        delta = dE * self.sigmoid_delta(self.Z)
        if(self.inputLayer):
            deltaW = delta * self.input
        else:
            deltaW = np.array([[element * weight for element in delta] for weight in self.input])
        deltaT = delta
        return (deltaW, deltaT)

class Perceptron(object):

    def __init__(self, height, width, input_n, output_n):
        self.layers = []
        self.layers.append(PerceptronLayer(input_n, -1))
        self.layers.append(PerceptronLayer(height, input_n))
        for i in xrange(1, width):
            self.layers.append(PerceptronLayer(height, height))
        self.layers.append(PerceptronLayer(output_n, height))
        self.learningrate = 0.1

    def set_learning_rate(self, learningrate):
        self.learningrate = learningrate

    def feed_forward(self, input):
        output = input
        inputs = [output]
        for layer in self.layers:
            output = layer.forwardProp(output)
            inputs.append(output)
        return output

    def train_gradients(self, input, gtruth):
        output = self.feed_forward(input)
        dWs = []
        dTs = []
        # get deltaW, and deltaTs
        dE = (gtruth - output) * self.layers[0].sigmoid_delta(self.layers[3].Z)            
        for i in xrange(len(self.layers)-1, -1, -1):
            # print "Layer %d: dE set to %s" % (i, np.array2string(dE))
            if(i+1 < len(self.layers)):
                dE = np.dot(self.layers[i+1].weights, dE)
            else:
                dE = (gtruth - output)
            dE *= self.layers[0].sigmoid_delta(self.layers[i].Z)
            dW, dT = self.layers[i].backprop(dE)
            dWs.insert(0, dW)
            dTs.insert(0, dT)
            # dWs.append(dW)
            # dTs.append(dT)
            # print "Layer %d: dW = %s, dT = %s" % (i, np.array2string(dW), np.array2string(dT))
        return dWs, dTs


    def train(self, inputs, gtruths):
        dW, dT = self.train_gradients(inputs[0], gtruths[0])
        dW = np.array(dW)
        dT = np.array(dT)
        for i in xrange(1, len(inputs)):
            dW2, dT2 = self.train_gradients(inputs[i], gtruths[i])
            dW += dW2
            dT += dT2
        dW /= len(inputs)
        dT /= len(inputs)
        for i in xrange(0, len(self.layers)):
            self.layers[i].weights -= self.learningrate * dW[i]
            self.layers[i].threshold -= self.learningrate * dT[i]
        return dW, dT

# Testing code
input_size = 2
output_size = 1
pp = Perceptron(input_size+1, 2, input_size, output_size)
inputs = []
gtruths = []
inputs.append(np.array([0,0]))
gtruths.append(np.array([0]))
inputs.append(np.array([0,1]))
gtruths.append(np.array([1]))
inputs.append(np.array([1,0]))
gtruths.append(np.array([1]))
inputs.append(np.array([1,1]))
gtruths.append(np.array([1]))

# for i in xrange(0,10):
#     inputs.append(np.random.rand(input_size))
#     gtruths.append(np.random.rand(output_size))

# pp.train(inputs, gtruths)

pp.learningrate = 0.1

def measure_error(pp, inputs, gtruths):
    error = 0
    for i in xrange(0, len(inputs)):
        error_i = abs(gtruths[i] - pp.feed_forward(inputs[i]))
        error += np.sum(error_i)
    return error

for i in xrange(0, 10000):
    for j in xrange(0, len(inputs)):
        pp.train([inputs[j]], [gtruths[j]])
    # pp.train(inputs, gtruths)
    print "%d: %f" % (i, measure_error(pp, inputs, gtruths))


# print pp.layers[3].backProp(dE)

def main():
    # pp = PerceptronLayer(5)
    # print pp.weights
    # print "Threshold"
    # print pp.threshold
    pp = Perceptron(10,2,9,3)

    print "Output: "
    print pp.feedForward(np.random.rand(9), None)
    print "Program running..."

# if __name__ == '__main__':
#     main()