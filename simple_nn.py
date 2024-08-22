import numpy as np

np.random.seed(1)
input_size = 3
hidden_size = 3
output_size = 3

# input == RGB value's representing the intensity of sunlight
# output == morning evening night
#Morning: Brighter colors, higher values in the RGB spectrum.
#Evening: Warmer tones, higher values in the red and yellow spectrum.
#Night: Darker colors, lower RGB values.

def sigmoid(x):
	return 1/(1+np.exp(-x))

def sigmoid_der(x):
	return x * (1-x)

w1 = 2* np.random.random((input_size,hidden_size))-1
w2 = 2* np.random.random((hidden_size,output_size))-1

x = np.array([
    [255, 223, 186],  # Morning: Light orange
    [255, 179, 102],  # Morning: Bright yellow-orange
    [102, 51, 0],     # Evening: Dark brown
    [255, 94, 77],    # Evening: Deep orange-red
    [0, 0, 51],       # Night: Dark blue
    [0, 0, 0],        # Night: Black
],dtype=np.float64)


y = np.array([
    [1, 0, 0],  # Morning
    [1, 0, 0],  # Morning
    [0, 1, 0],  # Evening
    [0, 1, 0],  # Evening
    [0, 0, 1],  # Night
    [0, 0, 1],  # Night
],dtype=np.float64)

#x norm 
x = x / 255.0

learning_rate = 0.1

for epoch in range(100):
	##forward propagation
	hidden_input = np.dot(x,w1)
	hidden_output = sigmoid(hidden_input)
	output_layer_input = np.dot(hidden_output,w2)
	output_layer_output = sigmoid(output_layer_input)

	error = y - output_layer_output

	##back propagation
	output_delta = error * sigmoid_der(output_layer_output)
	hidden_error = output_delta.dot(w2.T)
	hidden_delta = hidden_error * sigmoid_der(hidden_output)
	w2 += hidden_output.T.dot(output_delta) * learning_rate
	w1 += x.T.dot(hidden_delta) * learning_rate

test_in = np.array([50,50,150]) /250.0
test_hidden = sigmoid(np.dot(test_in,w1))
test_output = sigmoid(np.dot(test_hidden,w2))

print(test_output)
##sample_ouput [0.31839077 0.37876353 0.43210492]
