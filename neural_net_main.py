import numpy as np 
import matplotlib.pyplot as plt 

#seed for determinism
np.random.seed(1)

#make design array
input_list = [[3,2],[4,1],[6,10],[9,2],[3,7],[7,8],[1,2],[3,1]]
exp_output = [[1,1,0,1,0,0,0,1]]


#check if they are the same length
if(len(input_list) == len(exp_output)):
  print("List are equal")

#transpose lists to make them usuable
X = np.transpose(np.transpose(input_list))
Y = np.transpose(exp_output)
print(X)
print(Y)

#initialize weights and biases
syn0 = 2 * np.random.random((2,5)) - 1
syn1 = 2 * np.random.random((5,1)) - 1

b0 = 2 * np.random.random((1,5)) - 1
b1 = 2 * np.random.random((1,1)) - 1

#create the sigmoid activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))

#create the derivative for the activation function 
def sig_deriv(output):
  return output * (1-output)

#create array to store timesteps and the error values
errors = []
timeSteps = []

#feeding forward
for k in range(500):
  l0 = X #set the firet layer to the input_list
  
  #create the z (temp) variable that is fed into activation function
  z0 = np.dot(l0,syn0) + b0  

  #activate the function
  l1 = sigmoid(z0)

  #second z (temp) var that will be activated
  z1 = np.dot(l1,syn1) + b1 

  #activate the second layer
  l2 = sigmoid(z1)

  #time to find the error
  l2_error = l2 - Y

  #find the derivative of the values in l2 to give the direction in which to change and then multiply it by the error in order to have the direction and magnitude of the shift
  l2_delta = l2_error * sig_deriv(l2)

  #every couple of iterations - print error
  if(k%10 == 0):
    print("ERROR after " + str(k) + " iterations")
    print(str(np.mean(abs(l2_error))))
    
    #store in the arrays to be displayed later
    errors.append(np.mean(abs(l2_error)))
    timeSteps.append(k)

  #find the error of the l1 values
  l1_error = np.dot(l2_delta, np.transpose(syn1))

  #in what direction and magnitude should the values in l1 be shifted
  l1_delta = l1_error * sig_deriv(l1) 

  #change the weights and biases accordingly
  syn0 -= np.dot(np.transpose(l0),l1_delta)
  syn1 -= np.dot(np.transpose(l1),l2_delta)

  #b0 += np.transpose(np.dot(np.transpose(l0),l1_delta))
  b1 += np.mean(abs(np.transpose(np.dot(np.transpose(l1),l2_delta)))) #no idea what I'm doing

#display the errors over time to the screen
def displayError(X, Y, name, x_label, y_label):
  plt.figure(1)
  if(len(X) == len(Y)):
    plt.scatter(X, Y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(name)
  else:
    print("The two arrays are not equal in length. Cannot diplay graph")


#Time for testing of the model

#create function to feed_forward in 
def feed_forward(X_test, Y_test, l0, l1, syn0, syn1, b0, b1):

  #see above in the for loop to understand how this works
  l0 = X_test
  z0 = np.dot(l0,syn0) + b0
  l1 = sigmoid(z0)
  z1 = np.dot(l1,syn1) + b1
  l2 = sigmoid(z1)

  #calculate the error of the function on the new data
  l2_avg_error = np.mean(abs(l2 - Y_test))
  print("ERROR on new data  = " + str(l2_avg_error))

  return l2

#toy data to test 
input_test = [[3,4],[1,3],[2,6],[4,1],[3,2],[5,3]]
exp_out_test = [[0,0,0,1,1,1]]

#transpose to make useful
test = np.transpose(np.transpose(input_test))
exp_res_test = np.transpose(exp_out_test)

displayError(timeSteps, errors, "errors.png", "Timesteps", "Error")
pred = feed_forward(test, exp_res_test, l0, l1, syn0, syn1, b0, b1)
print(pred)
