# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries in python required for finding Gradient Design.
2. Read the dataset file and check any null value using .isnull() method.
3.Declare the default variables with respective values for linear regression.
4. Calculate the loss using Mean Square Error.
5. Predict the value of y.
6.Plot the graph respect to hours and scores using .scatterplot() method for Linear Regression.
7.Plot the graph respect to loss and iterations using .plot() method for Gradient Descent.
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: SWETHA.S
RegisterNumber: 212222230155

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv('/content/ex1 (1).txt', header = None)
plt.scatter(data[0], data[1])
plt.xticks(np.arange(5, 30, step = 5))
plt.yticks(np.arange(-5, 30, step = 5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")
def computeCost(X,y,theta):
    """
    Test in a numpy array x, y theta and generate the cost function
    in a linear regression model
    """
    m = len(y) # length of the training data
    h=X.dot(theta) #hypothesis
    square_err = (h - y)**2

    return 1/(2*m) * np.sum(square_err) #returning
data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m, 1)), data_n[:, 0].reshape(m, 1), axis = 1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(X,y,theta) # Call the function
def gradientDescent(X,y,theta,alpha,num_iters):
  """
  Take in numpy array X, y and theta and update theta by taking num_oters gradient steps
  with learning rate of alpha

  return theta and the list of the cost of theta during each iteration
  """

  m=len(y)
  J_history = []
    
  for i in range(num_iters):
      predictions = X.dot(theta)
      error = np.dot(X.transpose(),(predictions-y))
      descent = alpha * 1/m * error
      theta-=descent
      J_history.append(computeCost(X,y,theta))

  return theta, J_history
theta, J_history = gradientDescent(X, y, theta, 0.01, 1500)
print("h(x) ="+str(round(theta[0, 0], 2))+" + "+str(round(theta[1, 0], 2))+"x1")
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")
plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value = [y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value, y_value, color = "r")
plt.xticks(np.arange(5, 30, step = 5))
plt.yticks(np.arange(-5, 30, step = 5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")
def predict(x,theta):
   """
    Takes in numpy array of x and theta and return the predicted value of y based on theta
   """

   predictions= np.dot(theta.transpose(),x)

   return predictions[0]
predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1, 0)))
predict2 = predict(np.array([1, 7]), theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2, 0)))
*\
```

## Output:
Profit prediction:

![image](https://github.com/swethaselvarajm/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119525603/82c44aa0-d938-4356-ac6b-8ae9332dd509)

Function output:

![image](https://github.com/swethaselvarajm/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119525603/537bd015-9216-4859-8631-4cd7cdd61731)

Gradient descent:

![image](https://github.com/swethaselvarajm/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119525603/4e18d97b-f355-44f1-8203-d6890030c601)

Cost function using Gradient Descent:

![image](https://github.com/swethaselvarajm/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119525603/96fa3310-fd27-4ef3-ba59-ffeb1fb32f39)

Profit prediction:

![image](https://github.com/swethaselvarajm/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119525603/f422ed04-1c77-4c34-9db9-a5a3f0ee3734)

Profit prediction for a population of 35,000:
![image](https://github.com/swethaselvarajm/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119525603/9dcf44c2-bdc3-4bb9-b2ae-101b37ca20b1)

Profit prediction for a population of 70,000:
![image](https://github.com/swethaselvarajm/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119525603/d70df6c7-e0cf-4e83-a322-1b3eaecdfd1b)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
