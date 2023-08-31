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
*/
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("/content/student_scores.csv")
data.head()
data.tail()
data.isnull().sum()
#storing data to variable
X=data.Hours
Y=data.Scores
print(X)
print(Y)
#defining variables
n=len(X)
m=0
c=0
L=0.01
loss=[]
for i in range(10000):
    ypred = m*X + c
    MSE = (1/n) * sum((ypred - Y)*2)
    dm = (2/n) * sum(X*(ypred-Y))
    dc = (2/n) * sum(ypred-Y)
    c = c-L*dc
    m = m-L*dm
    loss.append(MSE)
print(m,c)
#Plotting Linear Regression Graph
ypred=m*X+c
plt.scatter(X,Y,color="red")
plt.plot(X,ypred,color="green")
plt.xlabel("Study hours")
plt.ylabel("Scores")
plt.title("Study hours vs Scores")
#Graph between iteration Vs loss
plt.show()plt.title("study iteration vs loss")
plt.plot(loss) 
plt.xlabel("iterations") 
plt.ylabel("loss") 
plt.show()
```

## Output:
Training data:

![TRAINING DATA](https://github.com/swethaselvarajm/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119525603/8fff474b-c8c6-4aae-b246-acb21f42990d)

Testing data:

![TESTING DATA](https://github.com/swethaselvarajm/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119525603/aae0ffe5-6ec9-47c5-b20f-9dcbeeeb15ad)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
