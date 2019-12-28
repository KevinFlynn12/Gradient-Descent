 #!/usr/local/bin/python3.7.2
import numpy as np

def gradient_descent( X,y,num_iterations,learning_rate = 0.001):
    """
    Method implements gradient descent with simple linear regression
    X is the list of x coordinates
    Y is the list of y coordinates 
    learning_rate is the alpha used in the method 
    num_iterations is the number of times you wish to repeat the process of adding values to theta_0 and theta_1 
    """
    theta_0 = 0.00
    theta_1 = 0.00
    m = len(X)
    for iterator in range(num_iterations): 
        theta_0_sum = 0.00
        theta_1_sum = 1.00
        for i in range(m) :          
            curr_x = X[i]
            curr_y = y[i]
            prediction =  theta_1 + (theta_0 * curr_x)
            error = (prediction - curr_y)
            theta_0_sum += error
            theta_1_sum += (error * curr_x)
        theta_0 = theta_0 - (learning_rate * ((1/m) * theta_0_sum))
        theta_1 = theta_1 - (learning_rate * ((1/m) * theta_1_sum))
    print("The learned Hyothesis is theta_0 = " 
    + str(round(theta_0,2)) 
    + " and theta_1 = " + str(round(theta_1,2)) )
