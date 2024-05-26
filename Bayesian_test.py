# Python program to perform Bayesian Regression  
  
# Importing modules that are required  
from sklearn.datasets import fetch_california_housing  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import r2_score  
from sklearn.linear_model import BayesianRidge  
     
# Loading the Boston dataset  
X, Y = fetch_california_housing(return_X_y = True)  
     
# Splitting the training and testing datasets  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 16)  
     
# Creating an instance of the model and training it  
br = BayesianRidge()  
br.fit(X_train, Y_train)  
     
# Predicting values for the unseen data, i.e., the testing data  
Y_pred = br.predict(X_test)  
     
# Computing the R-square score for the model  
print(f"The r2 score of the model is: {r2_score(Y_test, Y_pred)}")  