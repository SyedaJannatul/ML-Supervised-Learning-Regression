#Polynomial Regression

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset = pd.read_csv("Position_Salaries.csv")
     #Set matrix of idependent variable
X=dataset.iloc[:,1:2].values
    #Set matrix of dependent variable
Y=dataset.iloc[:,2].values

#Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

#Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg  = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,Y)

#Visualizing Linear Regression model
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg.predict(X),color = 'blue')
plt.title("Truth or Bluff(Linear Regression Model)")
plt.xlabel("Positionlevel")
plt.ylabel("Salary")
plt.show()

"""#Visualizing Polynomial Regression model (for higher resolution and smoother curve)
X_grid = np.arange(min(X),max(X+.1),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,color = 'red')
plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)),color = 'blue')
plt.title("Truth or Bluff(Polynomial Regression Model)")
plt.xlabel("Positionlevel")
plt.ylabel("Salary")
plt.show()"""

#Visualizing Polynomial Regression model
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color = 'blue')
plt.title("Truth or Bluff(Polynomial Regression Model)")
plt.xlabel("Positionlevel")
plt.ylabel("Salary")
plt.show()

#Predicting a new result with Linear Regression
lin_reg.predict([[6.5]])


#Predicting a new result with Polynomial Regression
lin_reg2.predict(poly_reg.fit_transform([[6.5]]))
