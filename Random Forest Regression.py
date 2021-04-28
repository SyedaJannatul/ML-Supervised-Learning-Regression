#Random Forest Regression

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

"""
#Categorical variable,Dummy variable & One hot encoding
    #Categorical variable
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
    #Dummy variable & One hot encoding
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy variable Trap
X = X[:,1:]
      
#Spliting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
"""

"""#Feature Scalimg
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y= sc_Y.fit_transform(np.reshape(Y,(10,1)))
"""

"""
#Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)
"""

#Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10,random_state = 0)
regressor.fit(X,Y)
"""
#Visualizing Linear Regression model
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg.predict(X),color = 'blue')
plt.title("Truth or Bluff(Linear Regression Model)")
plt.xlabel("Positionlevel")
plt.ylabel("Salary")
plt.show()
"""

#Visualizing Random Forest Regression model
plt.scatter(X,Y,color = 'red')
plt.plot(X,regressor.predict(X),color = 'blue')
plt.title("Truth or Bluff(Random Forest Regression Model)")
plt.xlabel("Positionlevel")
plt.ylabel("Salary")
plt.show()

#Visualizing Random Forest Regression model(for higher resolution and smoother curve)
X_grid = np.arange(min(X),max(X),.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,color = 'red')
plt.plot(X_grid,regressor.predict(X_grid),color = 'blue')
plt.title("Truth or Bluff(Random Forest Regression Model)")
plt.xlabel("Positionlevel")
plt.ylabel("Salary")
plt.show()

"""
#Predicting a new result with Linear Regression
lin_reg.predict([[6.5]])
"""

#Predicting a new result with Random Forest Regression
Y_pred = regressor.predict([[6.5]])
