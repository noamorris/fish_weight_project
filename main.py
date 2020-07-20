import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#splitting the data

feature = pd.read_csv('https://raw.githubusercontent.com/BolunDai0216/nyuMLSummerSchool/master/day05/fish_market_feature.csv')
label = pd.read_csv('https://raw.githubusercontent.com/BolunDai0216/nyuMLSummerSchool/master/day05/fish_market_label.csv')
X = feature.values
y = label.values
X_test = pd.read_csv('https://raw.githubusercontent.com/BolunDai0216/nyuMLSummerSchool/master/day05/fish_market_test_feature.csv').values
y_test = pd.read_csv('https://raw.githubusercontent.com/BolunDai0216/nyuMLSummerSchool/master/day05/fish_market_test_label.csv').values

from sklearn.model_selection import train_test_split
X_train, X_val, ytrain, yval = train_test_split(X, y, test_size=0.33, random_state=42)

plt.figure()
plt.scatter(X[:, 0], y, label='length1')
plt.scatter(X[:, 1], y, label='length2')
plt.scatter(X[:, 2], y, label='length3')
plt.scatter(X[:, 3], y, label='height')
plt.scatter(X[:, 4], y, label='width')
plt.legend()
plt.grid()
plt.xlabel("X")
plt.ylabel("weight")
plt.show()
print(X.shape)
print(y.shape)
plt.figure()
plt.scatter(X_test[:, 0], y_test, label='length1')
plt.scatter(X_test[:, 1], y_test, label='length2')
plt.scatter(X_test[:, 2], y_test, label='length3')
plt.scatter(X_test[:, 3], y_test, label='height')
plt.scatter(X_test[:, 4], y_test, label='width')
plt.legend()
plt.grid()
plt.xlabel("X")
plt.ylabel("weight")
plt.show()
print(X_test.shape)
print(y_test.shape)

#creating a design matrix

from sklearn.preprocessing import PolynomialFeatures
#training
poly = PolynomialFeatures(4)
Xtrain = poly.fit_transform(X_train)
#validation
Xval = poly.fit_transform(X_val)
#testing
Xtest = poly.fit_transform(X_test)
print(poly)

#predicting
from sklearn import linear_model

reg = linear_model.LinearRegression(fit_intercept=False)
reg.fit(Xtrain, ytrain)
w = reg.coef_

yhat = reg.predict(Xtrain)
yhatv = reg.predict(Xval)
yhatt = reg.predict(Xtest)

#plotting before regularization

reg = linear_model.LinearRegression(fit_intercept=False)
reg.fit(Xtrain, ytrain)
w = reg.coef_
#plotting
plt.figure()
plt.plot(ytrain,'o',markeredgecolor='black',label='train')
plt.plot(yhat,'o',markeredgecolor='blue',label='prediction')
plt.legend()
plt.grid()
plt.show()
#validation
plt.figure()
plt.plot(yval,'o',label='validation',markeredgecolor='black')
plt.plot(yhatv,'o',label='validation predicted',markeredgecolor='black')
plt.legend()
plt.grid()
plt.show()
#testing
plt.figure()
plt.plot(y_test,'o',label='testing',markeredgecolor='black',color='lime')
plt.plot(yhatt,'o',label='testing prediction',markeredgecolor='black', color='green')
plt.legend()
plt.grid()
plt.show()
print("w = ")
with np.printoptions(precision=2, suppress=True):
    print('the first 10 values of w \n', w.reshape(-1,1)[0:9])  

#plotting after regularization
reg = linear_model.Ridge(alpha=1.0,fit_intercept=False)
reg.fit(Xtrain,ytrain)
w = reg.coef_
yhat = reg.predict(Xtrain)
#plotting
plt.figure()
plt.plot(ytrain,'o',markeredgecolor='black',label='train')
plt.plot(yhat,'o',markeredgecolor='blue',label='prediction')
plt.legend()
plt.grid()
plt.show()
#validation
plt.figure()
yhatv = reg.predict(Xval)
plt.plot(yval,'o',label='validation',markeredgecolor='black')
plt.plot(yhatv,'o',label='validation predicted',markeredgecolor='black')
plt.legend()
plt.grid()
plt.show()
#testing
plt.figure()
plt.plot(y_test,'o',label='testing',markeredgecolor='black',color='lime')
plt.plot(yhatt,'o',label='testing prediction',markeredgecolor='black', color='green')
plt.legend()
plt.grid()
plt.show()
print("w = ")
with np.printoptions(precision=2, suppress=True):
    print('the first 10 values of w \n', w.reshape(-1,1)[0:9]) 
    
#rmse

msetrain = np.mean((ytrain-yhat)**2)
print('rmse of training data=',np.sqrt(msetrain))

mseval = np.mean((yval-yhatv)**2)
print('rmse of validation data=',np.sqrt(mseval))

msetest = np.mean((y_test-yhatt)**2)
print('rmse of testing data=',np.sqrt(msetest))

#scores

trainscore = reg.score(Xtrain,ytrain)
valscore = reg.score(Xval,yval)
testscore = reg.score(Xtest,y_test)

print('training data score=',trainscore)
print('validation data score=',valscore)
print('testing data score=',testscore)
