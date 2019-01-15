
'''
      Author: Christelle Lusso 
      Date: 03 Novembre 2018

      
      House prediction with polynomial regression models.

      1) Rent prediction by 
         - linear regression (surface) 
         - quadratic regression (surface)
         - linear regression (surface, borough)
         - quadratic regression (surface, borough)

      2) The code performs the comparison between each models and tells which is the best.

      3) The first part performs the calculations without splitting, on the whole data set.
      These results are considered as reference values (taking advantage of small dimension).
      Note: in the linear case we calculate the exact solution by the formula
            of the normal equation to compare it with the sklearn result.
 
      4) The second part performs a splitting. For the selection of the model with splitting, 
         a series of splitting (nbsplit) is carried out then we average the errors.
         This method is not optimal since I've coded it before knowing the cross-validation.


      *) For the error the code computes the relative errors between predictions and data:
           err = ||hat(y)-y||_2/||y||_2.

      **) For displaying figures, assign the plot boolean to True.

'''
import sys 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set()

# Plot boolean for printing figures 
plot = False
# Polynomial regression degree
deg = 2



#==================================
#
#       References values
#
#==================================
print('\n***********************')
print("* References values   *")
print("* Relatives errors :  *")
print('***********************')
#-----------------------------
#    LINEAR regression
#-----------------------------
house_data = pd.read_csv('house_data.csv')
#house_data.info()
#X = np.matrix([np.ones(house_data.shape[0]), house_data['surface'].as_matrix()]).T
X = np.matrix([np.ones(house_data.shape[0]), house_data['surface'].values]).T
y = np.matrix(house_data['price']).T
plt.xlabel('Surface'); plt.ylabel('Price')
plt.plot(X[:,1], y, 'ro', markersize=4)
# Exact solution (by normal equation): comparison with sklearn 
theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y) 
plt.plot(X[:,1],theta.item(0) + X[:,1] * theta.item(1))
diff = theta.item(0) + theta.item(1) * X[:,1] - y[:]
errL2 = np.sqrt(np.sum(np.multiply(diff,diff)))
errL2_verif = LA.norm(diff,2)

#---------- SKLEARN -----------
regr = linear_model.LinearRegression()
yy = y.A.squeeze()
regr.fit(X, yy)
# linear prediction for each surface 
y_predict_lin = regr.predict(X)
diff_lin = y_predict_lin - yy
errL2_lin = LA.norm(diff_lin,2)/LA.norm(y,2)
print("Linear regression:    ", errL2_lin, "( Exact solution: ", errL2/LA.norm(y,2),")")

#-------------------------------
#    QUADRATIC regression 
#-------------------------------
poly = PolynomialFeatures(degree=deg)
predict_coeff = poly.fit_transform(X[:,1])
regr_quad = linear_model.LinearRegression()
regr_quad.fit(predict_coeff,yy)

xx = np.linspace(0, X[:,1].max(), X[:,1].shape[0]) 
x_quad = poly.transform(xx.reshape(xx.shape[0], 1)) 
predict_quad = regr_quad.predict(x_quad)
plt.plot(xx, predict_quad)  

X_poly = poly.transform(X[:,1])
# quadratic prediction for each surface 
y_predict_quad = regr_quad.predict(X_poly)
diff_quad = y_predict_quad - yy
errL2_quad = LA.norm(diff_quad,2)/LA.norm(y,2)
print("Quadratic regression: ", errL2_quad)
plt.title('Polynomial regression without borough')
if plot==1: plt.show()



#**********************
#
#    With borough
#
#**********************
print('-------------------------------')
print("-  With borough (3 features)  -")
print('-------------------------------')
#-----------------------------
#    LINEAR regression
#-----------------------------
X3 = np.matrix([np.ones(house_data.shape[0]), house_data['surface'].values, house_data['arrondissement'].values]).T
theta3 = np.linalg.inv(X3.T.dot(X3)).dot(X3.T).dot(y) 

fig = plt.figure()
fig.suptitle('Lineaire regression with borough')
ax = fig.add_subplot(1,2,1, projection='3d')
x1 = X3[:,1].A.squeeze(); x2 = X3[:,2].A.squeeze(); x3 = y.A.squeeze()
ax.scatter(x1,x2,x3,c='r',marker='^')
ax.set_xlabel('Surface'); ax.set_ylabel('Borough'); ax.set_zlabel('Rent')
ax = fig.add_subplot(1,2,2, projection='3d')
predict3 = theta3.item(0) + theta3.item(1) * X3[:,1] + theta3.item(2) * X3[:,2]
ax.plot_trisurf(x1, x2, predict3.A.squeeze())
ax.set_xlabel('Surface'); ax.set_ylabel('Borough'); ax.set_zlabel('Rent')

diff3 = theta3.item(0) + theta3.item(1) * X3[:,1] + theta3.item(2) * X3[:,2] - y[:]
errL2 = np.sqrt(np.sum(np.multiply(diff3,diff3)))
errL2_verif = LA.norm(diff3,2)

#---------- SKLEARN -----------
regr3 = linear_model.LinearRegression()
regr3.fit(X3, yy)
# linear prediction for each (surface, borough) data
y_predict_lin3 = regr3.predict(X3) 
diff_lin3 = y_predict_lin3 - yy
errL2_lin3 = LA.norm(diff_lin3,2)/LA.norm(y,2)
print("Linear regression:    ", errL2_lin3, "( Exact solution: ", errL2/LA.norm(y,2), ")")
if plot==1: plt.show()


#-------------------------------
#    QUADRATIC regression
#-------------------------------
poly3 = PolynomialFeatures(degree=deg)
X3_ = np.matrix([X3[:,1].A.squeeze(),X3[:,2].A.squeeze()]).T
predict_coeff3 = poly3.fit_transform(X3_)
regr_quad3 = linear_model.LinearRegression()
regr_quad3.fit(predict_coeff3,yy)

X3_poly = poly3.transform(X3_)
# quadratic prediction for each features (surface,borough) 
y_predict_quad3 = regr_quad3.predict(X3_poly)
diff_quad3 = y_predict_quad3 - yy
errL2_quad3 = LA.norm(diff_quad3,2)/LA.norm(y,2)
print("Quadratic regression: ", errL2_quad3)

fig = plt.figure()
fig.suptitle('Quadratic regression with borough')
ax = fig.add_subplot(1,2,1, projection='3d')
x1 = X3[:,1].A.squeeze(); x2 = X3[:,2].A.squeeze(); x3 = y.A.squeeze()
ax.scatter(x1,x2,x3,c='r',marker='^')
ax.set_xlabel('Surface'); ax.set_ylabel('Borough'); ax.set_zlabel('Rent')
ax = fig.add_subplot(1,2,2, projection='3d')
ax.plot_trisurf(x1, x2, y_predict_quad3)
ax.set_xlabel('Surface'); ax.set_ylabel('Borough'); ax.set_zlabel('Rent')
if plot==1: plt.show()

mod = {
   errL2_lin: 'linear model with 2 features', 
   errL2_quad: 'quadratic model with 2 features',
   errL2_lin3: 'linear model with 3 features', 
   errL2_quad3: 'quadratic model with 3 features', 
}
best_mod = min(errL2_lin,errL2_quad,errL2_lin3,errL2_quad3)
print("\nWithout splitting the most performing model is the", mod[best_mod])




'''
   The solution depends on the splitting so
   we carry out a number (nbsplit) of splitting
   then we average the errors to select the final model.
   We chose to leave the fixed splitting size at 80%.

   Note: it is not optimal since it have been coded before knowing the cross-validation.

'''
#==================================
#
#        With Splitting
#
#==================================
nbsplit = 100
print('\n\n***************************************************')
print("* With splitting                                  *")
print("* Relatives means errors for", nbsplit, "splitting:")
print('***************************************************')
etot1 = 0.; etot2 = 0.; etot3 = 0.; etot4 = 0.
train_s = 0.8
test_s=1-train_s
for i in range(nbsplit):

   xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=train_s, test_size=test_s)
   
   #------------- linear regression ---------------
   regr_split = linear_model.LinearRegression()
   yytrain = ytrain.A.squeeze()
   regr_split.fit(xtrain, yytrain)
   y_lin_split = regr_split.predict(xtest) 
   yytest = ytest.A.squeeze()
   diff_lin_split = y_lin_split - yytest
   errL2_lin_split = LA.norm(diff_lin_split,2)
   err1 = errL2_lin_split/LA.norm(ytest,2)  
   etot1 = etot1+err1
   
   #----------- quadratic regression  -------------
   poly_split = PolynomialFeatures(degree=deg)
   predict_coeff_split = poly_split.fit_transform(xtrain[:,1])
   regr_quad_split = linear_model.LinearRegression()
   regr_quad_split.fit(predict_coeff_split,yytrain)
   
   xx_split = np.linspace(0, xtrain[:,1].max(), xtrain[:,1].shape[0]) 
   x_quad_split = poly_split.transform(xx_split.reshape(xx_split.shape[0], 1)) 
   predict_quad_split = regr_quad_split.predict(x_quad_split)
   
   X_poly_split = poly_split.transform(xtest[:,1])
   y_quad_split = regr_quad_split.predict(X_poly_split)
   diff_quad_split = y_quad_split - yytest
   errL2_quad_split = LA.norm(diff_quad_split,2)
   err2 = errL2_quad_split/LA.norm(ytest,2)
   etot2 = etot2+err2   
   
   #-----------------------
   # With borough
   #-----------------------
   xtrain3, xtest3, ytrain3, ytest3 = train_test_split(X3, y, train_size=train_s, test_size=test_s)

   #------------- linear regression   ---------------
   regr3_split = linear_model.LinearRegression()
   yytrain3 = ytrain3.A.squeeze()
   regr3_split.fit(xtrain3, yytrain3)
   y_lin3_split = regr3_split.predict(xtest3) 
   yytest3 = ytest3.A.squeeze()
   diff_lin3_split = y_lin3_split - yytest3
   errL2_lin3_split = LA.norm(diff_lin3_split,2)
   err3 = errL2_lin3_split/LA.norm(ytest3,2)
   etot3 = etot3+err3 
   
   #----------- quadratic regression   -------------
   poly3_split = PolynomialFeatures(degree=deg)
   X3_train_ = np.matrix([xtrain3[:,1].A.squeeze(),xtrain3[:,2].A.squeeze()]).T
   predict_coeff3_split = poly3_split.fit_transform(X3_train_)
   regr_quad3_split = linear_model.LinearRegression()
   regr_quad3_split.fit(predict_coeff3_split,yytrain3)
   
   X3_test_ = np.matrix([xtest3[:,1].A.squeeze(),xtest3[:,2].A.squeeze()]).T
   X3_poly_split = poly3_split.transform(X3_test_)
   y_quad3_split = regr_quad3_split.predict(X3_poly_split)
   diff_quad3_split = y_quad3_split - yytest3
   errL2_quad3_split = LA.norm(diff_quad3_split,2)
   err4 = errL2_quad3_split/LA.norm(ytest3,2)
   etot4 = etot4+err4 
   

print("Linear regression:     ", etot1/nbsplit)  
print("Quadractic regression: ", etot2/nbsplit) 
print('-------------------------------')
print("-  With borough (3 features)  -")
print('-------------------------------')
print("Linear regression:     ", etot3/nbsplit) 
print("Quadractic regression: ", etot4/nbsplit) 

mod = {
   etot1: 'linear model with 2 features', 
   etot2: 'quadratic model with 2 features',
   etot3: 'linear model with 3 features', 
   etot4: 'quadratic model with 3 features', 
}
best = min(etot1,etot2,etot3,etot4)
print("\nWith splitting the most performing model is the", mod[best])





