####################################
# Knack Data Analysis
# File Name: Knack_Analysis.py
# Created Date: 2 August 2016
# Author: Niharika Karia
####################################

import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np

knack_data = pd.read_csv('repeat_mm5_allligned.csv')

#iterate over all 56 skills(knacks)
for i in range (1,56):

####################################
#Regression model for various Knacks
####################################
	# Train the model for problem solving using 5 iterations of the training sets: user 1,2,3,4,5,7,8,9
	Xtrain=[[1],[2],[3],[4],[1],[2],[3],[4],[5],[1],[2],[3],[4],[5],[1],[2],[3],[4],[5],[1],[2],[3],[4],[5],[1],[2],[3],[4],[5],[1],[2],[3],[4],[5],[1],[2],[3],[4],[5]]
	Y_train=np.array([knack_data["User1_1"][i],knack_data["User1_2"][i], knack_data["User1_3"][i], knack_data["User1_4"][i],knack_data["User2_1"][i],knack_data["User2_2"][i], knack_data["User2_3"][i], knack_data["User2_4"][i],knack_data["User2_5"][i],knack_data["User3_1"][i], knack_data["User3_2"][i], knack_data["User3_3"][i],knack_data["User3_4"][i],knack_data["User3_5"][i], knack_data["User4_1"][i], knack_data["User4_2"][i],knack_data["User4_3"][i],knack_data["User4_4"][i], knack_data["User4_5"][i], knack_data["User5_1"][i],knack_data["User5_2"][i],knack_data["User5_3"][i], knack_data["User5_4"][i], knack_data["User5_5"][i],knack_data["User7_1"][i],knack_data["User7_2"][i],knack_data["User7_3"][i], knack_data["User7_4"][i], knack_data["User7_5"][i],knack_data["User8_1"][i],knack_data["User8_2"][i], knack_data["User8_3"][i], knack_data["User8_4"][i],knack_data["User8_5"][i],knack_data["User9_1"][i],knack_data["User9_2"][i], knack_data["User9_3"][i],knack_data["User9_4"][i],knack_data["User9_5"][i]])
	
	Ytrain=Y_train
	Xtest=[[1],[2],[3],[4],[5]]
	Ytest=[[0.38],[0.515],[0.65],[0.6],[0.6]]
	# Creating linear regression object
	regr = linear_model.LinearRegression()

	# Training the model for problem solving using the training sets: user 1,2,3,4,5,7,8,9
	regr.fit(Xtrain, Ytrain)

	# The coefficients of regression
	"print('Coefficients: ', regr.coef_)"

	# Plot outputs
	plt.scatter(Xtrain, Ytrain ,  color='black')
	#plt.plot(Xtest, regr.predict(Xtest), color='blue', linewidth=3)
	plt.title(knack_data["email"][i])
	
	#print "\nFor %s" %knack_data["email"][i]
	for degree in [1, 2, 3]:
		model = make_pipeline(PolynomialFeatures(degree), Ridge())
		model.fit(Xtrain, Ytrain)
		plt.plot(Xtest, model.predict(Xtest), label="degree %d" % degree, linewidth=2)
		#Percentage change over iteration
		if (degree==2):
			#print "Percentage change is : %.1f " %( (model.predict(Xtest[4][0])-model.predict(Xtest[0][0]))*100 )		
			# The residual sum of square error
			sres=  np.sum((model.predict(Xtest)- np.mean(Ytest)) ** 2)
			print "For %s residual-sum-of-squares error is : %.2f" % (knack_data["email"][i],sres)
			stot= np.sum((model.predict(Xtest) - Ytest) ** 2)
			#mean square error
			print "For %s root-mean-squares error is : %.2f" % ( knack_data["email"][i], stot)
			#r-squared value
			print "For %s r-squared value is : %.2f" % ( knack_data["email"][i], 1-(sres/stot) )
	plt.ylabel('Kanck_Score')
	plt.xlabel('Iteration')
	# Shrink current axis by 20% to fit the legend
	ax=plt.subplot(111)
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	# Put a legend to the right of the current axis
	ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="regression model")
	#plt.show()
	

####################################
#plot of all users Knack
####################################
	#use loop instead
	plt.plot([1,2,3,4],[knack_data["User1_1"][i],knack_data["User1_2"][i], knack_data["User1_3"][i], knack_data["User1_4"][i]] , color='red', label="User1", linewidth=3)
	plt.plot([1,2,3,4,5],[knack_data["User2_1"][i],knack_data["User2_2"][i], knack_data["User2_3"][i], knack_data["User2_4"][i],knack_data["User2_5"][i]] , color='blue', label="User2",linewidth=3)
	plt.plot([1,2,3,4,5],[knack_data["User3_1"][i], knack_data["User3_2"][i],knack_data["User3_3"][i], knack_data["User3_4"][i], knack_data["User3_5"][i]] , color='green',label="User3", linewidth=3)
	plt.plot([1,2,3,4,5],[knack_data["User4_1"][i],knack_data["User4_2"][i], knack_data["User4_3"][i], knack_data["User4_4"][i], knack_data["User4_5"][i]] , color='yellow',label="User4", linewidth=3)
	plt.plot([1,2,3,4,5],[knack_data["User5_1"][i],knack_data["User5_2"][i], knack_data["User5_3"][i], knack_data["User5_4"][i], knack_data["User5_5"][i]] , color='powderblue',label="User5", linewidth=3)
	plt.plot([1,2,3,4,5],[knack_data["User6_1"][i],knack_data["User6_2"][i], knack_data["User6_3"][i], knack_data["User6_4"][i], knack_data["User6_5"][i]] , color='magenta',label="User6", linewidth=3)
	plt.plot([1,2,3,4,5],[knack_data["User7_1"][i],knack_data["User7_2"][i], knack_data["User7_3"][i], knack_data["User7_4"][i], knack_data["User7_5"][i]] , color='cyan',label="User7", linewidth=3)
	plt.plot([1,2,3,4,5],[knack_data["User8_1"][i],knack_data["User8_2"][i], knack_data["User8_3"][i], knack_data["User8_4"][i], knack_data["User8_5"][i]] , color='orange',label="User8", linewidth=3)
	plt.plot([1,2,3,4,5],[knack_data["User9_1"][i],knack_data["User9_2"][i], knack_data["User9_3"][i], knack_data["User9_4"][i], knack_data["User9_5"][i]] , color='peachpuff',label="User9", linewidth=3)
	plt.ylabel('Kanck_Score')
	plt.xlabel('Iteration')
	plt.title(knack_data["email"][i])
	# Shrink current axis by 20% to fit the legend
	ax=plt.subplot(111)
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	# Put a legend to the right of the current axis
	ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.show()

print "have a good day."
