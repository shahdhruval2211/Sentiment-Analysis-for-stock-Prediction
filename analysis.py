import csv
import numpy as np
from sklearn import linear_model, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

dates = []
prices = []

def get_data(filename):
	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader) 
		for row in csvFileReader:
			dates.append(int(row[0].split('/')[0]))
			prices.append(float(row[1]))
	return

def linreg_predictprices(dates, prices,x):
	dates = np.reshape(dates, (len(dates),1)) 
	prices = np.reshape(prices, (len(prices),1))

	linear_mod = linear_model.LinearRegression() 
	linear_mod.fit(dates, prices) 

	plt.scatter(dates, prices, color= 'black', label= 'Data') 
	plt.plot(dates, linear_mod.predict(dates), color= 'red', label= 'Linear model') 
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title('Linear Regression')
	plt.legend()
	plt.show()
               
	return linear_mod.predict(x)[0][0], linear_mod.coef_[0][0], linear_mod.intercept_[0]

def logreg_predictprices(dates, prices,x):
	dates = np.reshape(dates, (len(dates),1)) 
	prices = np.reshape(prices, (len(prices),1))
	
	logistic_mod = LogisticRegression() 
	logistic_mod.fit(dates, prices) 

	plt.scatter(dates, prices, color= 'black', label= 'Data') 
	plt.plot(dates, logistic_mod.predict(dates), color= 'red', label= 'Logistic Regression model') 
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title('Logistic Regression')
	plt.legend()
	plt.show()
               
	return logistic_mod.predict(x), logistic_mod.coef_[0][0], logistic_mod.intercept_[0]

def nb_predictprices(dates, prices,x):
	dates = np.reshape(dates, (len(dates),1))
	prices = np.reshape(prices, (len(prices),1))

	nb_mod = GaussianNB() 
	nb_mod.fit(dates, prices) 

	plt.scatter(dates, prices, color= 'black', label= 'Data') 
	plt.plot(dates, nb_mod.predict(dates), color= 'red', label= 'Naive Bayes model') 
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title('Naive Bayes')
	plt.legend()
	plt.show()
               
	return nb_mod.predict(x)


def svr_predictprices(dates, prices, x):
	dates = np.reshape(dates,(len(dates), 1)) 

	svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1) 
	svr_lin = SVR(kernel= 'linear', C= 1e3)
	svr_rbf.fit(dates, prices) 
	svr_lin.fit(dates, prices)

	plt.scatter(dates, prices, color= 'black', label= 'Data') 
	plt.plot(dates, svr_rbf.predict(dates), color= 'red', label= 'RBF model') 
	plt.plot(dates,svr_lin.predict(dates), color= 'green', label= 'Linear model') 
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title('Support Vector Regression')
	plt.legend()
	plt.show()

	return svr_rbf.predict(x)[0], svr_lin.predict(x)[0]


get_data('finalData.csv') 
print ("Dates- ", dates)
print ("Prices- ", prices)

print("\nLinear Regression:")
linreg_predicted_price, linreg_coefficient, linreg_constant = linreg_predictprices(dates, prices,[[29]])
print ("\nThe stock open price for 29th Feb is: $", str(linreg_predicted_price))
print ("The regression coefficient is ", str(linreg_coefficient), ", and the constant is ", str(linreg_constant))
print ("the relationship equation between dates and prices is: price = ", str(linreg_coefficient), "* date + ", str(linreg_constant))

int_prices=[]
for y in prices:
        y = int(y)
        int_prices.append(y)

print("\nLogistic Regression:")
logreg_predicted_price, logreg_coefficient, logreg_constant = logreg_predictprices(dates, int_prices,[[29]])
print ("\nThe stock open price for 29th Feb is: $", str(logreg_predicted_price))
print ("The regression coefficient is ", str(logreg_coefficient), ", and the constant is ", str(logreg_constant))
print ("the relationship equation between dates and prices is: price = ", str(logreg_coefficient), "* date + ", str(logreg_constant))


print("\nNaive Bayes:")        
nb_predicted_price = nb_predictprices(dates, int_prices,[[29]])
print ("\nThe stock open price for 29th Feb is: $", str(nb_predicted_price))

print("\nSVR:")
predicted_price = svr_predictprices(dates, prices, [[29]])
print ("\nThe stock open price for 29th Feb is:")
print ("RBF kernel: $", str(predicted_price[0]))
print ("Linear kernel: $", str(predicted_price[1]))

Algorithms = ('Linear Regression','Logistic Regression', 'Naive Bayes', 'SVR_RBF', 'SVR_Linear')
y_pos = np.arange(len(Algorithms))
count = [linreg_predicted_price,logreg_predicted_price,nb_predicted_price,predicted_price[0],predicted_price[1]]
 
plt.bar(y_pos, count, align='center', alpha=0.5)
plt.xticks(y_pos, Algorithms)
plt.ylabel('Predicted Prices')
 
plt.show()
