import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import itertools
import warnings

df = pd.read_csv("C:/Users/hp/Downloads/BERGEPAINT.NS.csv")
df1 = pd.read_csv("C:/Users/hp/Downloads/BERGEPAINT.NS1.csv")
df.head(5)

df=df.dropna()
df1=df1.dropna()

#Initializing Data

train_data, test_data = df[0:int(len(df))], df1[0:int(len(df1))]
training_data = train_data['Close'].values
test_data = test_data['Close'].values
history = [x for x in training_data]
model_predictions = []
N_test_observations = len(test_data)

# Obtaining all Variations of Order

p=d=q=range(0,5)
pdq = list(itertools.product(p,d,q))  

warnings.filterwarnings('ignore')    

# Finding the order variation with minimum aic value 

y=[]
for param in pdq:
	try:
		model_arima1=ARIMA(history,order=param)
		model_arima_fit1=model_arima1.fit()
		y.append(model_arima_fit1.aic)
		a=min(y)
		if(a==model_arima_fit1.aic):
			z=param
		
	except:
		continue


print('The lowest aic value is :',a,'for the parameter ',z)  



#Training and Testing the ARIMA Model  



for time_point in range(N_test_observations):
    model = ARIMA(history, order=z)
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    model_predictions.append(yhat)
    true_test_value = test_data[time_point]
    history.append(true_test_value)

MSE_error = mean_squared_error(test_data, model_predictions)
print('Testing Mean Squared Error is {}'.format(MSE_error))

plt.plot(test_data,color='red',label='Actual Price')
plt.plot(model_predictions,color='blue',linestyle='dashed',label='Predicted Price')
plt.show()    
