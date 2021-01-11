import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

#read csv dataset
data=pd.read_csv('taxi.csv')

#data.head()
#data.shape
#data.isnull().sum()
#dependent & independent variable
x=data.iloc[:,0:-1].values
y=data.iloc[:,-1].values
#print(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

reg=LinearRegression()

reg.fit(x_train,y_train)

#print("Train score : ",reg.score(x_train,y_train))
#print("Test score : ",reg.score(x_test,y_test))

pickle.dump(reg, open('myproject/myapp/taxi.pkl', 'wb'))
model=pickle.load(open('myproject/myapp/taxi.pkl', 'rb'))
print(model.predict([[80, 1770000, 6000, 85]]))
