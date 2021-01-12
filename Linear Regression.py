#Linear Regression

import pandas as pd
import numpy as np

cd = pd.read_csv("Automobile price data _Raw_.csv", na_values=('?'))

#cleaning of data
obj = cd.select_dtypes(include='object').copy()       #to find and clean NaN values 
cd = cd.fillna({"num-of-doors":"four"})  
clean_up = {'num-of-doors':{"four":4,"two":2},
            'num-of-cylinders':{'four':4, 'six':6, 'five':5, 'three':3, 'twelve':12, 'two':2, 'eight':8}}
cd = cd.replace(clean_up)

#encoding of object
from sklearn.preprocessing import OrdinalEncoder
ord_en = OrdinalEncoder()
cd["make_en"] = ord_en.fit_transform(cd[["make"]])
cd["fuel_type_en"] = ord_en.fit_transform(cd[["fuel-type"]])
cd["aspiration_en"] = ord_en.fit_transform(cd[["aspiration"]])
cd["body_style_en"] = ord_en.fit_transform(cd[["body-style"]])
cd["drive_wheel_en"] = ord_en.fit_transform(cd[["drive-wheels"]])
cd["engine_location_en"] = ord_en.fit_transform(cd[["engine-location"]])
cd["engine_type_en"] = ord_en.fit_transform(cd[["engine-type"]])
cd["fuel_system_en"] = ord_en.fit_transform(cd[["fuel-system"]])

#a little more arrangements
cd = cd.select_dtypes(exclude=['object']).copy()
cd = cd.fillna(cd.mean())

#training and testing data split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(cd.drop('price',axis=1),cd['price'], test_size=0.2, random_state=65)

#model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
l_reg = LinearRegression()
l_reg.fit(X_train,y_train)

#prediction and accuracy score
l_pred = l_reg.predict( X_test)
accuracy = r2_score(y_test,l_pred)
print(np.round(accuracy,decimals=4))

