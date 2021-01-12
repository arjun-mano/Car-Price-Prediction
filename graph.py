#Graphs 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

cd = pd.read_csv("Automobile price data _Raw_.csv", na_values=('?'))
cd
#cleaning of data
obj = cd.select_dtypes(include='object').copy()       #to find and clean NaN values 
cd = cd.fillna({"num-of-doors":"four"})  
clean_up = {'num-of-doors':{"four":4,"two":2},
            'num-of-cylinders':{'four':4, 'six':6, 'five':5, 'three':3, 'twelve':12, 'two':2, 'eight':8}}
cd = cd.replace(clean_up)

#Graphs and charts
#graph 1
sns.countplot(cd['body-style'], palette='viridis')

#graph 2 
plt1 = pd.DataFrame(cd.groupby(['make'])['price'].mean().sort_values(ascending = False))
plt1.plot.bar()
plt.title('Make vs Average Price')
plt.show()

#Graph 3
sns.jointplot(x='horsepower',y='price',data=cd,kind='scatter',color='b')

#graph 4
sns.catplot(x='fuel-type',kind='count', palette="ch:.25", data=cd)

#graph 5
sns.relplot(x='peak-rpm', y='city-mpg', hue="city-mpg", palette="ch:r=-.5,l=.75", data=cd)

#graph 6
sns.relplot(x='engine-size',y='price', hue="price", size="price",sizes=(10,60), data=cd)

#graph 7
sns.relplot(x='curb-weight',y='engine-size',hue='engine-size',palette="ch:r=-.5,l=.75", size='engine-size',sizes=(30,130), data=cd)

