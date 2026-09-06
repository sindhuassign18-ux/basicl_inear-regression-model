import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


s1 =pd.read_csv(r'C:\\Users\\DELL\Desktop\\student data.csv')
#print(s1.head())

y= s1[['sepal_L']]
x=s1[['petal_L']]

x_train, x_test, y_train, y_test=train_test_split(x,y, test_size= 0.3)



lr = LinearRegression()
lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)

plt.scatter(x_test, y_test, label='Actual')
plt.plot(x_test, y_pred, label='Predicted')

plt.xlabel('Petal Length')
plt.ylabel('Sepal Length')
plt.title('Linear Regression: Petal Length vs Sepal Length')

plt.legend()
plt.show() 

print(y_test.head())
print( y_pred[0:5])

print(mean_squared_error(y_test, y_pred))