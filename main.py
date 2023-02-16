import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
inputFolder = '~/Documents/'
df = pd.read_csv(inputFolder + 'Boston.csv')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
print(df.head())
print(df.shape)
print(df.isnull().values.any())
print(df.describe())
df=df.astype(float)
print(df.dtypes)
corrMatrix = df.corr()
print (corrMatrix)
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

sn.heatmap(corrMatrix, annot=True)
plt.show()
x =pd.DataFrame(df.loc[:,["rm","lstat"]])
y=df.loc[:,"medv"]
y=np.array(y).reshape(-1,1)
print(x.shape)
print(y.shape)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
inputFolder = '~/Documents/'
df = pd.read_csv(inputFolder + 'Boston.csv')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
print(df.head())
print(df.shape)
print(df.isnull().values.any())
print(df.describe())
df=df.astype(float)
print(df.dtypes)
X =pd.DataFrame(df.loc[:,["rm","lstat"]])
Y=df.loc[:,"medv"]
Y=np.array(y).reshape(-1,1)
print(X.shape)
print(Y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)
y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
from sklearn.metrics import r2_score
r2 = r2_score(Y_train, y_train_predict)
print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")
y_test_predict = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2 = r2_score(Y_test, y_test_predict)
print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
