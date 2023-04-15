import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn import linear_model
# Lấy dữ liệu
df = pd.read_csv("FuelConsumptionCo2.csv")

data = df[['FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB_MPG', 'CO2EMISSIONS']]\
        .rename(columns={'FUELCONSUMPTION_HWY': 'HWY',
                 'FUELCONSUMPTION_COMB_MPG': 'COMB_MPG'})
# print(data.head())

#tính weight, bias
length = pd.Series.to_numpy(data.CO2EMISSIONS).shape[0]
x1 = pd.Series.to_numpy(data.COMB_MPG)
x2 = pd.Series.to_numpy(data.CO2EMISSIONS)
Xbar = np.concatenate((x1, x2, np.ones(length))).reshape((3, length)).T

A = Xbar.T @ Xbar
b = Xbar.T @ pd.Series.to_numpy(data.HWY).T
w = np.linalg.pinv(A) @ b
weight0, weight1, bias = w
print(w)

def predict(X):
    global w, bias
    return X @ w[0 : 2] + bias

def compute_cost(Y, Y_):
    return np.mean((Y - Y_)**2)


#sinh dữ liệu test
msk = np.random.rand(len(df)) < 0.8
train = data[msk]
test = data[~msk]


test_x = np.asanyarray(test[['COMB_MPG', "CO2EMISSIONS"]])
test_y = np.asanyarray(test[['HWY']])
test_y_ = predict(test_x)
# print(test_y_)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_))
print("Cost: %.2f" % compute_cost(test_y , test_y_) )

#trực quan hoá dữ liệu
xy_plt = np.concatenate([np.linspace(0, 50, 100)[:, None], np.linspace(0, 500, 100)[:, None]], axis=1)
X, Y = np.meshgrid(xy_plt[:, 0], xy_plt[:, 1])
# print(X, Y)
zs = np.array([x * weight0 + y * weight1 + bias for x, y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

ax = plt.axes(projection='3d')
ax.view_init(10, 3) # change view to see more

ax.scatter3D(test_x[:, 0], test_x[:, 1], test_y, 'blue')
ax.set_xlabel('COMB_MPG')
ax.set_ylabel("CO2EMISSIONS")
ax.set_zlabel('HYW')
ax.plot_surface(X, Y, Z, color='r', alpha=0.5)

plt.show()
