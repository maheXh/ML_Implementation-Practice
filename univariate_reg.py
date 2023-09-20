import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("cleanedHouseData.csv")
df = df.head(30)
x_train = df["LotArea"]
y_train = df["Price"]
x_train = x_train / 1000
y_train = y_train / 1000
plt.scatter(x_train, y_train)


def gen_f(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb


w = 11.953839202896042
b = 1.1246608943305345
# 11.953839202896042 1.1246608943305345
f_wb = gen_f(x_train, w, b)
plt.plot(x_train, f_wb)
plt.scatter(x_train, y_train)


def predict(x, w, b):
    return x * w + b


f_wb = gen_f(x_train, w, b)
plt.plot(x_train, f_wb)
plt.scatter(x_train, y_train)

x_i = int(input("enter the house area"))
y_hat = predict(x_i, w, b)

plt.scatter(x_i, y_hat, c="r")
plt.axvline(x=x_i, color="blue")
plt.axhline(y=y_hat, color="blue")
print(f"the predicted prce is {y_hat}")


def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb = predict(x[i], w, b)
        cost += (f_wb - y[i]) ** 2
    total_cost = cost * (1 / (2 * m))
    return total_cost


print(f"the cost is {compute_cost(x_train,y_train,w,b)}")


def derivative(x, y, w, b):
    m = x.shape[0]
    sum_w, sum_b = 0, 0
    for i in range(m):
        sum_w += (predict(x[i], w, b) - y[i]) * x[i]
        sum_b += predict(x[i], w, b) - y[i]
    return sum_w * (1 / m), sum_b * (1 / m)


def grad_Des(x, y, w, b, alpha=0.001, num_iter=10000):
    for i in range(num_iter):
        d_dw, d_db = derivative(x, y, w, b)
        temp_w = w - alpha * (d_dw)
        temp_b = b - alpha * (d_db)
        w = temp_w
        b = temp_b
    return w, b


w, b = grad_Des(x_train, y_train, w, b, alpha=0.0000001, num_iter=10000000)
print(w, b)
print(f"the cost is {compute_cost(x_train,y_train,w,b)}")
b = 1.1246608943305345
# 11.953839202896042 1.1246608943305345
f_wb = gen_f(x_train, w, b)
plt.plot(x_train, f_wb)
plt.scatter(x_train, y_train)


def predict(x, w, b):
    return x * w + b


f_wb = gen_f(x_train, w, b)
plt.plot(x_train, f_wb)
plt.scatter(x_train, y_train)

x_i = int(input("enter the house area"))
y_hat = predict(x_i, w, b)

plt.scatter(x_i, y_hat, c="r")
plt.axvline(x=x_i, color="blue")
plt.axhline(y=y_hat, color="blue")
print(f"the predicted prce is {y_hat}")


def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb = predict(x[i], w, b)
        cost += (f_wb - y[i]) ** 2
    total_cost = cost * (1 / (2 * m))
    return total_cost


print(f"the cost is {compute_cost(x_train,y_train,w,b)}")


def derivative(x, y, w, b):
    m = x.shape[0]
    sum_w, sum_b = 0, 0
    for i in range(m):
        sum_w += (predict(x[i], w, b) - y[i]) * x[i]
        sum_b += predict(x[i], w, b) - y[i]
    return sum_w * (1 / m), sum_b * (1 / m)


def grad_Des(x, y, w, b, alpha=0.001, num_iter=10000):
    for i in range(num_iter):
        d_dw, d_db = derivative(x, y, w, b)
        temp_w = w - alpha * (d_dw)
        temp_b = b - alpha * (d_db)
        w = temp_w
        b = temp_b
    return w, b


w, b = grad_Des(x_train, y_train, w, b, alpha=0.0000001, num_iter=10000000)
print(w, b)
print(f"the cost is {compute_cost(x_train,y_train,w,b)}")
