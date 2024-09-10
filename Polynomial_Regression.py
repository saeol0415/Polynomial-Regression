import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

file_path = 'Linear Regression/sample_data.csv'
data = pd.read_csv(file_path)

X = data['X'].values
Y = data['Y'].values

degree = int(input())

A = np.vander(X, N=degree+1)
B = Y.reshape(-1, 1)

Coef = np.linalg.solve(A.T @ A, A.T @ B)
Y_pred = A @ Coef

SSE = np.sum((Y - Y_pred.flatten()) ** 2)
SST = np.sum((Y - np.mean(Y)) ** 2)
R2score = 1 - SSE / SST

squared_errors = (Y_pred - B) ** 2
total_squared_error = np.sum(squared_errors)
print(f"Total squared error: {total_squared_error:.4f}")
print(f"R2 Score: {R2score:.4f}")

plt.scatter(X, Y, label='Data Points')

cont_X = np.linspace(np.min(X), np.max(X), 500)
cont_Y_pred = np.vander(cont_X, N=degree+1) @ Coef
plt.plot(cont_X, cont_Y_pred, label='Polynomial Regression Line')

for xi, yi, y_pred_i in zip(X, Y, Y_pred):
    plt.vlines(xi, yi, y_pred_i, color='red', linestyle='dotted')

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Sample Data Polynomial Regression")
plt.legend()
plt.show()
