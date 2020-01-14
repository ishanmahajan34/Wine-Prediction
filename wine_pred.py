# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing Data
x_train = pd.read_csv('hw2-predict-wine-goodness-from-review/trainData.csv', header=None).iloc[:, 1:].to_numpy()
y_train = pd.read_csv('hw2-predict-wine-goodness-from-review/trainLabels.csv', header=None).iloc[:, 1].to_numpy()

x_val = pd.read_csv('hw2-predict-wine-goodness-from-review/valData.csv', header=None).iloc[:, 1:].to_numpy()
y_val = pd.read_csv('hw2-predict-wine-goodness-from-review/valLabels.csv', header=None).iloc[:, 1].to_numpy()

x_test = pd.read_csv('hw2-predict-wine-goodness-from-review/testData_new.csv', header=None).iloc[:, 1:].to_numpy()


# Question 3
# 3.1
# Ridge Regularization function
def ridgeReg(X, Y, lambda_):
    """
    Input
    X : k * N matrix
    y : N * 1 labels
    lambda_ : regularization parameter

    Output
    w : k * 1 weights vector
    b : bias term
    obj : objective function value
    cvErrs : N * 1 vector of LOOCV error for i-th sample
    """
    k, N = X.shape
    X = np.append(X, np.ones((1, N)), axis=0)

    # Don't penalize the bias term
    I = np.identity(k + 1)
    I[-1, -1] = 0

    W_num = np.matmul(X, X.T) + lambda_ * I
    W_den = np.matmul(X, Y)

    W = np.linalg.solve(W_num, W_den)

    hypo = np.matmul(X.T, W)
    diff = hypo - Y

    reg = lambda_ * np.matmul(np.matmul(W.T, I), W)

    # Objective function
    cost = np.sum(np.square(diff)) + reg

    c_inv = np.linalg.inv(W_num)

    # Vectorized code
    W_ndim = np.array([W] * N)
    err_num = np.matmul(W_ndim, X).diagonal() - Y

    ones_ndim = np.ones((N))
    err_den = ones_ndim - np.matmul(np.matmul(X.T, c_inv), X).diagonal()

    cvErrs = np.divide(err_num, err_den)

    # Original code : inefficient because of looping
    # cvErrs = []
    # for i in range(N):
    #     x_i = X[:, i]
    #     y_i = Y[i]
    #
    #     c_x = np.array(np.linalg.lstsq(C, x_i, rcond=-1)[0])
    #
    #     err_num = np.matmul(W.T, x_i) - y_i
    #     err_den = 1 - np.matmul(x_i.T, c_x)
    #
    #     err = err_num / err_den;
    #     cvErrs.append(err)

    return [W[:-1], W[-1], cost, cvErrs]


# Calculate RMSE value of predicted and expected Y values
def rmse(y_, y):
    diff = y_ - y
    return np.sqrt(np.mean(np.square(diff)))


# Question 3.2.1
lambda_values = [0.01, 0.1, 1, 10, 100, 1000, 10000]

rmse_train = []
rmse_val = []
rmse_loocv = []

for i, lambda_ in enumerate(lambda_values):
    print("Iteration ", i)
    w, b, cost, cvErrs = ridgeReg(x_train.T, y_train, lambda_)
    y_ = np.matmul(x_train, w) + b
    print("Training Done")

    rmse_ = rmse(y_, y_train)
    rmse_train.append(rmse_)
    print("RMSE Training: ", rmse_)

    y_v = np.matmul(x_val, w) + b

    rmse_v = rmse(y_v, y_val)
    rmse_val.append(rmse_v)
    print("RMSE Validation: ", rmse_v)

    rmse_loo = np.sqrt(np.mean(np.square(cvErrs)))
    rmse_loocv.append(rmse_loo)
    print("RMSE LOOCV: ", rmse_loo)

    print("Iteration ", i, "Done")
    print()


# Plot the train, validation and leave-one-out-cross-validation RMSE values together on a plot against λ
# plt.figure(figsize=(15, 15))
plt.xlabel('Lambda Value')
plt.ylabel('RMSE')
plt.title('RMSE vs Lambda')

plt.plot(lambda_values, rmse_train, 'r', label="Training", )
plt.plot(lambda_values, rmse_val, 'g', label="Validation")
plt.plot(lambda_values, rmse_loocv, 'b', label="LOOCV")

plt.legend()
plt.show()

# Range of X axis is very large compared to Y axis. Taking log of X axis to make it look smaller.
# plt.figure(figsize=(15, 15))

plt.xlabel('Log 10 Lambda Value')
plt.ylabel('RMSE')
plt.title('Log RMSE vs Lambda')

plt.plot(np.log10(lambda_values), rmse_train, 'r', label="Training", )
plt.plot(np.log10(lambda_values), rmse_val, 'g', label="Validation")
plt.plot(np.log10(lambda_values), rmse_loocv, 'b', label="LOOCV")

plt.legend()
plt.show()


# Question 3.2.2
# From this graph we can clearly see that Log10(Lambda) is minimum at point which is slightly less than 0.
# So lambda value should be close to 1 and it achieves best LOOCV performance.
# Calculating parameters on lambda_ = 0.85

lambda_min = 0.85
w_train_req, b_train_req, cost_train_req, errCV_train_req = ridgeReg(x_train.T, y_train, lambda_min)

y_train_predicted = np.matmul(x_train, w_train_req) + b_train_req
sum_of_square_error = np.sum(np.square(y_train_predicted - y_train))

w_train = np.append(w_train_req, b_train_req)

I = np.identity(w_train.shape[0])
I[-1, -1] = 0

regularization_term = 1 * np.matmul(np.matmul(w_train.T, I), w_train)

print('Training Set :')
print('Objective Value : ', cost_train_req)
print("Sum of Square Error : ", sum_of_square_error)
print('Regularization Term Value :', regularization_term)


# Question 3.2.3
# Input feature vector
file = open("hw2-predict-wine-goodness-from-review/featureTypes.txt", "r")
feature = []
for line in file:
    feature.append(line)

# Finding top 10 most important features and the top 10 least important features.
# Ridge regression reduces weights of features which are not very relevant but it fails to make them zero.
# That is one disadvantage of ridge regression.
max_importance = np.abs(w_train_req).argsort()[-10:][::-1]
min_importance = np.abs(w_train_req).argsort()[:10]

# Most important 10 features:
print("Most important 10 features")
for i in max_importance:
    print(w_train_req[i], " ", feature[i])

# Most least 10 features:
print("Least important 10 features")
for i in min_importance:
    print(w_train_req[i], " ", feature[i])


# Question 3.3.4
# Running Ridge Regression on Test Data

# Adding features to the dataset to improve prediction
# print(x_train.shape)
mean_ = np.mean(x_train, axis=1).reshape(5000, 1)
median_ = np.median(x_train, axis=1).reshape(5000, 1)

x_train = np.concatenate((x_train, mean_), axis=1)
x_train = np.concatenate((x_train, median_), axis=1)
# print(x_train.shape)

# print(x_val.shape)
mean_ = np.mean(x_val, axis=1).reshape(5000, 1)
median_ = np.median(x_val, axis=1).reshape(5000, 1)

x_val = np.concatenate((x_val, mean_), axis=1)
x_val = np.concatenate((x_val, median_), axis=1)
# print(x_val.shape)

# print(x_test.shape)
mean_ = np.mean(x_test, axis=1).reshape(x_test.shape[0], 1)
median_ = np.median(x_test, axis=1).reshape(x_test.shape[0], 1)

x_test = np.concatenate((x_test, mean_), axis=1)
x_test = np.concatenate((x_test, median_), axis=1)
# print(x_test.shape)


# Using training as well as validation dataset for training to get more accuracy
dataset_training = np.concatenate((x_train, x_val))
features_training = np.concatenate((y_train, y_val))

# Calculating y_test value on updated dataset
w_combine, b_combine, cost_combine, errCV_combine = ridgeReg(dataset_training.T, features_training, lambda_min)
y_pred_combined = np.matmul(x_test, w_combine) + b_combine

# Converting to Panda DataFrame
result = {'Id': np.arange(0, y_pred_combined.shape[0]), 'Expected': y_pred_combined}
df = pd.DataFrame(result, columns= ['Id', 'Expected'])

# Export to CSV
export_csv = df.to_csv (r'hw2-predict-wine-goodness-from-review/predTestLabels.csv', index=None, header=True)