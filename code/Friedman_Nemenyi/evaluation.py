# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:28:13 2019

@author: liuxin
"""

import numpy as np
import matplotlib.pyplot as plt
from regression import linear_scores, ridge_scores, lasso_scores, elasticNet_scores
import statsmodels.api as sa
import scikit_posthocs as sp
sp.sign_plot(pc, **heatmap_args)
"""
    构造降序排序矩阵
"""
def rank_matrix(matrix):
    cnum = matrix.shape[1]
    rnum = matrix.shape[0]
    ## 升序排序索引
    sorts = np.argsort(matrix)
    for i in range(rnum):
        k = 1
        n = 0
        flag = False
        nsum = 0
        for j in range(cnum):
            n = n+1
            ## 相同排名评分序值
            if j < 3 and matrix[i, sorts[i,j]] == matrix[i, sorts[i,j + 1]]:
                flag = True;
                k = k + 1;
                nsum += j + 1;
            elif (j == 3 or (j < 3 and matrix[i, sorts[i,j]] != matrix[i, sorts[i,j + 1]])) and flag:
                nsum += j + 1
                flag = False;
                for q in range(k):
                    matrix[i,sorts[i,j - k + q + 1]] = nsum / k
                k = 1
                flag = False
                nsum = 0
            else:
                matrix[i, sorts[i,j]] = j + 1
                continue
    return matrix

"""
    Friedman检验
    参数：数据集个数n, 算法种数k, 排序矩阵rank_matrix(k x n)
    函数返回检验结果（对应于排序矩阵列顺序的一维数组）
"""
def friedman(n, k, rank_matrix):
    # 计算每一列的排序和
    sumr = sum(list(map(lambda x: np.mean(x) ** 2, rank_matrix.T)))
    result = 12 * n / (k * ( k + 1)) * (sumr - k * (k + 1) ** 2 / 4)
    result = (n - 1) * result /(n * (k - 1) - result)
    return result

"""
    Nemenyi检验
    参数：数据集个数n, 算法种数k, 排序矩阵rank_matrix(k x n)
    函数返回CD值
"""

def nemenyi(n, k, q):
    return q * (np.sqrt(k * (k + 1) / (6 * n)))
    
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 读取10个数据
X = []
Y = []
X_train = [0 for i in range(10)]
X_test = [0 for i in range(10)]
Y_train = [0 for i in range(10)]
Y_test = [0 for i in range(10)]
sc = StandardScaler()
for i in range(10):
    data = np.loadtxt("Data" + str(i + 1) + ".txt", dtype = np.float32)
    #对训练集数据进行归一化
    sc.fit(data[:,:-1])
    X.append(sc.transform(data[:,:-1]))
    Y.append(data[:,-1])
    #将数据分为训练集和测试集
    X_train[i], X_test[i], Y_train[i], Y_test[i]= train_test_split(X[i], Y[i], test_size = 0.3, random_state = 1)

from sklearn import linear_model
from sklearn.metrics import mean_squared_error

# train regressors
#导入处理过的数据
from load_data import X_train, Y_train, X_test, Y_test

#创建四个数组记录每个模型的得分
linear_scores = [0 for i in range(10)]
ridge_scores = [0 for i in range(10)]
lasso_scores = [0 for i in range(10)]
elasticNet_scores = [0 for i in range(10)]

#评估函数MSE
MSE = mean_squared_error

# 线性回归
model_LR = linear_model.LinearRegression()
for i in range(10):
    model_LR.fit(X_train[i], Y_train[i])
    Y_pred = model_LR.predict(X_test[i])
    linear_scores[i] = MSE(Y_test[i], Y_pred)

# 岭回归
model_RR = linear_model.Ridge()
for i in range(10):
    model_RR.fit(X_train[i], Y_train[i])
    Y_pred = model_RR.predict(X_test[i])
    ridge_scores[i] = MSE(Y_test[i], Y_pred)

# Lasso回归
model_Lasso = linear_model.Lasso()
for i in range(10):
    model_Lasso.fit(X_train[i], Y_train[i])
    Y_pred = model_Lasso.predict(X_test[i])
    lasso_scores[i] = MSE(Y_test[i], Y_pred)

# ElasticNet回归
model_elastic = linear_model.ElasticNet()
for i in range(10):
    model_elastic.fit(X_train[i], Y_train[i])
    Y_pred = model_elastic.predict(X_test[i])
    elasticNet_scores[i] = MSE(Y_test[i], Y_pred)

# evaluation
matrix = np.array([linear_scores, ridge_scores, lasso_scores, elasticNet_scores])
matrix_r = rank_matrix(matrix.T)
Friedman = friedman(10, 4, matrix_r)
CD = nemenyi(10, 4, 2.569)
##画CD图
rank_x = list(map(lambda x: np.mean(x), matrix))
name_y = ["linear_scores", "ridge_scores", "lasso_scores", "elasticNet_scores"]
min_ = [x for x in rank_x - CD/2]
max_ = [x for x in rank_x + CD/2]

print(matrix_r)
plt.title("Friedman")
plt.scatter(rank_x, name_y)
plt.hlines(name_y, min_, max_)
plt.show()
