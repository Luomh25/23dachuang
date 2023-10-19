import time
import mnist_load as mnist_load
from sklearn.metrics import classification_report  # 生产报告
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

print('reading training and testing data...')
x_train, y_train, x_test, y_test = mnist_load.get_data()
print('data: ', x_train.shape, x_test.shape)
print('label: ', y_train.shape, y_test.shape)

'''
# rf
start_time = time.time()
rf = RandomForestClassifier(n_jobs=-1)
rf.fit(x_train, y_train)
print('training took %fs!' % (time.time() - start_time))
start_time = time.time()
# 根据模型做预测，返回预测结果
pred_rf = rf.predict(x_test)
print('predict took %fs!' % (time.time() - start_time))
report_rf = classification_report(y_test, pred_rf)
confusion_mat_rf = confusion_matrix(y_test, pred_rf)
print(report_rf)
print(confusion_mat_rf)
print('随机森林准确率: %0.4lf' % accuracy_score(pred_rf, y_test))

# LR
start_time = time.time()
lr = LogisticRegression()
lr.fit(x_train, y_train)
print('training took %fs!' % (time.time() - start_time))
start_time = time.time()
# 根据模型做预测，返回预测结果
pred_lr = lr.predict(x_test)
print('predict took %fs!' % (time.time() - start_time))
report_lr = classification_report(y_test, pred_lr)
confusion_mat_lr = confusion_matrix(y_test, pred_lr)
print(report_lr)
print(confusion_mat_lr)
print('LR准确率: %0.4lf' % accuracy_score(pred_lr, y_test))

# 决策树
start_time = time.time()
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
print('training took %fs!' % (time.time() - start_time))
start_time = time.time()
# 根据模型做预测，返回预测结果
pred_dtc = dtc.predict(x_test)
print('predict took %fs!' % (time.time() - start_time))
report_dtc = classification_report(y_test, pred_dtc)
confusion_mat_dtc = confusion_matrix(y_test, pred_dtc)
print(report_dtc)
print(confusion_mat_dtc)
print('决策树准确率: %0.4lf' % accuracy_score(pred_dtc, y_test))
'''

# SVM
param_grid = {'C': [1, 10], 'gamma': [1, 10]}
# param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}
scoring = 'accuracy'
cv = 5

svm = SVC()
'''
start_time1 = time.time()
svm.fit(x_train, y_train)
print('training took %fs!' % (time.time() - start_time1))
start_time2 = time.time()
# 根据模型做预测，返回预测结果
pred_svm = svm.predict(x_test)
report_svm = classification_report(y_test, pred_svm)
print('predict took %fs!' % (time.time() - start_time2))
confusion_mat_svm = confusion_matrix(y_test, pred_svm)
print(report_svm)
print(confusion_mat_svm)
print('SVC准确率: %0.4lf' % accuracy_score(pred_svm, y_test))
'''
start_time3 = time.time()
grid_search = GridSearchCV(svm, param_grid, scoring=scoring, cv=cv)
grid_search.fit(x_train, y_train)
print('search took %fs!' % (time.time() - start_time3))

# 输出最优参数组合和模型性能
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# 在测试集上测试模型性能
best_model = grid_search.best_estimator_
test_score = best_model.score(x_test, y_test)
print("Test score:", test_score)