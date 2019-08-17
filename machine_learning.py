import pandas as pd
import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt
from mlxtend.plotting import category_scatter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier


test_demographics = pd.read_csv('testdemographics.csv')
test_perf         = pd.read_csv('testperf.csv')


train_demographics= pd.read_csv('traindemographics.csv')
train_perf        = pd.read_csv('trainperf.csv')

train_prev_loans  = pd.read_csv('trainprevloans/trainprevloans.csv')
test_prev_loans   = pd.read_csv('testprevloans/testprevloans.csv')


#Display this information
print test_demographics.head(3),'<<<<<<<<test_demogr>>>>>>>>>>>>'
print test_perf.head(3),'<<<<<<<<<<<test_perf>>>>>>>>>>>>'
print train_demographics.head(3),'<<<<<<<<<<<<<<<<train_dem>>>>>>>>>>>>>'
print train_perf.head(3),'<<<<<<<<<<<<<train_perf>>>>>>>>>>>>>>>>'
print train_prev_loans.head(3),'<<<<<<<<<<<train_prev_loans>>>>>>>>>'
print test_prev_loans.head(3),'<<<<<<<<<<<test_prev_loans>>>>>>>>>>>>'


#Focus on train_prev_loans
# print train_prev_loans['customerid'].shape


#Check if all customer id are unique


# # Map from customerid to ordinary number
# unique_customer_id = set(train_prev_loans['customerid'])

# customers = {}

# for i, cusid in tqdm(enumerate(unique_customer_id)):
#     customers[str(cusid)] = i
# # print customers

# closedate = pd.to_datetime(train_prev_loans['closeddate'])
# firstduedate= pd.to_datetime(train_prev_loans['firstduedate'])
# time_difference= closedate - firstduedate
# train_prev_loans['time_difference'] = pd.Series(time_difference)


# # replace customerid with numbers
# for i, cusid in tqdm(enumerate(train_prev_loans['customerid'])):
#     train_prev_loans['customerid'][i] = customers[str(cusid)]
    
# plt.scatter(train_prev_loans['customerid'], train_prev_loans['time_difference'])
# plt.show()


# plt.scatter(train_perf['loanamount'], train_perf['termdays'])
# plt.scatter(train_perf['loanamount'], train_perf['totaldue'])

# for i, flag in enumerate(train_perf['good_bad_flag']):
#     train_perf['good_bad_flag'][i] = 1 if i == 'Good' else 0

# train_perf.plot(x='termdays', y=['loanamount','totaldue'], kind='hist')
# plt.show()






train_demographics['employment_status_clients'][np.where(train_demographics['employment_status_clients'] == 'Permanent')[0]] = 1
train_demographics['employment_status_clients'][np.where(train_demographics['employment_status_clients'] == 'Self-Employed')[0]] = 2
train_demographics['employment_status_clients'][np.where(train_demographics['employment_status_clients'] == 'Student')[0]] = 3
train_demographics['employment_status_clients'][np.where(train_demographics['employment_status_clients'] == 'Unemployed')[0]] = 4

train_firstduedate = pd.to_datetime(train_prev_loans['firstduedate'])
train_firstrepaiddate = pd.to_datetime(train_prev_loans['firstrepaiddate'])
train_date_diff = (train_firstduedate - train_firstrepaiddate)/np.timedelta64(1,'D')



df = pd.DataFrame()
df['termdays']=train_perf['termdays']
df['totaldue']=train_perf['totaldue']
df['loanamount']=train_perf['loanamount']
df['daysdiff'] = pd.Series(train_date_diff)
# df['employment_status_clients'] = train_demographics['employment_status_clients']

df['good_bad_flag2']=train_perf['good_bad_flag2']



test_demographics['employment_status_clients'][np.where(test_demographics['employment_status_clients'] == 'Permanent')[0]] = 1
test_demographics['employment_status_clients'][np.where(test_demographics['employment_status_clients'] == 'Self-Employed')[0]] = 2
test_demographics['employment_status_clients'][np.where(test_demographics['employment_status_clients'] == 'Student')[0]] = 3
test_demographics['employment_status_clients'][np.where(test_demographics['employment_status_clients'] == 'Unemployed')[0]] = 4

test_firstduedate = pd.to_datetime(test_prev_loans['firstduedate'])
test_firstrepaiddate = pd.to_datetime(test_prev_loans['firstrepaiddate'])
test_date_diff = (test_firstduedate - test_firstrepaiddate)/np.timedelta64(1,'D')

df_test = pd.DataFrame()
df_test['termdays']   =test_prev_loans['termdays']
df_test['totaldue']   =test_prev_loans['totaldue']
df_test['loanamount'] =test_prev_loans['loanamount']
df_test['daysdiff']   = pd.Series(test_date_diff)
# df_test['employment_status_clients'] = test_demographics['employment_status_clients']


# fig = category_scatter(x='loanamount', y='totaldue', label_col='good_bad_flag2', 
#                        data=df, legend_loc='upper left')


# plt.xlabel('loanamount')
# plt.ylabel('totaldue')
# plt.show()


# df.plot()
# plt.show()
#---------------------------------------------------------------------



X = df.drop('good_bad_flag2',1)
y = list(df.good_bad_flag2)
X=pd.get_dummies(X)
df=pd.get_dummies(df)
df_test=pd.get_dummies(df_test)
x_train,x_cv,y_train,y_cv = train_test_split(X,y,test_size=0.1)
model=LogisticRegression()
model.fit(x_train,y_train)
pred_cv=model.predict(x_cv)
print '<<<<<<<<<<<LOGISTIC REGRESSION ALGORITHM>>>>>>>>>>>>>>>>>>>'
print accuracy_score(y_cv,pred_cv) #78.86%
matrix=confusion_matrix(y_cv,pred_cv)
print(matrix)
print '<<<<<<<<<<<LOGISTIC REGRESSION ALGORITHM DONE!!!!>>>>>>>>>>>>>>>>>>>'
dt=tree.DecisionTreeClassifier(criterion='gini')
dt.fit(x_train,y_train)
pred_cv1=dt.predict(x_cv)
print '<<<<<<<<<<<<DECISION TREE ALGORITHM>>>>>>>>>>>>>>>>>>'
print accuracy_score(y_cv,pred_cv1) #71.54%
matrix1=confusion_matrix(y_cv,pred_cv1)
print(matrix1)
print '<<<<<<<<<<<<<DECISION TREE ALGORITHM DONE!!!>>>>>>>>>>>>>>>>>>'
rf=RandomForestClassifier()
rf.fit(x_train,y_train)
pred_cv2=rf.predict(x_cv)
print '<<<<<<<<<<<<<<<<<<,RANDOM FOREST ALGORITHM>>>>>>>>>>>>>>'
print accuracy_score(y_cv,pred_cv2) #77.23%
matrix2=confusion_matrix(y_cv,pred_cv2)
print(matrix2)
print '<<<<<<<<<<<<<<<<<<<<<RANDOM FOREST ALGORITHM DONE!!!!>>>>>>>>>>>>>>>>>>'
svm_model=svm.SVC()
svm_model.fit(x_train,y_train)
pred_cv3=svm_model.predict(x_cv)
print '<<<<<<<<<<<<<<<<<<<<<<UPPORT VECTOR MACHINE (SVM) ALGORITHM>>>>>>>>>>>>>>>>>>>>>'
print accuracy_score(y_cv,pred_cv3) #64.23%
matrix3=confusion_matrix(y_cv,pred_cv3)
print(matrix3)
print '<<<<<<<<<<<<<<<<<<<<<<<SUPPORT VECTOR MACHINE (SVM) ALGORITHM DONE!!!>>>>>>>>>>>>>>>>>>>>>>'
nb=GaussianNB()
nb.fit(x_train,y_train)
pred_cv4=nb.predict(x_cv)
print '<<<<<<<<<<<<<<<<<<NAIVE BAYES ALGORITHM>>>>>>>>>>>>>>>>>>>>>'
print accuracy_score(y_cv,pred_cv4) #80.49%
matrix4=confusion_matrix(y_cv,pred_cv4)
print(matrix4)
print '<<<<<<<<<<<<<<<<<<NAIVE BAYES ALGORITHM DONE!!!>>>>>>>>>>>>>>>'
kNN=KNeighborsClassifier()
kNN.fit(x_train,y_train)
pred_cv5=kNN.predict(x_cv)
print '<<<<<<<<<<<<<<<<<K-NEAREST NEIGHBOR(kNN) ALGORITHM>>>>>>>>>>>>>>>>>>'
print accuracy_score(y_cv,pred_cv5) #64.23%
matrix5=confusion_matrix(y_cv,pred_cv5)
print(matrix5)
print '<<<<<<<<<<<<<<<<<<<<<<K-NEAREST NEIGHBOR(kNN) ALGORITHM DONE!!!!>>>>>>>>>>>>>>>>>>>>>>>'
gbm=GradientBoostingClassifier()
gbm.fit(x_train,y_train)
pred_cv6=gbm.predict(x_cv)
print '<<<<<<<<<<<<<<<<<GRADIENT BOOSTING MACHINE ALGORITHM>>>>>>>>>>>>>>>>>>>>>'
print accuracy_score(y_cv,pred_cv6) #78.86%
matrix6=confusion_matrix(y_cv,pred_cv6)
print(matrix6)
print '<<<<<<<<<<<<<<<<<<GRADIENT BOOSTING MACHINE ALGORITHM DONE!!!>>>>>>>>>>>>>>>>>>>>>'
pred_test=nb.predict(df_test)
predictions=pd.DataFrame(test_prev_loans['customerid'], columns=['customerid'])
predictions['predictions'] = pred_test
predictions.to_csv('Predicted_output.csv') #Writing into a prediction file