# Fady Gouda & Griffin Noe
# CSCI 297
# Professor Watson
# November 19, 2020

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, QuantileTransformer
from sklearn import preprocessing
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from mlxtend.plotting import heatmap
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, accuracy_score
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
from mlxtend.plotting import heatmap
from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


cols = ['spmaxl', 'jdze', 'nhm', 'f01nn', 'f04cn', 
        'nssssc', 'ncb', 'cp', 'ncp', 'no', 'f03cn', 
        'sdssc', 'hywibm', 'loc', 'sm6l', 'f03co', 'me', 
        'mi', 'nnn', 'narno2', 'ncrx3', 'spposabp', 'ncir', 
        'b01cbr' 'b03ccl', 'n073', 'spmaxa', 'psii1d', 'b04cbr', 
        'sdo', 'ti2l', 'ncrt', 'c026', 'f02cn', 'nhdon', 'spmaxbm', 
        'psiia', 'nh', 'sm6bm', 'narcoor', 'nx', 'target']

data=pd.read_csv('NewBioDeg.csv', header=0, names=cols)

le = preprocessing.LabelEncoder()
le.fit(data['target'])
data['target']=le.transform(data['target'])

# print("\nfeature\t\t# of NAs\n\n"+str(data.isna().sum()))

data = data.dropna()

x_train, x_test, y_train, y_test = train_test_split(data.drop('target', axis=1),
                                                    data['target'], test_size=0.3, 
                                                    random_state=1)

cm = np.corrcoef(data[cols].values.T)
hm = heatmap(cm, row_names=cols, column_names=cols)
#plt.show()

sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)
# acc=0

# for i in range(1,41):
#         pca=PCA(n_components=i)
#         pca.fit(x_train_std,y_train)
#         x_train_pca= pca.transform (x_train_std)
#         x_test_pca=pca.transform(x_test_std)
#         print("Post-PCA shape of x-train:", x_train_pca.shape)
#         # SVM_grid=[{
#         #     'C':[0.001, 0.01, 100, 1000], 'kernel': ['linear','rbf'],'degree':[1,2],'gamma':[100,0.01,0.1]
#         # }]

#         # SVM= GridSearchCV(SVR(),SVM_grid,verbose=1,scoring='r2')
#         # SVM.fit(x_train_std,y_train)
#         # SVM_pred=SVM.predict(x_test_std)
#         # print("SVM acc score:", accuracy_score(y_test,SVM_pred))

#         # # Print the results of the most important features according to univariate selection
#         # print('Univariate Selection Results: ')
#         # print(sorted(best_features.keys(), reverse=True))
#         # print('Number of features selected: ', len(best_features.keys()))

#         # svr=SVR(C=1000,kernel='rbf',degree=2,gamma=0.03)
#         # svr.fit(x_train_pca,y_train)
#         # svr_pred=svr.predict(x_test_pca)
#         # print("SVM acc score:", accuracy_score(y_test,svr_pred))

#         svc=SVC(C=1000,kernel='rbf',degree=2,gamma=0.03)
#         svc.fit(x_train_pca,y_train)
#         svc_pred=svc.predict(x_test_pca)
#         if accuracy_score(y_test,svc_pred) > acc:
#                 acc=accuracy_score(y_test,svc_pred)
                
#         print("SVM acc score:", accuracy_score(y_test,svc_pred))

# print (acc)

# # Instantiate the LDA model and fit it to the train data
# lda = LDA(n_components=1)
# lda.fit(x_train_std, y_train)

# # Transform the x train data with the lda and output the size
# x_train_lda = lda.transform(x_train_std)
# x_test_lda = lda.transform(x_test_std)

# svc=SVC(C=1000,kernel='rbf',degree=2,gamma=0.03)
# svc.fit(x_train_lda,y_train)
# svc_pred=svc.predict(x_test_lda)
# print("SVM lda accuracy score: %.2f" % (accuracy_score(y_test,svc_pred)*100), "%")

pca=PCA(n_components=23)
pca.fit(x_train_std,y_train)
x_train_pca= pca.transform (x_train_std)
x_test_pca=pca.transform(x_test_std)

# lda = LDA(n_components=1)
# lda.fit(x_train_pca, y_train)

# x_train_pca = lda.transform(x_train_pca)
# x_test_pca = lda.transform(x_test_pca)

svc=SVC(C=500,kernel='rbf',degree=2,gamma=0.03)
svc.fit(x_train_pca,y_train)
svc_pred=svc.predict(x_test_pca)
print("\nTrain Cross Val Score:\n", cross_val_score(svc, x_train_pca, y_train))
print("\nTest Cross Val Score:\n", cross_val_score(svc, x_test_pca, y_test))
print("\nSVM pca accuracy score: %.2f" % (accuracy_score(y_test,svc_pred)*100), "%\n")
print(classification_report(y_test, svc_pred))

# # Plot the confusion matrix for the training and testing data
plot_confusion_matrix(svc, x_test_pca, y_test)
plt.title("Test Data Confusion Matrix")
plot_confusion_matrix(svc, x_train_pca, y_train)
plt.title("Train Data Confusion Matrix")
# plt.show()