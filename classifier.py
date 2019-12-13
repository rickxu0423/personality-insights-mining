import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

#plt.rcParams["figure.figsize"] = (16, 9)
plt.style.use('ggplot')

#fig = plt.figure()
#ax = Axes3D(fig)
df = pd.read_csv("data/personality-labeled.csv")
"""
df.plot.scatter(x="Component2", y="Component3")
sns.FacetGrid(df, 
    hue="Label").map(plt.scatter, "Component1", "Component2").add_legend()

plt.show()
"""
y = np.asarray(df.Label)
df_selected = df.drop(["Label"], axis=1)
df_X = df_selected.to_dict(orient='records')
vec = DictVectorizer()
X = vec.fit_transform(df_X).toarray()

"""
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.20, random_state=42)
"""

# Randomforest
# initialize
clf = RandomForestClassifier()

# K-Fold Cross Validation
acc_train, acc_test, precision, recall = [], [], [], []
cv = KFold(n_splits=10, random_state=42, shuffle=False)
for train_index, test_index in cv.split(X):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    # train the classifier using the training data
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    # compute accuracy using test data
    acc_train.append(clf.score(X_train, y_train))
    acc_test.append(clf.score(X_test, y_test))
    precision.append(precision_score(y_test, pred, average="weighted"))
    recall.append(recall_score(y_test, pred, average="weighted"))
print ("Train Accuracy:", acc_train)
print ("Mean Train Accuracy:", np.mean(acc_train))
print("Test Accuracy:", acc_test)
print ("Mean Test Accuracy:", np.mean(acc_test))
print ("Precision:", precision)
print ("Mean Precision:", np.mean(precision))
print ("Recall:", recall)
print ("Mean Recall:", np.mean(recall))

# Boxplot
fig1, ax1 = plt.subplots()
ax1.set_title('Boxplot of 10-fold Cross Validation Accuracy')
ax1.set_ylabel('Accurancy')
ax1.boxplot([acc_train, acc_test])
ax1.set_xticklabels(["Train", "Test"])
ax1.figure.savefig('boxplot1.png')

fig2, ax2 = plt.subplots()
ax2.set_title('Boxplot of Precision and Recall')
ax2.set_ylabel('Percentage')
ax2.boxplot([precision, recall])
ax2.set_xticklabels(["Precision", "Recall"])
ax2.figure.savefig('boxplot2.png')




"""
new = [[-0.29323978839853354,5.181183888470347,9.976173355441897e-16],
        [5.826563601170381,-2.3899742956952457,9.976173355441897e-16],
        [-5.533323812771847,-2.7912095927751004,9.976173355441895e-16]]
class_code = clf.predict(new)
print(class_code)
"""