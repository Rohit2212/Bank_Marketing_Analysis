import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from plotly import tools
import plotly.plotly as py
import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from scipy.stats import kurtosis 


dataset = pd.read_csv('bank.csv')
dataset.head()
dataset.describe()
dataset.info()
dataset['job'].unique()
dataset['education'].unique()
dataset['contact'].unique()
dataset['poutcome'].unique()
dataset['month'].unique()
dataset['marital'].unique()
dataset['default'].unique()
dataset['age'].value_counts()


# Drop the Job,Education,Contact,Poutcome Occupations that are "Unknown"
dataset = dataset.drop(dataset.loc[dataset["job"] == "unknown"].index)
dataset = dataset.drop(dataset.loc[dataset["education"] == "unknown"].index)
dataset = dataset.drop(dataset.loc[dataset["contact"] == "unknown"].index)
dataset = dataset.drop(dataset.loc[dataset["poutcome"] == "unknown"].index)

term_deposits = dataset.copy()


dataset.hist(bins=30,figsize=(14,10),color="#E14906")


f, ax = plt.subplots(1,2, figsize=(16,8))

colors = ["#FA5858", "#64FE2E"]
labels ="Did not Open Term Suscriptions", "Opened Term Suscriptions"

plt.suptitle('Information on Term Suscriptions', fontsize=20)

dataset["deposit"].value_counts().plot.pie(explode=[0,0.20], autopct='%1.2f%%', ax=ax[0], shadow=True, colors=colors, 
                                             labels=labels, fontsize=12, startangle=30)
ax[0].set_ylabel('% of Condition of Loans', fontsize=14)
palette = ["#64FE2E", "#FA5858"]

sns.barplot(x="education", y="balance", hue="deposit", data=dataset, palette=palette, estimator=lambda x: len(x) / len(dataset) * 100)
ax[1].set(ylabel="(%)")
ax[1].set_xticklabels(dataset["education"].unique(), rotation=0, rotation_mode="anchor")
plt.show()



sns.pairplot(data=dataset)
sns.jointplot(x='campaign',y='age',data=dataset,kind='hex')

ax=sns.barplot(x='job',y='balance',data=dataset)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")

ax=sns.barplot(x='job',y='age',data=dataset)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")

figJ = plt.figure(figsize=(20,20))
g=sns.boxplot(x='job',y='balance',hue='deposit',data=dataset)
g.set_xticklabels(g.get_xticklabels(),rotation=90,ha="right")

figE = plt.figure(figsize=(20,20))
e=sns.boxplot(x='education',y='balance',hue='deposit',data=dataset)




# Admin and management are basically the same let's put it under the same categorical value
lst = [dataset]

for col in lst:
    col.loc[col["job"] == "admin.", "job"] = "management"


jobs=dataset['job'].unique()
print(jobs)
vals = dataset['job'].value_counts().tolist()
labels = jobs

data = [go.Bar(
            x=labels,
            y=vals,
    marker=dict(
    color="#FE9A2E")
    )]

layout = go.Layout(
    title="Count by Job",
)

fig = go.Figure(data=data, layout=layout)



plot(fig, filename='basic-bar')



#Married
valsMar = dataset['marital'].value_counts().tolist()
labelsMar = ['married', 'divorced', 'single']

dataMar = [go.Bar(
            x=labelsMar,
            y=valsMar,
    marker=dict(
    color="#FE9A2E")
    )]

layout = go.Layout(
    title="Count by Marital Status",
)

fig = go.Figure(data=dataMar, layout=layout)



plot(fig, filename='basic-bar')

figMar = plt.figure(figsize=(7,3))
mar = sns.barplot(x='marital',y='balance',hue='deposit',data=dataset)

#Married/Secondary yo
dataset = dataset.drop(dataset.loc[dataset["education"] == "unknown"].index)
dataset['education'].unique()

education_groups = dataset.groupby(['marital','education'], as_index=False)['balance'].median()

figEdu = plt.figure(figsize=(14,8))




sns.barplot(x="balance", y='marital',hue='education', data=education_groups,palette="RdBu")

plt.title('Median Balance by Educational/Marital Group', fontsize=16)

#Rahasya !
dataset['marital/education'] = np.nan
lst = [dataset]

for col in lst:
    col.loc[(col['marital'] == 'single') & (dataset['education'] == 'primary'), 'marital/education'] = 'single/primary'
    col.loc[(col['marital'] == 'married') & (dataset['education'] == 'primary'), 'marital/education'] = 'married/primary'
    col.loc[(col['marital'] == 'divorced') & (dataset['education'] == 'primary'), 'marital/education'] = 'divorced/primary'
    col.loc[(col['marital'] == 'single') & (dataset['education'] == 'secondary'), 'marital/education'] = 'single/secondary'
    col.loc[(col['marital'] == 'married') & (dataset['education'] == 'secondary'), 'marital/education'] = 'married/secondary'
    col.loc[(col['marital'] == 'divorced') & (dataset['education'] == 'secondary'), 'marital/education'] = 'divorced/secondary'
    col.loc[(col['marital'] == 'single') & (dataset['education'] == 'tertiary'), 'marital/education'] = 'single/tertiary'
    col.loc[(col['marital'] == 'married') & (dataset['education'] == 'tertiary'), 'marital/education'] = 'married/tertiary'
    col.loc[(col['marital'] == 'divorced') & (dataset['education'] == 'tertiary'), 'marital/education'] = 'divorced/tertiary'
    
    
dataset.head()

#Pairplot with mar/edu yo
sns.set(style="ticks")

sns.pairplot(dataset, hue="marital/education", palette="Set1")
plt.show()


# Let's see the group who had loans from the marital/education group

loan_balance = dataset.groupby(['marital/education', 'loan'], as_index=False)['balance'].median()


no_loan = loan_balance['balance'].loc[loan_balance['loan'] == 'no'].values
has_loan = loan_balance['balance'].loc[loan_balance['loan'] == 'yes'].values


labels = loan_balance['marital/education'].unique().tolist()


trace0 = go.Scatter(
    x=no_loan,
    y=labels,
    mode='markers',
    name='No Loan',
    marker=dict(
        color='rgb(175,238,238)',
        line=dict(
            color='rgb(0,139,139)',
            width=1,
        ),
        symbol='circle',
        size=16,
    )
)
trace1 = go.Scatter(
    x=has_loan,
    y=labels,
    mode='markers',
    name='Has a Previous Loan',
    marker=dict(
        color='rgb(250,128,114)',
        line=dict(
            color='rgb(178,34,34)',
            width=1,
        ),
        symbol='circle',
        size=16,
    )
)

data = [trace0, trace1]
layout = go.Layout(
    title="The Impact of Loans to Married/Educational Clusters",
    xaxis=dict(
        showgrid=False,
        showline=True,
        linecolor='rgb(102, 102, 102)',
        titlefont=dict(
            color='rgb(204, 204, 204)'
        ),
        tickfont=dict(
            color='rgb(102, 102, 102)',
        ),
        showticklabels=False,
        dtick=10,
        ticks='outside',
        tickcolor='rgb(102, 102, 102)',
    ),
    margin=dict(
        l=140,
        r=40,
        b=50,
        t=80
    ),
    legend=dict(
        font=dict(
            size=10,
        ),
        yanchor='middle',
        xanchor='right',
    ),
    width=1000,
    height=800,
    paper_bgcolor='rgb(255,250,250)',
    plot_bgcolor='rgb(255,255,255)',
    hovermode='closest',
)
fig = go.Figure(data=data, layout=layout)
plot(fig, filename='lowest-oecd-votes-cast')


dataset.kurtosis()

dataset.skew(axis=0)
dataset.skew(axis=1)

plt.hist(x=dataset.skew(axis=0),bins=20)
dataset.skew(axis=0).hist(bins=30,figsize=(14,10),color="#E14906")
plt.xlabel("Skew Axis=0")
dataset.skew(axis=1).hist(bins=30,figsize=(14,10),color="#E14906")
plt.xlabel("Skew Axis=1")
dataset.kurtosis().hist(bins=30,figsize=(14,10),color="#E14906")
plt.xlabel("Kurtosis")
plt.show()


#Correlation Part Yo
from sklearn.preprocessing import LabelEncoder
fig = plt.figure(figsize=(14,12))
dataset['deposit'] = LabelEncoder().fit_transform(dataset['deposit'])



# Separate both dataframes into 
numeric_dataset = dataset.select_dtypes(exclude="object")
# categorical_dataset = dataset.select_dtypes(include="object")

corr_numeric = numeric_dataset.corr()


ax = sns.heatmap(corr_numeric, cbar=True, cmap="RdBu_r")
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.title("Correlation Matrix", fontsize=16)
plt.show()



#The Classification Part Yo
loner = dataset.pop('loan')
X = dataset.iloc[:,0:15]
y = dataset.iloc[:,15]
X.head()


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelEncoderX = LabelEncoder()
le = LabelEncoder()
y=le.fit_transform(y)
loner = le.fit_transform(loner)
ls=['default','housing']
for i in ls:
    X[i] = labelEncoderX.fit_transform(dataset[i])
ct = ColumnTransformer(transformers =[('encoder',OneHotEncoder(categories='auto'),[1,2,3,7,9,14])],remainder='passthrough')
X=np.array(ct.fit_transform(X))
X.shape
y.shape
#X = pd.DataFrame(data = X)



#Standard Scaling
from sklearn.preprocessing import StandardScaler
scX = StandardScaler()
X = scX.fit_transform(X)



#Train-Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0,stratify=loner)



#Logistic Regression
from sklearn.linear_model import LogisticRegression
lrClassifier = LogisticRegression()
lrClassifier.fit(X_train,y_train)

y_pred=lrClassifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test,y_pred)

#K Cross Validation
# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = lrClassifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()


#KNN
# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
knnClassifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knnClassifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = knnClassifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test,y_pred)

#SVM
from sklearn.svm import SVC
svmClassifier = SVC(kernel = 'rbf', random_state = 0)
svmClassifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = svmClassifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test,y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = svmClassifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()


#NaiveBayes
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
nbClassifier = GaussianNB()
nbClassifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = nbClassifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test,y_pred)

#Decision Tree
# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
dcClassifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dcClassifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = dcClassifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test,y_pred)

#RAndom Forest
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
rfClassifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rfClassifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = rfClassifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test,y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = rfClassifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

#XGBoost yo
xgClassifier = xgb.XGBClassifier()
xgClassifier.fit(X_train,y_train)
# Predicting the Test set results
y_pred = xgClassifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,classification_report,roc_curve,auc,accuracy_score
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test,y_pred)
#roc = roc_curve(y_test,y_pred)
fpr, tpr, threshold = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# method I: plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()



# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = xgClassifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

#Accuracy
# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


#Conclusion Analysis
#Month yo
fig = plt.figure(figsize=(14,6))
sns.countplot(x='month',data=dataset)
fig = plt.figure(figsize=(14,6))
sns.countplot(x='month',hue='deposit',data=dataset)

#Age yo
fig = plt.figure(figsize=(14,6))
g=sns.countplot(x='age',hue='deposit',data=dataset)
g.set_xticklabels(g.get_xticklabels(),rotation=90,ha="right")

#Job yo
fig = plt.figure(figsize=(14,6))
g=sns.countplot(x='job',hue='deposit',data=dataset)
g.set_xticklabels(g.get_xticklabels(),rotation=90,ha="right")












