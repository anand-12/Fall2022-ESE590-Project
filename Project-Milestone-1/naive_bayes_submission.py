import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import confusion_matrix, classification_report
warnings.filterwarnings("ignore")

def visual(df):


    l = list(df.columns)
    print(l)
 
 
    sns.set_style("ticks")
    sns.pairplot(df, hue='Survived', diag_kind="hist")
    plt.show()
    #plt.savefig('pairplot.png')



df=pd.read_csv("train_clean.csv") 

df.drop(["Cabin","Name","PassengerId","Ticket"],axis=1,inplace=True)
df=df[['Age', 'Fare', 'Embarked','Parch', 'Pclass', 'Sex', 'SibSp', 'Title', 'Family_Size','Survived']]
df.Survived=df.Survived.astype(int)
df2 = df.copy(deep=True)
visual(df2)
df2.drop(columns=["Survived"], inplace=True)
df2["Sex"].replace(["male", "female"], [0,1], inplace=True)
df2["Embarked"].replace(["S", "C", "Q"], [0,1,2], inplace=True)
df2["Title"].replace(["Mr", "Mrs", "Miss", "Master", "Dr", "Rev"], [0,1,2,3,4,5], inplace=True)

corr_matrix = df2.corr()
sns.heatmap(corr_matrix, annot=True)
sns.set_style("ticks")
plt.show()

l = len(df.columns) - 1
train, test = train_test_split(df, test_size=0.2, random_state=31)

survived_yes=train.loc[train.Survived==1]
survived_no=train.loc[train.Survived==0]

prior_yes=len(survived_yes)/len(train)
prior_no=len(survived_no)/len(train)

    
atr=list(df.columns.values)
truth = []
pred = []

for i in test.itertuples():
    test1=list(i)
    test1.pop(0) 
    ans=test1.pop() 
    truth.append(ans)
    py=1
    for i in range(l):
        likelihood_yes = train[(train[atr[i]] == test1[i]) & (train.Survived == 1)].count().values.item(0)
        py = py * (likelihood_yes) / len(survived_yes)
        total_yes = py * prior_yes
    pn=1
    for i in range(l):
        likelihood_no = train[(train[atr[i]] == test1[i]) & (train.Survived == 0)].count().values.item(0)
        pn = pn * (likelihood_no) / len(survived_no)
        total_no = pn * prior_no
    if total_yes>total_no:
        pred.append(1)      
    else:
        pred.append(0)
        
print(confusion_matrix(truth, pred))
print(classification_report(truth, pred))


