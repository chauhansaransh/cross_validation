import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn import model_selection
from sklearn import datasets
from sklearn import manifold

get_ipython().run_line_magic('matplotlib', 'inline')




# CROSS VALIDATION

# In[3]:


df=pd.read_csv("winequality-red.csv")


# In[4]:


df.tail()


# In[5]:


a=df.quality.unique()
a.sort()
a


# In[6]:


quality_mapping={3:0,
                 4:1,
                 5:2,
                 6:3,
                 7:4,
                 8:5}


# In[7]:


df.loc[:,"quality"]=df.quality.map(quality_mapping)


# In[8]:


df.tail()


# In[9]:


df=df.sample(frac=1).reset_index(drop=True)


# In[10]:


df_train=df.head(1000)
df_train.tail()


# In[11]:


df_test=df.tail(599)
df_test.tail()


# In[12]:


from sklearn import tree
from sklearn import metrics


# In[13]:


clf=tree.DecisionTreeClassifier(max_depth=12)


# In[14]:


df.columns


# In[15]:


cols=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']


# In[16]:


clf.fit(df_train[cols],df_train.quality)


# In[17]:


train_predictions=clf.predict(df_train[cols])
test_predictions=clf.predict(df_test[cols])


# In[18]:


train_accuracy=metrics.accuracy_score(df_train.quality,train_predictions)
test_accuracy=metrics.accuracy_score(df_test.quality,test_predictions)


# In[19]:


train_accuracy


# In[20]:


test_accuracy


# In[23]:


matplotlib.rc('xtick',labelsize=20)
matplotlib.rc('ytick',labelsize=20)


# In[26]:


train_accuracies=[0.5]
test_accuracies=[0.5]
cols=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']


# In[27]:


for depth in range(1,25):
    clf=tree.DecisionTreeClassifier(max_depth=depth)
    clf.fit(df_train[cols],df_train.quality)
    train_predictions=clf.predict(df_train[cols])
    test_predictions=clf.predict(df_test[cols])
    train_accuracy=metrics.accuracy_score(df_train.quality,train_predictions)
    test_accuracy=metrics.accuracy_score(df_test.quality,test_predictions)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)


# In[30]:


plt.figure(figsize=(10,5))
sns.set_style("whitegrid")
plt.plot(train_accuracies,label="train accuracy")
plt.plot(test_accuracies,label="test accuracy")
plt.legend(loc="upper left",prop={'size' : 15})
plt.xticks(range(0,26,5))
plt.xlabel("max_depth",size=20)
plt.ylabel("accuracy",size=20)
plt.show()


# In[31]:


b=sns.countplot(x='quality',data=df)
b.set_xlabel("quality",fontsize=20)
b.set_ylabel("count",fontsize=20)


# In[34]:


df["kfold"]=-1
df=df.sample(frac=1).reset_index(drop=True)
y=df.quality.values


# In[41]:


kf=model_selection.StratifiedKFold(n_splits=5)
for fold,(trn_,val_) in enumerate(kf.split(X=df,y=y)):
    df.loc[val_,'kfold']=fold


# In[42]:


df.head()


# In[45]:


b=sns.countplot(x='kfold',data=df)
b.set_xlabel("kfold",fontsize=20)
b.set_ylabel("count",fontsize=20)


# In[ ]:




