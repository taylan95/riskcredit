#!/usr/bin/env python
# coding: utf-8

# # Context

# The original dataset contains 1000 entries with 20 categorial/symbolic attributes prepared by Prof. Hofmann. In this dataset, each entry represents a person who takes a credit by a bank. Each person is classified as good or bad credit risks according to the set of attributes. The link to the original dataset can be found below.

# # Content

# * Age (numeric)
# * Sex (text: male, female)
# * Job (numeric: 0 - unskilled and non-resident, 1 - unskilled and resident, 2 - skilled, 3 - highly skilled)
# * Housing (text: own, rent, or free)
# * Saving accounts (text - little, moderate, quite rich, rich)
# * Checking account (numeric, in DM - Deutsch Mark)
# * Credit amount (numeric, in DM)
# * Duration (numeric, in month)
# * Purpose (text: car, furniture/equipment, radio/TV, domestic appliances, repairs, education, business, vacation/others)

# First of all, this work on Credit Risk is my first work I have tried so far.
# Our data for our work is German_Credit data.
# Firstly, I will start by loading our libraries and data, and then I will make some analysis.
# Let's start!!

# 1. Data & Data overview
# 2. Exploratory Data Analysis
# 3. Data Preprocessing & WoE
# 4. Variable Selection
# 5. Final Model
# 6. Evaluating

# **Library**

# In[1]:


import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,mean_squared_error


# In[2]:


dataset = pd.read_csv("german_credit_data_3.csv")


# In[3]:


#We take an overview of the first 5 lines of our data.


# In[4]:


dataset.head()


# In[5]:


#we look at the structure of the data.


# In[6]:


dataset.info()


# In[7]:


#we look at the missing data.


# In[8]:


dataset.isnull().sum()


# In[9]:


dataset.Risk.value_counts()


# In[10]:


plt.scatter(dataset['Credit amount'],dataset["Age"])
plt.figure()


# In[11]:


plt.scatter(dataset['Credit amount'],dataset["Duration"])
plt.figure();


# In[12]:


dataset.plot.hist(figsize = (5,10),subplots = True);


# In[13]:


sns.barplot(x = "Age",y = "Risk",hue = "Sex",data = dataset);


# In[14]:


sns.boxplot(x = "Job",y = "Credit amount", hue = "Sex",data = dataset);


# In[15]:


dataset.Duration.describe()


# In[16]:


dataset.Age.describe()


# In[17]:


df = dataset.copy()


# In[18]:


df = df.iloc[:,1:11]


# In[19]:


df.head()


# In[20]:


#Since our variables have some missing values, I assigned them as "no_inf".


# In[21]:


df["Saving accounts"].value_counts()


# In[22]:


df["Saving accounts"].fillna("no_inf",inplace = True)


# In[23]:


df["Checking account"].value_counts()


# In[24]:


df["Checking account"].fillna("no_inf",inplace = True)


# In[25]:


#Since the "Risk" variable is our target, we give "0" value to default customers and "1" to non-default customers.


# In[26]:


df["Risk"] = np.where((df["Risk"] == "good"),1,0)


# In[27]:


df["Job"].value_counts()


# In[28]:


#I preferred the "Label Encoder" method to numerate the gender variable


# In[29]:


lab = LabelEncoder()
df["Sex"] = lab.fit_transform(df["Sex"])


# In[30]:


df.head()


# In[31]:


categories_features = ["Job","Housing","Saving accounts","Checking account","Purpose"]
dummies = []
def dummynew(feat_nam):
    dummies.append(pd.get_dummies(df[feat_nam],prefix = feat_nam,prefix_sep = ":"))
    
for feat_nam in categories_features:
    dummynew(feat_nam)


# In[32]:


dummies_collected = pd.concat(dummies,axis = 1)   
df = pd.concat([df,dummies_collected],axis = 1)
df.shape


# In[33]:


#To avoid "overfitting", we divide the data into two labels which are train and test.


# In[34]:


y = df["Risk"]
X = df.drop("Risk",axis = 1)


# In[35]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)


# In[36]:


input_data = X_train
target_data = y_train


# # **WoE (Weight of Evidence)**

# Since we will use WoE after separating our data as train and test, we write a function to run WoE.
# It is used as a criterion for scanning variables in credit risk modeling projects, such as the probability of WoE Default.
# The weight of evidence tells the predictive power of an independent variable in relation to the
# dependent variable. Since it evolved from credit scoring world, it is generally described as a
# measure for the separation of good and bad customers. "Bad Customers" refers to the customers
# who defaulted on a loan. and "Good Customers" refers to the customers who paid the loan.

# In[37]:


def woe_cat(df, discrete_variabe_name, good_bad_variable_df):
    df = pd.concat([df[discrete_variabe_name], good_bad_variable_df], axis = 1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].mean()], axis = 1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    df = df.sort_values(['WoE'])
    df = df.reset_index(drop = True)
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()
    return df


# In[38]:


#I make the following settings to see the complete data 


# In[39]:


pd.options.display.max_columns = None


# In[40]:


def plot_by_woe(df_woe,rotation_of_x_axis_labels = 0):
    x = np.array(df_woe.iloc[:,0].apply(str))
    Y = df_woe["WoE"]
    plt.figure(figsize = (18,6))
    plt.plot(x,Y,marker = "o",linestyle = "--",color = "k")
    plt.xlabel(df_woe.columns[0])
    plt.ylabel("Weight of Evidence")
    plt.title(str("Weight of Evidence by" + df_woe.columns[0]))
    plt.xticks(rotation = rotation_of_x_axis_labels)


# In[41]:


df_temp = woe_cat(input_data,"Job",target_data)
df_temp


# In[42]:


plot_by_woe(df_temp)


# In[43]:


df_temp = woe_cat(input_data,"Housing",target_data)
df_temp


# In[44]:


plot_by_woe(df_temp)


# In[45]:


def input_data_dropper(input_data,key_list,axis = 1):
    for key in key_list:
        input_data = input_data.drop(key, axis = axis)
    return input_data


# In[46]:


key_list = ["Housing:free","Housing:rent"]
input_data["Housing:free_rent"] = sum([input_data["Housing:free"],input_data["Housing:rent"]])
input_data = input_data_dropper(input_data,key_list)


# In[47]:


input_data.info()


# In[48]:


df_temp = woe_cat(input_data,"Saving accounts",target_data)


# In[49]:


df_temp


# In[50]:


plot_by_woe(df_temp)


# In[51]:


key_list = ["Saving accounts:little","Saving accounts:moderate"]
input_data["Saving accounts:little_moderate"] = sum([input_data["Saving accounts:little"],input_data["Saving accounts:moderate"]])
input_data = input_data_dropper(input_data,key_list)


# In[52]:


df_temp = woe_cat(input_data,"Checking account",target_data)


# In[53]:


df_temp


# In[54]:


plot_by_woe(df_temp)


# In[55]:


df_temp = woe_cat(input_data,"Purpose",target_data)


# In[56]:


df_temp


# In[57]:


plot_by_woe(df_temp)


# In[58]:


key_list = ["Purpose:domestic appliances","Purpose:education","Purpose:business","Purpose:repairs"]
input_data["Purpose:dom_edu"] = sum([input_data["Purpose:domestic appliances"],input_data["Purpose:education"]])
input_data["Purpose:bus_rep"] = sum([input_data["Purpose:business"],input_data["Purpose:repairs"]])                         
input_data = input_data_dropper(input_data,key_list)


# In[59]:


def woe_continuous(df, discrete_var_name, good_bad_variable_df):
    df = pd.concat([df[discrete_var_name], good_bad_variable_df], axis = 1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].mean()], axis = 1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    #df = df.sort_values(['WoE'])
    #df = df.reset_index(drop = True)
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()
    return df


# In[60]:


input_data["Age_2"] = pd.cut(input_data["Age"],5)
df_temp = woe_continuous(input_data,"Age_2",target_data)


# In[61]:


df_temp


# In[62]:


plot_by_woe(df_temp)


# In[63]:


input_data["Age:30"] = np.where((input_data["Age"] <= 30),1,0)
input_data["Age:30-42"] = np.where((input_data["Age"]>30) & (input_data["Age"] <= 42),1,0)
input_data["Age:42-52"] = np.where((input_data["Age"]>42) & (input_data["Age"] <=52),1,0)
input_data["Age:52-64"] = np.where((input_data["Age"]>52) & (input_data["Age"] <= 64),1,0)
input_data["Age:64"] = np.where((input_data["Age"] > 64),1,0)


# In[64]:


input_data["cr_amount"] = pd.cut(input_data["Credit amount"],10)
df_temp = woe_continuous(input_data,"cr_amount",target_data)


# In[65]:


df_temp


# In[66]:


plot_by_woe(df_temp,rotation_of_x_axis_labels = 1)


# In[67]:


input_data["cr_amount:2068"] = np.where((input_data["Credit amount"] <= 2068),1,0)
input_data["cr_amount:2068-3885"] = np.where((input_data["Credit amount"] > 2068) & (input_data["Credit amount"] <= 3885),1,0)
input_data["cr_amount:3885-5703"] = np.where((input_data["Credit amount"] > 3885) & (input_data["Credit amount"] <= 5703),1,0)
input_data["cr_amount:5703-11155"] = np.where((input_data["Credit amount"] > 5703) & (input_data["Credit amount"] <= 11155),1,0)
input_data["cr_amount:11155"] = np.where((input_data["Credit amount"] > 11155),1,0)


# In[68]:


input_data["Duration"].unique()


# In[69]:


input_data["Duration_2"] = pd.cut(input_data["Duration"],10)
df_temp = woe_continuous(input_data,"Duration_2",target_data)


# In[70]:


df_temp


# In[71]:


plot_by_woe(df_temp)


# In[72]:


input_data["Duration:11"] = np.where((input_data["Duration"] <= 11),1,0)
input_data["Duration:11-18"] = np.where((input_data["Duration"] >11) & (input_data["Duration"] <= 18),1,0)
input_data["Duration:18-31"] = np.where((input_data["Duration"] >18) & (input_data["Duration"] <= 31),1,0)
input_data["Duration:31-38"] = np.where((input_data["Duration"] >31) & (input_data["Duration"] <= 38),1,0)
input_data["Duration:38-45"] = np.where((input_data["Duration"] >38) & (input_data["Duration"] <= 45),1,0)
input_data["Duration:45-52"] = np.where((input_data["Duration"] >45) & (input_data["Duration"] <= 52),1,0)
input_data["Duration:52"] = np.where((input_data["Duration"] > 52),1,0)


# In[73]:


key_list = ["cr_amount","Age_2","Duration_2","Purpose","Checking account","Saving accounts","Housing","Job","Age",
            "Credit amount","Duration"]
input_data = input_data_dropper(input_data,key_list)


# In[74]:


key_list = ["Job:0","Housing:own","Saving accounts:no_inf","Checking account:rich","Purpose:car","Age:64",
            "Duration:11","cr_amount:2068"]
input_data = input_data_dropper(input_data,key_list)


# In[75]:


#After numerating the X_train data, we will follow the same steps for X_test. I share this part below.


# In[76]:


X_train = input_data


# In[77]:


X_train.shape


# In[78]:


input_data = X_test


# In[79]:


input_data.shape


# In[80]:


input_data.head()


# In[81]:


key_list = ["Housing:free","Housing:rent"]
input_data["Housing:free_rent"] = sum([input_data["Housing:free"],input_data["Housing:rent"]])
input_data = input_data_dropper(input_data,key_list)


# In[82]:


key_list = ["Saving accounts:little","Saving accounts:moderate"]
input_data["Saving accounts:little_moderate"] = sum([input_data["Saving accounts:little"],input_data["Saving accounts:moderate"]])
input_data = input_data_dropper(input_data,key_list)


# In[83]:


key_list = ["Purpose:domestic appliances","Purpose:education","Purpose:business","Purpose:repairs"]
input_data["Purpose:dom_edu"] = sum([input_data["Purpose:domestic appliances"],input_data["Purpose:education"]])
input_data["Purpose:bus_rep"] = sum([input_data["Purpose:business"],input_data["Purpose:repairs"]])                         
input_data = input_data_dropper(input_data,key_list)


# In[84]:


input_data["Age_2"] = pd.cut(input_data["Age"],5)


# In[85]:


input_data["Age:30"] = np.where((input_data["Age"] <= 30),1,0)
input_data["Age:30-42"] = np.where((input_data["Age"]>30) & (input_data["Age"] <= 42),1,0)
input_data["Age:42-52"] = np.where((input_data["Age"]>42) & (input_data["Age"] <=52),1,0)
input_data["Age:52-64"] = np.where((input_data["Age"]>52) & (input_data["Age"] <= 64),1,0)
input_data["Age:64"] = np.where((input_data["Age"] > 64),1,0)


# In[86]:


input_data["cr_amount"] = pd.cut(input_data["Credit amount"],10)


# In[87]:


input_data["cr_amount:2068"] = np.where((input_data["Credit amount"] <= 2068),1,0)
input_data["cr_amount:2068-3885"] = np.where((input_data["Credit amount"] > 2068) & (input_data["Credit amount"] <= 3885),1,0)
input_data["cr_amount:3885-5703"] = np.where((input_data["Credit amount"] > 3885) & (input_data["Credit amount"] <= 5703),1,0)
input_data["cr_amount:5703-11155"] = np.where((input_data["Credit amount"] > 5703) & (input_data["Credit amount"] <= 11155),1,0)
input_data["cr_amount:11155"] = np.where((input_data["Credit amount"] > 11155),1,0)


# In[88]:


input_data["Duration:11"] = np.where((input_data["Duration"] <= 11),1,0)
input_data["Duration:11-18"] = np.where((input_data["Duration"] >11) & (input_data["Duration"] <= 18),1,0)
input_data["Duration:18-31"] = np.where((input_data["Duration"] >18) & (input_data["Duration"] <= 31),1,0)
input_data["Duration:31-38"] = np.where((input_data["Duration"] >31) & (input_data["Duration"] <= 38),1,0)
input_data["Duration:38-45"] = np.where((input_data["Duration"] >38) & (input_data["Duration"] <= 45),1,0)
input_data["Duration:45-52"] = np.where((input_data["Duration"] >45) & (input_data["Duration"] <= 52),1,0)
input_data["Duration:52"] = np.where((input_data["Duration"] > 52),1,0)


# In[89]:


#We extract the extra columns from X_test data that we created for X_test.


# In[90]:


key_list = ["cr_amount","Age_2","Purpose","Checking account","Saving accounts","Housing","Job","Age",
            "Credit amount","Duration"]
input_data = input_data_dropper(input_data,key_list)


# In[91]:


input_data.shape


# In[92]:


#In order not to fall into the dummy trap, we extract our reference variables from the data.


# In[93]:


key_list = ["Job:0","Housing:own","Saving accounts:no_inf","Checking account:rich","Purpose:car","Age:64",
            "Duration:11","cr_amount:2068"]
input_data = input_data_dropper(input_data,key_list)


# In[94]:


X_test = input_data


# In[95]:


X_test.shape


# In[96]:


X_train.shape


# # Variable Selection

# Feature Selection is one of the core concepts in machine learning which hugely impacts the performance of your model. The data features that you use to train your machine learning models have a huge influence on the performance you can achieve.Irrelevant or partially relevant features can negatively impact model performance.

# In[97]:


#First of all, I check whether it has constant feature.


# In[98]:


constant_features = [
        feat for feat in X_train.columns if X_train[feat].std() == 0]
len(constant_features)


# In[99]:


#I check for duplicate data.


# In[100]:


data_t = X_train.T
data_t.head()
data_t.duplicated().sum()


# In[101]:


#Before running the model, I will look at the correlation so that if there are variables in it, 
#I will extract them from the train and test data.


# Correlation states how the features are related to each other or the target variable.Heatmap makes it easy to identify which features are most related to the target variable, we will plot heatmap of correlated features using the seaborn library.

# In[102]:


corrmat = X_train.corr()
fig, ax = plt.subplots()
fig.set_size_inches(11,11)
sns.heatmap(corrmat)


# In[103]:


def correlation(dataset,threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr
corr_features = correlation(X_train,0.8)
len(set(corr_features))#In this way, we can reach variables that are correlated with each other.


# Feature selection is an important pre-processing step for solving classification problems. A good feature selection method may not only improve the performance of the final classifier, but also reduce the computational complexity of it.

# In[104]:


#We make variable elimination with the Roc-auc curve.


# In[105]:


roc_values = []
for feature in X_train.columns:
    clf = LogisticRegression()
    clf.fit(X_train[feature].fillna(0).to_frame(),y_train)
    y_scored = clf.predict_proba(X_test[feature].fillna(0).to_frame())
    roc_values.append(roc_auc_score(y_test,y_scored[:,1]))

roc_values = pd.Series(roc_values)
roc_values.index = X_train.columns
roc_values.sort_values(ascending = False)


# In[106]:


roc_values.sort_values(ascending = False).plot.bar(figsize = (20,8))
len(roc_values[roc_values > 0.5])


# In[107]:


#We extract variables below 0.5 from our data.


# In[108]:


key_list = ["Age:52-64","Purpose:vacation/others","Duration:52","Purpose:bus_rep","Job:3","Duration:18-31",
            "cr_amount:3885-5703","Job:2"]
X_train = input_data_dropper(X_train,key_list)
X_test = input_data_dropper(X_test,key_list)


# In[109]:


#We extract our variables that are not significant comparing to p value, ie below 0.05. 
#However, if the part we need to pay attention is between 0.05 and all the dummies, we will eliminate variables.
#Since there is no package that we can see p-value ready for Logistic Regression, we will run it with the class method.


# In[110]:


from sklearn import linear_model
import scipy.stats as stat

class LogisticRegression_with_p_values:
    
    def __init__(self,*args,**kwargs):#,**kwargs):
        self.model = linear_model.LogisticRegression(*args,**kwargs)#,**args)

    def fit(self,X,y):
        self.model.fit(X,y)
        
        #### Get p-values for the fitted model ####
        denom = (2.0 * (1.0 + np.cosh(self.model.decision_function(X))))
        denom = np.tile(denom,(X.shape[1],1)).T
        F_ij = np.dot((X / denom).T,X) ## Fisher Information Matrix
        Cramer_Rao = np.linalg.inv(F_ij) ## Inverse Information Matrix
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores = self.model.coef_[0] / sigma_estimates # z-score for eaach model coefficient
        p_values = [stat.norm.sf(abs(x)) * 2 for x in z_scores] ### two tailed test for p-values
        
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.p_values = p_values
        
reg = LogisticRegression_with_p_values()
reg.fit(X_train, y_train)

variable_names = X_train.columns.values
summary_table = pd.DataFrame(columns = ["variable_names"],data = variable_names)
summary_table['Coefficients'] = np.transpose(reg.coef_)
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
summary_table = summary_table.sort_index()
p_values = reg.p_values
p_values = np.append(np.nan, np.array(p_values))
summary_table['p_values'] = p_values
summary_table


# In[111]:


#As a result of p_value; 
#Since it was observed that the variables, which are Sex, Age, Purpose and Job, are insignificant, we extract these values from the data.


# In[112]:


key_list = ["Sex","Job:1","Age:30","Age:30-42","Age:42-52","Purpose:furniture/equipment","Purpose:dom_edu",
            "Purpose:radio/TV"]
X_train = input_data_dropper(X_train,key_list)


# In[113]:


#Let's do it for X_test too.


# In[114]:


X_test = input_data_dropper(X_test,key_list)


# ## Final Model

# In[115]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
reg = LogisticRegression()
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)
cm = confusion_matrix(y_test,y_pred)


# In[116]:


reg.score(X_test,y_test)


# **KNN&Random Forest Results**

# In[117]:


#I will run the KNN and Random Forest algorithms to select the right model and select the model 
#according to the result of between Logistic, KNN and Random Forest


# In[118]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# Random forests are one the most popular machine learning algorithms. They are so successful because they provide in general a good predictive performance, low overfitting and easy interpretability.

# In[119]:


models = {"KNN" : KNeighborsClassifier(),
         "RandomForest" : RandomForestClassifier()}


# In[120]:


def model_results(models,X_train,X_test,y_train,y_test):
    np.random.seed(42)
    for name,model in models.items():
        model.fit(X_train,y_train)
        model.predict(X_test)
        print(name)
        print("Acc_Score:" + str(accuracy_score(y_test,model.predict(X_test))))
        print(classification_report(y_test,model.predict(X_test)))


# In[121]:


model_results(models,X_train,X_test,y_train,y_test)


# **Confusion Matrix**

# In the field of machine learning and specifically the problem of statistical classification, a confusion matrix, also known as an error matrix. A confusion matrix is a table that is often used to describe the performance of a classification model (or “classifier”) on a set of test data for which the true values are known. It allows the visualization of the performance of an algorithm.It allows easy identification of confusion between classes. 

# In[122]:


for name, model in models.items():
    ax= plt.subplot()
    plt.figure(figsize = (6,4))
    sns.heatmap(confusion_matrix(y_test,model.predict(X_test)), annot=True, ax = ax,fmt = "g"); 
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix:' + name); 
    ax.xaxis.set_ticklabels(['Good', 'Bad']); ax.yaxis.set_ticklabels(['Bad', 'Good']);
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top-0.5)


# **GridSearchCV & RandomizedSearchCV**

# When tuning the hyperparameters of an estimator, Grid Search and Random Search are both popular methods.
# Grid Search can be thought of as an exhaustive search for selecting a model. In Grid Search, the data scientist sets up a grid of hyperparameter values and for each combination, trains a model and scores on the testing data.

# In[123]:


#Tuning Logistic Regression


# In[124]:


from sklearn.model_selection import RandomizedSearchCV


# In[125]:


log_reg_grid = {"C": np.logspace(-4,4,20), 
                "solver": ["liblinear"]}
                
rf_gri  = {"n_estimators": np.arange(10,1000,50),
           "max_depth": [None, 3,5, 10],
           "min_samples_split": np.arange(2,20,2),
           "min_samples_leaf": np.arange(1,20,2)}


# In[126]:


rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                                param_distributions = log_reg_grid,
                                cv = 5,
                                n_iter = 20,
                                verbose = True)


# In[127]:


#fitting to Logistic Regression


# In[128]:


rs_log_reg.fit(X_train,y_train)


# In[129]:


rs_log_reg.best_params_


# In[130]:


rs_log_reg.score(X_train,y_train)


# In[131]:


#Tuning Random Forest


# In[132]:


rs_rf = RandomizedSearchCV(RandomForestClassifier(),
                           param_distributions=rf_gri,
                           cv = 5,
                           n_iter = 20,
                           verbose = True,
                           n_jobs = -1)


# In[133]:


#Fitting to RandomForestClassifier


# In[134]:


rs_rf.fit(X_train,y_train)


# In[135]:


rs_rf.score(X_test,y_test)


# In[136]:


#Since the logistic regression model works best, we will try to improve the model using GridSearchCV.


# In[137]:


log_reg_grid = {"C": np.logspace(-4,4,30),
                "solver": ["liblinear"]}


# In[138]:


from sklearn.model_selection import GridSearchCV


# In[139]:


gs_log_reg = GridSearchCV(LogisticRegression(),
                          param_grid = log_reg_grid,
                          cv = 5,
                          n_jobs = -1,
                          verbose = True)


# In[140]:


gs_log_reg.fit(X_train,y_train)


# In[141]:


gs_log_reg.best_params_


# In[142]:


gs_log_reg.score(X_test,y_test)


# In[143]:


Log = LogisticRegression(C = 78.47599703514607, solver = "liblinear")


# In[144]:


Log.fit(X_train,y_train)


# In[145]:


Log.score(X_test,y_test)


# In[146]:


#When we compare model results for Logistic Regression with GridSearchCV, we prefer the first model since we observed 
#that the parameters of the first model we ran were more successful.


# # Evaluating#

# In[147]:


accuracy_score(y_test,y_pred)


# In[148]:


from sklearn.model_selection import cross_val_score


# In[149]:


print(classification_report(y_test,y_pred))


# In[150]:


#Let's test the model performance with the Roc-auc curve and gini.


# An ROC curve (receiver operating characteristic curve) is a graph showing the performance of a classification model at all classification thresholds. This curve plots two parameters:
# 
# True Positive Rate(tpr),
# False Positive Rate(fpr)

# In[151]:


y_pred_proba = reg.predict_proba(X_test)
y_pred_proba_good = y_pred_proba[:,1]
y_pred_proba_bad = y_pred_proba[:,0]
y_test_2 = y_test
y_test_2.reset_index(drop = True, inplace = True)
actual_pred_probs = pd.concat([y_test_2,pd.DataFrame(y_pred_proba_good)],axis = 1)
actual_pred_probs.columns = ["y_test","y_test_proba"]
actual_pred_probs.index = X_test.index
actual_pred_probs = actual_pred_probs.sort_values("y_test_proba")
actual_pred_probs = actual_pred_probs.reset_index()


# In[152]:


from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test,y_pred_proba_good)
plt.plot(fpr,tpr)
plt.plot(fpr,fpr,linestyle = "--",color = "k")
plt.xlabel("fpr")
plt.ylabel("tpr")
plt.title("ROC_Curve")


# In[153]:


AUROC = roc_auc_score(y_test,y_pred_proba_good)


# if AUROC;
# btw 50% and 60% == BAD
# btw 60% and 70% == POOR
# btw 70% and 80% == FAIR
# btw 80% and 90% == GOOD
# btw 90% and 100% == EXCELLENT

# In[154]:


AUROC


# In[155]:


#Gini


# In[156]:


Gini = AUROC*2-1
Gini


# In[157]:


feature_dict = dict(zip(X_test.columns, list(reg.coef_[0])))


# In[164]:


probabilities_of_good.shape


# In[163]:


probabilities_of_good = y_pred_proba[:,1]


# In[165]:


Rating = []
for prob in probabilities_of_good:
    if prob >= 0.80:
        Rating.append("A")
    elif prob >= 0.60:
        Rating.append("B")
    elif prob >= 0.40:
        Rating.append("C")
    elif prob >= 0.20:
        Rating.append("D")
    else:
        Rating.append("E")


# In[166]:


Rating


# In[168]:


len(Rating)


# In[169]:


Rating.count("A")


# In[177]:


RatingScore = ["A","B","C","D","E"]
RatingName = []
Density = []
for i in RatingScore:
    RatingName.append(i)
    Density.append(((Rating.count(i)/len(Rating))**2))
    
Rating_Density = pd.DataFrame({"RatingScore":RatingName,"Density":Density})


# In[179]:


Rating_Density


# In[181]:


HHI = sum(Density)


# In[182]:


HHI


# In[183]:


y_pred


# In[188]:


y_pred_proba[:,0]


# In[189]:


PD = y_pred_proba[:,0]


# In[213]:


Score_model = pd.DataFrame({"PD":PD,"Default_Flag":y_pred,"Score":Rating})


# In[214]:


Score_model[:10]


# In[220]:


Score_model[(Score_model["Default_Flag"] == 1) & (Score_model["Score"] == "A")].Score.count()


# In[231]:


RatingScore = ["A","B","C","D","E"]
non_default = []
for score in RatingScore:
    non_default.append(Score_model[(Score_model["Default_Flag"] == 1) & (Score_model["Score"] == score)].Score.count())


# In[232]:


non_default


# In[233]:


RatingScore = ["A","B","C","D","E"]
default_count = []
for score in RatingScore:
    default_count.append(Score_model[(Score_model["Default_Flag"] == 0) & (Score_model["Score"] == score)].Score.count())


# In[234]:


default_count


# In[241]:


pd_average = []
for score in RatingScore:
    pd_average.append(Score_model[Score_model["Score"] == score].PD.mean())


# In[242]:


pd_average


# In[256]:


Score_model[Score_model["Score"] == "E"]


# In[260]:


Score_model.loc[Score_model["Score"] == "E","PD"] = 1


# In[261]:


Score_model.loc[Score_model["Score"] == "E","PD"]


# In[ ]:




