#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


supply=pd.read_csv("supply_data.csv")


# In[3]:


demand=pd.read_csv("demand_data.csv")


# In[4]:


supply.head(5)


# ## Supply_dataset(Monthly_data)

# 
# *1.Building Permits(Permit Number)*-     Number of building permits allotted.<br>
# *2.Construction Spending (Million $)*-  The amount spent (in millions of USD) is a measure of the activity in the construction industry.<br>
# *3.Housing Starts(New Housing Project)*-  This is a measure of the number of units of new housing projects started in a given period.<br>
# *4.Homes Sold(units)*-   House for sale is a basic measure of supply.<br>

# In[5]:


demand.head(5)


# ## Demand_dataset(Quaterly_data)

# *1.Mortgage Rates(%)*<br>
# *2.USA GDP(Billions$ )-Quarterly Real GDP (adjusted for inflation)*<br>
# *3.Unemployment(%)*<br>
# *4.Delinquency Rate(%) on Mortgages(Foreclosure on the mortgage)-an indicator of the number of foreclosures in real estate*<br>

# # DATA CLEANING

# In[6]:


# Removing non-usefull column name 'Unnamed:0'
supply=supply.drop(['Unnamed: 0'], axis=1)


# In[7]:


supply.head(5)


# In[8]:


print(supply.info())


# In[9]:


print(demand.info())


# In[10]:


# Converting "DATE" column in demand to consistent date format
demand['DATE'] = pd.to_datetime(demand['DATE'])


# In[11]:


# Converting "period" column in supply to consistent date format
supply['Period'] = pd.to_datetime(supply['Period'], format='%d-%m-%Y').dt.strftime('%Y-%m-%d')
supply.rename(columns={'Period': 'DATE'}, inplace=True)
supply['DATE'] = pd.to_datetime(supply['DATE'])


# In[12]:


#Mergeing supply and demand into DF
DF=pd.merge(supply,demand)


# In[13]:


DF.head(5)


# In[14]:


# I extracted the year, month and day from the DATE column to facilitate training, for the model.
DF['year'] = DF['DATE'].dt.year
DF['month'] = DF['DATE'].dt.month
DF['day'] = DF['DATE'].dt.day


# In[15]:


DF.head()


# In[16]:


DF= DF.drop('DATE', axis=1)


# In[17]:


DF.head(3)


# In[18]:


print(DF.columns)


# In[19]:


DF.isnull().sum()


# In[20]:



 sns.boxplot(x='month',y='HPI', data = DF)
 
  


# ## Checking outliers

# In[21]:



for feature in DF.columns:
    DF.boxplot(column=feature)
    plt.ylabel(feature)
    plt.title(feature)
    plt.show()


# ## NO outliers as such found .

# In[22]:


plt.figure(figsize=(15,7))
sns.barplot(x='year',y='HPI',data=DF)


# In[23]:


plt.figure(figsize=(15,5))
sns.lineplot(x='year',y='HPI',data=DF)


# In[24]:


sns.catplot(x="month",y="HPI",kind="violin", data=DF)


# In[25]:


sns.catplot(x='Homes_Sold',y='HPI',hue='month',data=DF)


# ## checking the corelation of different features with housing price Index(HPI) with heatmap

# In[26]:


plt.figure(figsize=(15,15))
sns.heatmap(DF.corr(),annot=True)


# ### GDPC and Construction have strong coorelation with HPI

# # Model Training

# In[27]:


y=DF['HPI']


# In[28]:


x=DF.drop(['HPI'], axis=1)


# In[29]:


x


# In[30]:


y


# In[31]:


#spliting our dependent and independent features
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)


# # Linear_Regression

# In[32]:


import math  
import sklearn.metrics  
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

regressor = LinearRegression()
regressor.fit(x_train,y_train)
predictions = regressor.predict(x_test)
mse = sklearn.metrics.mean_squared_error(y_test, predictions)
rmse = math.sqrt(mse)
print(rmse)


# In[33]:



y_pred=regressor.predict(x_test)
r2_score(y_test,y_pred)


# In[34]:


print("Intercept=", regressor.intercept_)
(pd.DataFrame(zip(x.columns, regressor.coef_)))


# In[35]:


intercept=regressor.intercept_
print(intercept)
coefs=regressor.coef_
print(coefs)


# In[36]:


print ("HPI = {0:.2f} + {1:.2f}*permit_NO + {2:.2f}*construction + {3:.2f}*Homes_sold + {4:.2f}*Housing_starts  +{5:.2f}*UNEM_rate +{6:.2f}*MORTAGE +{7:.2f}*GDPC1 +{8:.2f}*Foreclousers +{9:.2f}*year +{10:.2f}*month + {11:.2f}*day"
       .
       format(intercept, coefs[0], coefs[1], coefs[2], coefs[3],coefs[4], coefs[5], coefs[6], coefs[7],coefs[8], coefs[9], coefs[10]))


# In[41]:


plt.scatter(y_test,y_pred)
plt.title("Scatter plot Linear Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# ## HPI = 12655.62 + -0.02*permit_NO + 0.00*construction + 0.00*Homes_sold + -0.01*Housing_starts  +5.80*UNEM_rate +-1.27*MORTAGE +0.03*GDPC1 +-2.59*Foreclousers +-6.48*year +-0.59*month + 0.00*day

# # RandomForestRegressor
# 

# In[38]:


from sklearn.ensemble import RandomForestRegressor

random_regressor = RandomForestRegressor(oob_score =True ,n_jobs = 1,random_state =100)

random_regressor.fit(x_train, y_train)
preds = random_regressor.predict(x_test)

mean_error = sklearn.metrics.mean_squared_error(y_test, preds)  
root_mean = math.sqrt(mean_error) 
root_mean


# In[39]:


ry_pred=random_regressor.predict(x_test)
r2_score(y_test,ry_pred)


# In[42]:


plt.scatter(y_test,ry_pred)
plt.title("Scatter plot for Random forest ")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# In[ ]:




