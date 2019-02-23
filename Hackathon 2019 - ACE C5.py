#!/usr/bin/env python
# coding: utf-8

# In[850]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import cross_val_score,KFold
from sklearn.neighbors import KNeighborsRegressor


# In[851]:


###DATA FRAMES###

# Supplied Dataset BAH #
power_df = pd.DataFrame.from_csv("energy_db.csv")

# GDP - World Bank Group #
gdp_df = pd.DataFrame.from_csv("gdp_power.csv")

# Population - World Bank Group #
population_df = pd.DataFrame.from_csv("population.csv")

# Hydropower Status Report - International Hydropower Association # 
hydro_df = pd.DataFrame.from_csv("hydro_power.csv")

# Freedom in the World 2018 - Freedomhouse.org #
freedom_df = pd.DataFrame.from_csv("freedom.csv")

# Natural Disaster Probability - World Risk Report 2016 #
disasters_df = pd.DataFrame.from_csv("NaturalDisasterChance.csv")

# Nuclear Power Statistics - World Nuclear Association #
nuclear_df = pd.DataFrame.from_csv("Third World Nuclear Totals.csv")

# Wind Energy Statistics - World Energy Council #
wind_df = pd.DataFrame.from_csv("Wind energy - Wind energy.csv")

# Solar Energy Statistics - World Energy Council / World Bank Group #
solar_df = pd.DataFrame.from_csv("solar_potential - hydro_power.csv")

# Class Attributes for Training Data #
classattr_df = pd.DataFrame.from_csv("TrainClassAtributes.csv")


# In[852]:


# Remove irrelevant values #
power_df = power_df.drop(columns=["gppd_idnr", "owner", "source", "url", "geolocation_source", "year_of_capacity_data", "generation_gwh_2013", "generation_gwh_2014", "generation_gwh_2015", "generation_gwh_2016"])


# In[853]:


# Dataframe Merge #


# In[854]:


power_df = pd.merge(power_df, gdp_df, on=["country"])


# In[855]:


power_df = pd.merge(power_df, population_df, on=["country"])


# In[856]:


power_df = pd.merge(power_df, hydro_df, on=["country_long"])


# In[857]:


power_df = pd.merge(power_df, freedom_df, on=["country_long"])


# In[858]:


power_df = pd.merge(power_df, disasters_df, on=["country_long"])


# In[859]:


power_df = pd.merge(power_df, nuclear_df, on=["country_long"])


# In[860]:


power_df = pd.merge(power_df, wind_df, on=["country_long"])


# In[861]:


power_df = pd.merge(power_df, solar_df, on=["country_long"])


# In[862]:


power_df = pd.merge(power_df, classattr_df, on=["country_long"])


# In[863]:


# Primary Dataframe # 


# In[864]:


len(list(power_df))

power_df = power_df.drop(columns=["fuel1","fuel2","fuel3","fuel4"])

# In[867]:


power_df


# In[868]:


features = power_df.copy()


# In[ ]:





# In[869]:


features = features.drop(columns="top resources")
features = features.drop(columns="name")
features = features.drop(columns="country_long")


# In[881]:


feature_list = list(features)


# In[882]:


feature_list.remove(feature_list[5])
feature_list.remove(feature_list[6])
feature_list.remove(feature_list[7])
feature_list.remove(feature_list[8])


# In[ ]:





# In[883]:


X = power_df[feature_list]


# In[ ]:





# In[ ]:





# In[885]:


y = power_df["top resources"]


# In[886]:


# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X, y)

# predict the response values for the observations in X
logreg.predict(X)


# In[ ]:





# In[ ]:




