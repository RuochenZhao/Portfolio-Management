#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.version
sys.version_info


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[3]:


inpath = "/Users/zhaoruochen/Desktop/"
outpath = "/Users/zhaoruochen/Desktop/"


# In[4]:


infile = "ReturnsPortfolios (1).csv"

print(inpath+infile)
indata = pd.read_csv( inpath + infile )


# In[5]:


indata.head(5)


# In[6]:


RiskyAsset = ["MSCI EUROPE", "MSCI USA", "MSCI PACIFIC", "Treasury.Bond.10Y"]
RiskFreeAsset  = "Treasury.Bill.90D"


# In[7]:


indata.loc[1:3,RiskyAsset]


# In[8]:


#part 1
indata[RiskyAsset].mean()


# In[9]:


indata[RiskyAsset].median()


# In[10]:


indata[RiskyAsset].std()


# In[11]:


indata[RiskyAsset].skew()


# In[12]:


indata[RiskyAsset].kurt()


# In[13]:


a = indata[RiskyAsset].mean()
b = indata[RiskyAsset].std()


# In[14]:


a/b


# In[15]:


plt.scatter(a,b)


# In[16]:


#The MSCI USA index has the highest mean and median, representing a higher return.
#The MSCI PACIFIC index has the highest standard deviation, which corresponds to a higher level of risk.
#Overall, the Treasury Bond 10Y has highest return-risk ratio because of its significanly lower risk.
#This means that Tresuary Bond 10Y provides highest return for each additional unit of risk taken on.


# In[17]:


#part 2
indata["MSCI EUROPE"].plot.hist(stacked=True, bins=20)


# In[18]:


indata["MSCI USA"].plot.hist(stacked=True, bins=20)


# In[19]:


indata["MSCI PACIFIC"].plot.hist(stacked=True, bins=20)


# In[20]:


indata["Treasury.Bond.10Y"].plot.hist(stacked=True, bins=20)


# In[21]:


#We could see that overall, the Treasury.Bond.10Y is more centered and spans a narrower return range (-0.06-0.08). 
#This corresponds to its low standard deviation.
#MSCI Pacific and Treasury.Bond are more symmetric, which corresponds to their skew values that are close to 0.
#MSCI PACIFIC distribution is most varied (from -0.2 to 0.2), which corresponds to its highest standard deviation.
#Overall, MSCI USA has higher distributions in high returns. This corresponds to its highest mean and median return.


# In[22]:


#part 3
#covariance matrix
indata[RiskyAsset].cov()


# In[23]:


#Correlation Matrix
indata[RiskyAsset].corr()


# In[40]:


determinant_covariance = np.linalg.det(indata[RiskyAsset].cov())
if determinant_covariance != 0:
    print("The covariance matrix is non singular")


# In[26]:


def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)
check_symmetric(indata[RiskyAsset].cov())


# In[27]:


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)
is_pos_def(indata[RiskyAsset].cov())


# In[32]:


#part 4
#minimum variance portfolio weights
c = indata[RiskyAsset].cov()
inv_c = np.linalg.inv(c)
u = np.matrix([1,1,1,1])
u_t = np.matrix("1;1;1;1")
w_mvp = (u*inv_c)/(u*inv_c*u_t)
print("Minimum Variance Portfolio weights are", w_mvp)


# In[33]:


#minimum variance portfolio returns
m = np.matrix(indata[RiskyAsset].mean())
m_t = np.transpose(m)
return_mvp = w_mvp*m_t
print("Minimum Variance Portfolio return is", return_mvp)


# In[39]:


#minimum variance portfolio risk
w_mvp_t = np.transpose(w_mvp)
C = np.matrix(c)
var_mvp = float(w_mvp*C*w_mvp_t)
std_mvp = var_mvp**0.5
print("Minimum Variance Portfolio risk is", std_mvp)


# In[45]:


#Portfolio with expected return of 9% and minimum variance
monthly_return_p2 = 0.09/12
A_ = u*np.linalg.inv(c)*u.transpose()
B_ = u*np.linalg.inv(c)*m.transpose()
C_ = m*np.linalg.inv(c)*m.transpose()
delta = A_*C_-B_**2
lamda1 = (C_ - monthly_return_p2*B_)/delta
lamda2 = (monthly_return_p2*A_ - B_)/delta
a = float(lamda2)*np.linalg.inv(c)
b = float(lamda1)*np.linalg.inv(c)*u.transpose()
w_p2_old = b + a*m.transpose()
w_p2 = w_p2_old.transpose()
print("Weights of portfolio with expected return of 9% and minimum variance are", w_p2)
return_p2 = w_p2*m_t
print("Return of this portfolio is", return_p2)
w_p2_t = np.transpose(w_p2)
var_p2 = float(w_p2*C*w_p2_t)
std_p2 = var_p2**0.5
print("Risk of this portfolio is", std_p2)


# In[76]:


#Efficicent frontier (calculate the weights, returns and risk of at least 7 portfolios)
for i in range (2,9):
    monthly_return = i*0.01/12
    A_EF = u*np.linalg.inv(c)*u.transpose()
    B_EF = u*np.linalg.inv(c)*m.transpose()
    C_EF = m*np.linalg.inv(c)*m.transpose()
    delta_EF = A_EF*C_EF-B_EF**2
    lamda1_EF = (C_EF - monthly_return*B_EF)/delta
    lamda2_EF = (monthly_return*A_EF - B_EF)/delta
    a_EF = float(lamda2_EF)*np.linalg.inv(c)
    b_EF = float(lamda1_EF)*np.linalg.inv(c)*u.transpose()
    w_EF_old = b_EF + a_EF*m.transpose()
    w_EF = w_EF_old.transpose()
    print("Weights of portfolio with expected return of monthly return", monthly_return, "are", w_EF)
    return_EF = w_EF*m_t
    print("Return of this portfolio is", return_EF)
    w_EF_t = np.transpose(w_EF)
    var_EF = float(w_EF*C*w_EF_t)
    std_EF = var_EF**0.5
    print("Risk of this portfolio is", std_EF)
    print(" ")


# In[77]:


x = [0.019887, 0.055957, 0.041358, 0.049996, 0.020469243651232684, 0.017422098344221055, 0.0395253981012201, 0.03372393570627051, 0.02827649864916253, 0.023431318871676852, 0.01963931832195301, 0.017595133875889853, 0.01790766427942696]
y = [0.005627, 0.003608, 0.009155, 0.007798, 0.09/12, 0.00614397, 0.02/12, 0.03/12, 0.04/12, 0.05/12, 0.06/12, 0.07/12, 0.08/12]
plt.scatter(x,y)


# In[62]:


#equal weighted portfolio
w_equal = np.matrix([0.25,0.25,0.25,0.25])
w_equal_t = np.transpose(w_equal)
return_equal = w_equal*m_t
C = np.matrix(c)
var_equal = float(w_equal*C*w_equal_t)
std_equal = var_equal**0.5
print ("the return of equal weighted portfolio is", return_equal)
print ("the risk of equal weighted portfolio is", std_equal)


# In[78]:


#plot equal weighted portfolio in graph
x = [std_equal, 0.019887, 0.055957, 0.041358, 0.049996, 0.020469243651232684, 0.017422098344221055, 0.0395253981012201, 0.03372393570627051, 0.02827649864916253, 0.023431318871676852, 0.01963931832195301, 0.017595133875889853, 0.01790766427942696]
y = [return_equal, 0.005627, 0.003608, 0.009155, 0.007798, 0.09/12, 0.00614397, 0.02/12, 0.03/12, 0.04/12, 0.05/12, 0.06/12, 0.07/12, 0.08/12]
plt.scatter(x,y)


# In[75]:


#part 5
#market portfolio 1:MSCI WORLD
for i in list(indata[RiskyAsset]):
    import pandas as pd
    import statsmodels.formula.api as sm
    df = pd.DataFrame({"A":list(indata[i]), "B": list(indata["MSCI AC WORLD"])})
    result = sm.ols(formula = "A~B", data=df).fit()
    print(" ")
    print("The beta of asset",i, "to the market portfolio is the B below")
    print(result.params)


# In[74]:


#market portfolio 2: MSCI USA
for i in list(indata[RiskyAsset]):
    import pandas as pd
    import statsmodels.formula.api as sm
    df = pd.DataFrame({"A":list(indata[i]), "B": list(indata["MSCI USA"])})
    result = sm.ols(formula = "A~B", data=df).fit()
    print(" ")
    print("The beta of asset",i, "to the market portfolio is the B below")
    print(result.params)


# In[98]:


#PART 6
#take lamda = 0.5
pd.ewma(indata.loc[1:3,RiskyAsset], com=0.5, span=None, halflife=None, min_periods=0, freq=None, adjust=True, how=None)


# In[93]:


#take lamda = 0.7
pd.ewma(indata.loc[1:3,RiskyAsset], com=0.7, span=None, halflife=None, min_periods=0, freq=None, adjust=True, how=None)


# In[100]:


#take lamda = 0.8
pd.ewma(indata.loc[1:3,RiskyAsset], com=0.7, span=None, halflife=None, min_periods=0, freq=None, adjust=True, how=None)


# In[102]:


#take lamda = 0.9
pd.ewma(indata.loc[1:3,RiskyAsset], com=0.9, span=None, halflife=None, min_periods=0, freq=None, adjust=True, how=None)


# In[ ]:




