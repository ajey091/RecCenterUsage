import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy
from sklearn.svm import SVR
import seaborn as sns

data = pd.read_excel('Anonymized - 2017 Summer, 2017 Fall, 2018 Spring CoRec Swipe Data.xlsx',header=1)
data2 = data.sample(n=1000,random_state = 2)
data2 = data2[data2['Overall GPA'].notna()]
sns.scatterplot(x="Year Swipes", y="Overall GPA", data=data2)
plt.savefig('Swipes_GPA.png',dpi=300)

X = np.array(data2['Year Swipes'])
X = sm.add_constant(X)
y = np.array(data2['Overall GPA'])

res_ols = sm.OLS(y, X).fit()

fig, ax = plt.subplots(figsize=(10,9))
ax.plot(X[:,1], y, 'o', label="Raw Data")
ax.plot(X[:,1], res_ols.fittedvalues, 'r',label="Fitted OLS",LineWidth=4)
plt.legend(loc='lower right')
plt.xlabel("Year Swipes")
plt.ylabel("Overall GPA")
plt.rcParams.update({'font.size': 26})
plt.tight_layout()
plt.savefig('OLS.png', bbox_inches = 'tight',
    pad_inches = 0,dpi=300)
plt.show()

residuals = (res_ols.fittedvalues - y)**2

fig, ax = plt.subplots(figsize=(10,9))
ax.plot(X[:,1], residuals, 'bo')
ax.set_xlabel('Year Swipes')
ax.set_ylabel('Squared residuals')
plt.ylim([-0.1,4])
plt.savefig('SquaredResiduals.png', dpi=300)

exp_params = scipy.optimize.curve_fit(lambda t,a,b: a*np.exp(b*t),
                         X[:,1], residuals,p0=(-4,-0.1) )
exp_vals = exp_params[0][0] * np.exp(exp_params[0][1]*np.sort(X[:,1]))

fig, ax = plt.subplots(figsize=(10,9))
ax.plot(X[:,1], residuals, 'bo',label='Raw data')
ax.plot(np.sort(X[:,1]),exp_vals,'r',LineWidth=4,label='Exponential fit')
ax.set_xlabel('Year Swipes')
ax.set_ylabel('Squared residuals')
plt.legend(loc='upper right')
plt.ylim([-0.1,4])
plt.savefig('SquaredResiduals_fitted.png', dpi=300)

w_exp = 1/exp_vals
res_wls_exp = sm.WLS(y, X, weights=w_exp).fit()

fig, ax = plt.subplots(figsize=(10,9))
plt.plot(X[:,1], y, 'o', label="Raw Data")
plt.plot(X[:,1], res_ols.fittedvalues, 'r--',label="OLS",LineWidth=4)
plt.plot(X[:,1], res_wls_exp.fittedvalues, 'g',label="WLS (exp)",LineWidth=4)
plt.ylim([1.5,4.1])
plt.xlabel("Year Swipes")
plt.ylabel("Overall GPA")
plt.legend(loc='lower right')
plt.savefig('WLS_exp.png', dpi=300)

rgr = SVR(C=10, epsilon=0.2)
rgr.fit(X[:,1].reshape(-1, 1), residuals.reshape(-1, 1))

w_svm = 1/rgr.predict(X[:,1].reshape(-1,1))
res_wls_svm = sm.WLS(y, X, weights=w_svm).fit()

fig, ax = plt.subplots(figsize=(10,9))
plt.plot(X[:,1], y, 'o', label="Raw Data")
plt.plot(X[:,1], res_ols.fittedvalues, 'r--',label="OLS",LineWidth=4)
plt.plot(X[:,1], res_wls_svm.fittedvalues, 'g',label="WLS (SVM)",LineWidth=4)
plt.ylim([1.5,4.1])
plt.xlabel("Year Swipes")
plt.ylabel("Overall GPA")
plt.legend(loc='lower right')
plt.savefig('WLS_svm.png', dpi=300)


from statsmodels.iolib.table import (SimpleTable, default_txt_fmt)
mydata = np.vstack([[res_ols.params], [res_wls_exp.params], [res_wls_svm.params]])
mydata = np.round(mydata,4)
headers = [ "a", "b" ]
rows = [ "OLS", "WLS (exp)", "WLS (SVM)" ]
tabl_1 = SimpleTable(mydata, headers, rows, txt_fmt=default_txt_fmt)
print(tabl_1)

mydata = np.vstack([[res_ols.bse], [res_wls_exp.bse], [res_wls_svm.bse]])
mydata = np.round(mydata,4)
headers = [ "Std err (a)", "Std err (b)" ]
rows = [ "OLS", "WLS (exp)", "WLS (SVM)" ]
tabl_2 = SimpleTable(mydata, headers, rows, txt_fmt=default_txt_fmt)
print(tabl_2)
