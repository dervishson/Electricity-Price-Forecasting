# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 16:55:04 2022

@author: ibrahim
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from statsmodels.graphics.tsaplots import plot_pacf
plt.rcParams["figure.figsize"]=(10,6)
mpl.rcParams["axes.linewidth"]=3

###Data retrieving from the  source#####
df=pd.read_csv("https://raw.githubusercontent.com/dervishson/Electricity-Price-Forecasting/master/Daten%20Spotmarktanalyse3.csv")

####Spotprice lag identification########3
plot_pacf(df["spotprice"],lags=300);plt.xticks(np.arange(0,300,24));plt.grid(True);plt.xlabel("lag",fontweight="bold",style="oblique");plt.ylabel("PACF",fontweight="bold",style="oblique")
#########Load lag identification########
plot_pacf(df["load"],lags=300);plt.xticks(np.arange(0,300,24));plt.grid(True);plt.xlabel("lag",fontweight="bold",style="oblique");plt.ylabel("PACF",fontweight="bold",style="oblique")
#####Rel coal identification########
plot_pacf(df["rel_coal"],lags=300);plt.xticks(np.arange(0,300,24));plt.grid(True);plt.xlabel("lag",fontweight="bold",style="oblique");plt.ylabel("PACF",fontweight="bold",style="oblique")
#####Rel wind identification########
plot_pacf(df["rel_wind"],lags=300);plt.xticks(np.arange(0,300,24));plt.grid(True);plt.xlabel("lag",fontweight="bold",style="oblique");plt.ylabel("PACF",fontweight="bold",style="oblique")
#####Rel solar identification########
plot_pacf(df["rel_solar"],lags=300);plt.xticks(np.arange(0,300,24));plt.grid(True);plt.xlabel("lag",fontweight="bold",style="oblique");plt.ylabel("PACF",fontweight="bold",style="oblique")
#####W_Avg identification########
plot_pacf(df["W_Avg"],lags=300);plt.xticks(np.arange(0,300,24));plt.grid(True);plt.xlabel("lag",fontweight="bold",style="oblique");plt.ylabel("PACF",fontweight="bold",style="oblique")
#####intra_day identification########
plot_pacf(df["intra_day"],lags=300);plt.xticks(np.arange(0,300,24));plt.grid(True);plt.xlabel("lag",fontweight="bold",style="oblique");plt.ylabel("PACF",fontweight="bold",style="oblique")
#####EUETS_EU identification########
plot_pacf(df["EUETS_EU"],lags=300);plt.xticks(np.arange(0,300,24));plt.grid(True);plt.xlabel("lag",fontweight="bold",style="oblique");plt.ylabel("PACF",fontweight="bold",style="oblique")
#####gas_price identification########
plot_pacf(df["gas_price"],lags=300);plt.xticks(np.arange(0,300,24));plt.grid(True);plt.xlabel("lag",fontweight="bold",style="oblique");plt.ylabel("PACF",fontweight="bold",style="oblique")

#########Detecting pairwise correlations#########
df.columns

spotp=df["spotprice"]
load=df["load"]
rel_coal=df["rel_coal"]
rel_gas=df["rel_gas"]
rel_wind=df["rel_wind"]
rel_solar=df["rel_solar"]
W_Avg=df["W_Avg"]
S_Avg=df["S_Avg"]
intra_day=df["intra_day"]
EUETS_EU=df["EUETS_EU"]

print("Load correlation is ", spotp.corr(load))             #0.51
print("rel_coal correlation is ", spotp.corr(rel_coal))     #0.12
print("rel_gas correlation is ", spotp.corr(rel_gas))       #-0.18
print("rel_wind correlation is ", spotp.corr(rel_wind))     #-0.16
print("rel_solar correlation is ", spotp.corr(rel_solar))   #0.1
print("W_Avg correlation is ", spotp.corr(W_Avg))           #-0.39
print("S_Avg correlation is ", spotp.corr(S_Avg))           #-0.05
print("intra_day correlation is ", spotp.corr(intra_day))   #0.96
print("EUETS_EU correlation is ", spotp.corr(EUETS_EU))     #0.41

"""
Might be rels are one group, intraday,EUETS,load another single groups. In total, 4 groups
along with spotprice input."""
"""
Forecast Period            First Observation               Last Observation              Training Period
1  Oct17                2017-10-01 00:00:00 CEST        2017-10-30 22:00:00 CET     2016-10-05 00:00:00 CEST - 2017-09-30 23:00:00 CEST
2  Nov17                2017-10-30 23:00:00 CET         2017-11-29 22:00:00 CET     2016-11-04 23:00:00 CET - 2017-10-30 22:00:00 CET
3  Dec17                2017-11-29 23:00:00 CET         2017-12-29 22:00:00 CET     2016-12-04 23:00:00 CET - 2017-11-29 22:00:00 CET
4  Jan18                2017-12-29 23:00:00 CET         2018-01-28 22:00:00 CET     2017-01-03 23:00:00 CET - 2017-12-29 22:00:00 CET
5  Feb18                2018-01-28 23:00:00 CET         2018-02-27 22:00:00 CET     2017-02-02 23:00:00 CET - 2018-01-28 22:00:00 CET
6  Mar18                2018-02-27 23:00:00 CET         2018-03-29 23:00:00 CEST    2017-03-04 23:00:00 CET - 2018-02-27 22:00:00 CET
7  Apr18                2018-03-30 00:00:00 CEST        2018-04-28 23:00:00 CEST    2017-04-04 00:00:00 CEST - 2018-03-29 23:00:00 CEST
8  Mai18                2018-04-29 00:00:00 CEST        2018-05-28 23:00:00 CEST    2017-05-04 00:00:00 CEST - 2018-04-28 23:00:00 CEST
9  Jun18                2018-05-29 00:00:00 CEST        2018-06-27 23:00:00 CEST    2017-06-03 00:00:00 CEST - 2018-05-28 23:00:00 CEST
10 Jul18                2018-06-28 00:00:00 CEST        2018-07-28 23:00:00 CEST    2017-07-03 00:00:00 CEST - 2018-06-27 23:00:00 CEST
11 Aug18                2018-07-29 00:00:00 CEST        2018-08-27 23:00:00 CEST    2017-08-02 00:00:00 CEST - 2018-07-28 23:00:00 CEST
12 Sep18                2018-08-28 00:00:00 CEST        2018-09-27 23:00:00 CEST    2017-09-01 00:00:00 CEST - 2018-08-27 23:00:00 CEST
"""

#For training Oct17 (Modified according to 168 time lag)
train_start=df.loc[df["date"]=="2016-09-28 00:00:00"].index.tolist()[0]
train_end=df.loc[df["date"]=="2017-09-30 23:00:00"].index.tolist()[0]+1

test_start=df.loc[df["date"]=="2017-09-24 00:00:00"].index.tolist()[0]
######Problem indication#####
"""Test  set includes 719 in original however we have chosen 720, but paper uses 720 datapoints for each month. Correct it if necessary"""
test_end=df.loc[df["date"]=="2017-10-30 23:00:00"].index.tolist()[0]


P=(df["spotprice"].to_numpy()).astype(float).reshape(-1,1)
L=(df["load"].to_numpy()).astype(float).reshape(-1,1)
R_c=(df["rel_coal"].to_numpy()).astype(float).reshape(-1,1)
R_g=(df["rel_gas"].to_numpy()).astype(float).reshape(-1,1)
R_w=(df["rel_wind"].to_numpy()).astype(float).reshape(-1,1)
R_s=(df["rel_solar"].to_numpy()).astype(float).reshape(-1,1)
W_a=(df["W_Avg"].to_numpy()).astype(float).reshape(-1,1)
S_a=(df["S_Avg"].to_numpy()).astype(float).reshape(-1,1)
I=(df["intra_day"].to_numpy()).astype(float).reshape(-1,1)
E=(df["EUETS_EU"].to_numpy()).astype(float).reshape(-1,1)   # maybe less effective


"""Hour cycle"""
H=((df["dummyhour"].to_numpy()).astype(float).reshape(-1,1))
h_c=np.sin(2*np.pi*H/24)
"""Day dummies"""
D=(pd.get_dummies(df["dummyweek"])).to_numpy().astype(float)

####Mean and std values for train inputs######
m_p,s_p=np.mean(P[train_start:train_end]),np.std(P[train_start:train_end])
m_l,s_l=np.mean(L[train_start:train_end]),np.std(L[train_start:train_end])
m_i,s_i=np.mean(I[train_start:train_end]),np.std(I[train_start:train_end])

train_price=np.zeros(((train_end-168-train_start)*168,1))
train_load=np.zeros(((train_end-168-train_start)*168,1))
train_intra=np.zeros(((train_end-168-train_start)*168,1))
train_day=np.zeros(((train_end-168-train_start)*168,7))
train_hour=np.zeros(((train_end-168-train_start)*168,1))

for i in range(train_end-train_start-168):
    train_price[i*168:(i+1)*168,0]=P[train_start+i:train_start+i+168,0]
    train_load[i*168:(i+1)*168,0]=L[train_start+i:train_start+i+168,0]
    train_intra[i*168:(i+1)*168,0]=I[train_start+i:train_start+i+168,0]
    train_day[i*168:(i+1)*168,:]=D[train_start+i:train_start+i+168,:]
    train_hour[i*168:(i+1)*168,0]=h_c[train_start+i:train_start+i+168,0]

train_price=train_price.reshape(-1,168,1)
train_load=train_load.reshape(-1,168,1)
train_intra=train_intra.reshape(-1,168,1)
train_hour=train_hour.reshape(-1,168,1)
train_day=train_day.reshape(-1,168,7)

###Normalizations###
train_price[:,:,0]=(train_price[:,:,0]-m_p)/s_p
train_load[:,:,0]=(train_load[:,:,0]-m_l)/s_l
train_intra[:,:,0]=(train_intra[:,:,0]-m_i)/s_i


train_labs=P[train_start+168:train_end,0].reshape(-1,1)
train_labs=(train_labs-m_p)/s_p


test_price=np.zeros(((test_end-test_start-168)*168,1))
test_load=np.zeros(((test_end-test_start-168)*168,1))
test_intra=np.zeros(((test_end-test_start-168)*168,1))
test_day=np.zeros(((test_end-test_start-168)*168,7))
test_hour=np.zeros(((test_end-test_start-168)*168,1))

for i in range(test_end-test_start-168):
    test_price[i*168:(i+1)*168,0]=P[test_start+i:test_start+i+168,0]
    test_load[i*168:(i+1)*168,0]=L[test_start+i:test_start+i+168,0]
    test_intra[i*168:(i+1)*168,0]=I[test_start+i:test_start+i+168,0]
    test_day[i*168:(i+1)*168,:]=D[test_start+i:test_start+i+168,:]
    test_hour[i*168:(i+1)*168,0]=h_c[test_start+i:test_start+i+168,0]

test_price=test_price.reshape(-1,168,1)
test_load=test_load.reshape(-1,168,1)
test_intra=test_intra.reshape(-1,168,1)
test_hour=test_hour.reshape(-1,168,1)
test_day=test_day.reshape(-1,168,7)

###Normalizations###
test_price[:,:,0]=(test_price[:,:,0]-m_p)/s_p
test_load[:,:,0]=(test_load[:,:,0]-m_l)/s_l
test_intra[:,:,0]=(test_intra[:,:,0]-m_i)/s_i


test_labs=P[test_start+168:test_end,0].reshape(-1,1)
test_labs=(test_labs-m_p)/s_p


