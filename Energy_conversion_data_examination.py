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

df=pd.read_csv("Daten Spotmarktanalyse3.csv")
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
print("S_Avg correlation is ", spotp.corr(S_Avg))           #-0.5
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

#For training Oct17
train_start=df.loc[df["date"]=="2016-10-05 00:00:00"].index.tolist()[0]
train_end=df.loc[df["date"]=="2017-09-30 23:00:00"].index.tolist()[0]