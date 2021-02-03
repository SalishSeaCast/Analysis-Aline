import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
import netCDF4 as nc
import datetime as dt
from salishsea_tools import evaltools as et, places, viz_tools, visualisations
import xarray as xr
import pandas as pd
import pickle
import os


# Metric 1:
def metric1_bloomtime(phyto_alld,no3_alld,bio_time):
    # a) get avg phytplankton in upper 3m
    phyto_alld_df=pd.DataFrame(phyto_alld)
    upper_3m_phyto=pd.DataFrame(phyto_alld_df[[0,1,2,3]].mean(axis=1))
    upper_3m_phyto.columns=['upper_3m_phyto']
    #upper_3m_phyto

    # b) get average no3 in upper 3m
    no3_alld_df=pd.DataFrame(no3_alld)
    upper_3m_no3=pd.DataFrame(no3_alld_df[[0,1,2,3]].mean(axis=1))
    upper_3m_no3.columns=['upper_3m_no3']
    #upper_3m_no3

    # make bio_time into a dataframe
    bio_time_df=pd.DataFrame(bio_time)
    bio_time_df.columns=['bio_time']
    metric1_df=pd.concat((bio_time_df,upper_3m_phyto,upper_3m_no3), axis=1)
    
    # c)  Find first location where nitrate crosses below 0.5 micromolar and 
    #     stays there for 2 days (inlcuding the first day I'm assuming?)
    for i, row in metric1_df.iterrows():
        try:
            if metric1_df['upper_3m_no3'].iloc[i]<0.5 and metric1_df['upper_3m_no3'].iloc[i+1]<0.5:
                location1=i
                break
        except IndexError:
            location1=np.nan
            print('bloom not found')

    # d) Find date with maximum phytoplankton concentration within four days (say 9 day window) of date in c)
    bloomrange=metric1_df[location1-4:location1+5]
    bloomtime1=bloomrange.loc[bloomrange.upper_3m_phyto.idxmax(), 'bio_time']

    return bloomtime1


# Metric 2: 
def metric2_bloomtime(sphyto,sno3,bio_time):
    
    df = pd.DataFrame({'bio_time':bio_time, 'sphyto':sphyto, 'sno3':sno3})

    # to find all the peaks:
    df['phytopeaks'] = df.sphyto[(df.sphyto.shift(1) < df.sphyto) & (df.sphyto.shift(-1) < df.sphyto)]
    
    # need to covert the value of interest from ug/L to uM N (conversion factor: 1.8 ug Chl per umol N)
    chlvalue=5/1.8

    # extract the bloom time date   
    for i, row in df.iterrows():
        try:
            if df['sphyto'].iloc[i-1]>chlvalue and df['sphyto'].iloc[i-2]>chlvalue and pd.notna(df['phytopeaks'].iloc[i]):
                bloomtime2=df.bio_time[i]
                break
            elif df['sphyto'].iloc[i+1]>chlvalue and df['sphyto'].iloc[i+2]>chlvalue and pd.notna(df['phytopeaks'].iloc[i]):
                bloomtime2=df.bio_time[i]
                break
        except IndexError:
            bloomtime2=np.nan
            print('bloom not found')
    return bloomtime2


# Metric 3: 
def metric3_bloomtime(sphyto,sno3,bio_time):
    # 1) determine threshold value    
    df = pd.DataFrame({'bio_time':bio_time, 'sphyto':sphyto, 'sno3':sno3})   
    
    # a) find median chl value of that year, add 5% (this is only feb-june, should we do the whole year?)
    threshold=df['sphyto'].median()*1.05
    # b) secondthresh = find 70% of threshold value
    secondthresh=threshold*0.7    

    # 2) Take the average of each week and make a dataframe with start date of week and weekly average
    weeklychl = pd.DataFrame(df.resample('W', on='bio_time').sphyto.mean())
    weeklychl.reset_index(inplace=True)

    # 3) Loop through the weeks and find the first week that reaches the threshold. 
        # Is one of the two week values after this week > secondthresh? 

    for i, row in weeklychl.iterrows():
        try:
            if weeklychl['sphyto'].iloc[i]>threshold and weeklychl['sphyto'].iloc[i+1]>secondthresh:
                bloomtime3=weeklychl.bio_time[i]
                break
            elif weeklychl['sphyto'].iloc[i]>threshold and weeklychl['sphyto'].iloc[i+2]>secondthresh:
                bloomtime3=weeklychl.bio_time[i]
                break
        except IndexError:
            bloomtime2=np.nan
            print('bloom not found')

    return bloomtime3

# wind speed cubed
def janfebmar_wspeed3(twind,wspeed):
    dfwind=pd.DataFrame({'twind':twind, 'wspeed':wspeed})
    monthlyws=pd.DataFrame(dfwind.resample('M', on='twind').wspeed.mean())
    monthlyws.reset_index(inplace=True)
    jan_ws3=(monthlyws.iloc[0]['wspeed'])**3
    feb_ws3=(monthlyws.iloc[1]['wspeed'])**3
    mar_ws3=(monthlyws.iloc[2]['wspeed'])**3
    return jan_ws3, feb_ws3, mar_ws3

# surface irradiance:
def janfebmar_irradiance(twind,solar):
    dfsolar=pd.DataFrame({'twind':twind, 'solar':solar})
    monthlysolar=pd.DataFrame(dfsolar.resample('M', on='twind').solar.mean())
    monthlysolar.reset_index(inplace=True)
    jan_solar=monthlysolar.iloc[0]['solar']
    feb_solar=monthlysolar.iloc[1]['solar']
    mar_solar=monthlysolar.iloc[2]['solar']
    return jan_solar, feb_solar, mar_solar

# surface PAR:
def janfebmar_spar(bio_time,spar):
    dfspar=pd.DataFrame({'bio_time':bio_time, 'spar':spar})
    monthlyspar=pd.DataFrame(dfspar.resample('M', on='bio_time').spar.mean())
    monthlyspar.reset_index(inplace=True)
    jan_spar=monthlyspar.iloc[0]['spar']
    feb_spar=monthlyspar.iloc[1]['spar']
    mar_spar=monthlyspar.iloc[2]['spar']
    return jan_spar, feb_spar, mar_spar

#surface temperature:
def janfebmar_temp(grid_time,temp):
    dftemp=pd.DataFrame({'grid_time':grid_time, 'temp':temp})
    monthlytemp=pd.DataFrame(dftemp.resample('M', on='grid_time').temp.mean())
    monthlytemp.reset_index(inplace=True)
    jan_temp=monthlytemp.iloc[0]['temp']
    feb_temp=monthlytemp.iloc[1]['temp']
    mar_temp=monthlytemp.iloc[2]['temp']
    return jan_temp, feb_temp, mar_temp

# surface salinity:
def janfebmar_temp(grid_time,salinity):
    dfsal=pd.DataFrame({'grid_time':grid_time, 'sal':salinity})
    monthlysal=pd.DataFrame(dfsal.resample('M', on='grid_time').sal.mean())
    monthlysal.reset_index(inplace=True)
    jan_sal=monthlysal.iloc[0]['sal']
    feb_sal=monthlysal.iloc[1]['sal']
    mar_sal=monthlysal.iloc[2]['sal']
    return jan_sal, feb_sal, mar_sal

# Fraser river flow:
def janfebmar_fraserflow(riv_time,rivFlow):
    dfrivFlow=pd.DataFrame({'riv_time':riv_time, 'rivFlow':rivFlow})
    dfrivFlow["riv_time"] = pd.to_datetime(dfrivFlow["riv_time"])
    monthlyrivFlow=pd.DataFrame(dfrivFlow.resample('M', on='riv_time').rivFlow.mean())
    monthlyrivFlow.reset_index(inplace=True)
    jan_rivFlow=monthlyrivFlow.iloc[0]['rivFlow']
    feb_rivFlow=monthlyrivFlow.iloc[1]['rivFlow']
    mar_rivFlow=monthlyrivFlow.iloc[2]['rivFlow']
    return jan_rivFlow, feb_rivFlow, mar_rivFlow

# surface zooplankton concentration:
def janfebmar_zooplankton(bio_time,zoop):
    dzoop=pd.DataFrame(zoop)
    szoop=np.array(dzoop[[0]]).flatten()
    dfzoop=pd.DataFrame({'bio_time':bio_time, 'zoop':szoop})
    monthlyzoop=pd.DataFrame(dfzoop.resample('M', on='bio_time').zoop.mean())
    monthlyzoop.reset_index(inplace=True)
    jan_zoop=monthlyzoop.iloc[0]['zoop']
    feb_zoop=monthlyzoop.iloc[1]['zoop']
    mar_zoop=monthlyzoop.iloc[2]['zoop']
    return jan_zoop, feb_zoop, mar_zoop