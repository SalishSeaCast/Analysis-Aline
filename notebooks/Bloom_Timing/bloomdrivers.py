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
import gsw

# Extracting winds from the correct path
def getWindVarsYear(year):
    if year>2014:
        opsdir='/results/forcing/atmospheric/GEM2.5/operational/'
        nam_fmt='ops'
        jW,iW=places.PLACES['S3']['GEM2.5 grid ji']
    else:
        opsdir='/data/eolson/results/MEOPAR/GEMLAM/'
        nam_fmt='gemlam'
        jW=135
        iW=145
    return jW,iW,opsdir,nam_fmt

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
    #     stays there for 2 days 
    # NOTE: changed the value to 2 micromolar
    for i, row in metric1_df.iterrows():
        try:
            if metric1_df['upper_3m_no3'].iloc[i]<2 and metric1_df['upper_3m_no3'].iloc[i+1]<2:
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
def janfebmar_zooplankton(bio_time,zoop_alld):
    dzoop=pd.DataFrame(zoop_alld)
    szoop=np.array(dzoop[[0]]).flatten()
    dfzoop=pd.DataFrame({'bio_time':bio_time, 'zoop':szoop})
    monthlyzoop=pd.DataFrame(dfzoop.resample('M', on='bio_time').zoop.mean())
    monthlyzoop.reset_index(inplace=True)
    jan_zoop=monthlyzoop.iloc[0]['zoop']
    feb_zoop=monthlyzoop.iloc[1]['zoop']
    mar_zoop=monthlyzoop.iloc[2]['zoop']
    return jan_zoop, feb_zoop, mar_zoop

# surface mesozooplankton concentration:
def janfebmar_mesozooplankton(bio_time,mesozoo_alld):
    dzoop=pd.DataFrame(mesozoo_alld)
    szoop=np.array(dzoop[[0]]).flatten()
    dfzoop=pd.DataFrame({'bio_time':bio_time, 'zoop':szoop})
    monthlyzoop=pd.DataFrame(dfzoop.resample('M', on='bio_time').zoop.mean())
    monthlyzoop.reset_index(inplace=True)
    jan_zoop=monthlyzoop.iloc[0]['zoop']
    feb_zoop=monthlyzoop.iloc[1]['zoop']
    mar_zoop=monthlyzoop.iloc[2]['zoop']
    return jan_zoop, feb_zoop, mar_zoop

# surface microzooplankton concentration:
def janfebmar_microzooplankton(bio_time,microzoo_alld):
    dzoop=pd.DataFrame(microzoo_alld)
    szoop=np.array(dzoop[[0]]).flatten()
    dfzoop=pd.DataFrame({'bio_time':bio_time, 'zoop':szoop})
    monthlyzoop=pd.DataFrame(dfzoop.resample('M', on='bio_time').zoop.mean())
    monthlyzoop.reset_index(inplace=True)
    jan_zoop=monthlyzoop.iloc[0]['zoop']
    feb_zoop=monthlyzoop.iloc[1]['zoop']
    mar_zoop=monthlyzoop.iloc[2]['zoop']
    return jan_zoop, feb_zoop, mar_zoop

# depth integrated zooplankton concentration:
def janfebmar_depth_intzoop(bio_time,intzoop):
    dfzoop=pd.DataFrame({'bio_time':bio_time, 'intzoop':intzoop})
    monthlyzoop=pd.DataFrame(dfzoop.resample('M', on='bio_time').intzoop.mean())
    monthlyzoop.reset_index(inplace=True)
    jan_zoop=monthlyzoop.iloc[0]['intzoop']
    feb_zoop=monthlyzoop.iloc[1]['intzoop']
    mar_zoop=monthlyzoop.iloc[2]['intzoop']
    return jan_zoop, feb_zoop, mar_zoop

# depth integrated mesozooplankton concentration:
def janfebmar_depth_intmesozoop(bio_time,intmesoz):
    dfzoop=pd.DataFrame({'bio_time':bio_time, 'intmesoz':intmesoz})
    monthlyzoop=pd.DataFrame(dfzoop.resample('M', on='bio_time').intmesoz.mean())
    monthlyzoop.reset_index(inplace=True)
    jan_zoop=monthlyzoop.iloc[0]['intmesoz']
    feb_zoop=monthlyzoop.iloc[1]['intmesoz']
    mar_zoop=monthlyzoop.iloc[2]['intmesoz']
    return jan_zoop, feb_zoop, mar_zoop

# depth integrated microzooplankton concentration:
def janfebmar_depth_intmicrozoop(bio_time,intmicroz):
    dfzoop=pd.DataFrame({'bio_time':bio_time, 'intmicroz':intmicroz})
    monthlyzoop=pd.DataFrame(dfzoop.resample('M', on='bio_time').intmicroz.mean())
    monthlyzoop.reset_index(inplace=True)
    jan_zoop=monthlyzoop.iloc[0]['intmicroz']
    feb_zoop=monthlyzoop.iloc[1]['intmicroz']
    mar_zoop=monthlyzoop.iloc[2]['intmicroz']
    return jan_zoop, feb_zoop, mar_zoop

# mid depth nitrate (30-90m):
def janfebmar_mid_depth_no3(bio_time,no3_30to90m):
    dfno3=pd.DataFrame({'bio_time':bio_time, 'no3_30to90m':no3_30to90m})
    monthlyno3=pd.DataFrame(dfno3.resample('M', on='bio_time').no3_30to90m.mean())
    monthlyno3.reset_index(inplace=True)
    jan_no3=monthlyno3.iloc[0]['no3_30to90m']
    feb_no3=monthlyno3.iloc[1]['no3_30to90m']
    mar_no3=monthlyno3.iloc[2]['no3_30to90m']
    return jan_no3, feb_no3, mar_no3

# deep no3 in entire SoG (past 250m):
def janfebmar_deepno3(riv_time,no3_past250m):
    dfdeepno3=pd.DataFrame({'riv_time':riv_time, 'no3_past250m':no3_past250m})
    dfdeepno3["riv_time"] = pd.to_datetime(dfdeepno3["riv_time"])
    monthlydeepno3=pd.DataFrame(dfdeepno3.resample('M',on='riv_time').no3_past250m.mean())
    monthlydeepno3.reset_index(inplace=True)
    jan_deepno3=monthlydeepno3.iloc[0]['no3_past250m']
    feb_deepno3=monthlydeepno3.iloc[1]['no3_past250m']
    mar_deepno3=monthlydeepno3.iloc[2]['no3_past250m']
    return jan_deepno3, feb_deepno3, mar_deepno3

def halo_de(ncname,ts_x,ts_y):
    
    ''' given a path to a SalishSeaCast netcdf file and an x, y pair, 
        returns halocline depth, where halocline depth is defined a midway between 
        two cells that have the largest salinity gradient
        ie max abs((sal1-sal2)/(depth1-depth2))

            Parameters:
                    ncname (str): path to a netcdf file containing 
                    a valid salinity variable (vosaline)
                    ts_x (int): x-coordinate at which halocline is calculated
                    tx_y (int): y-coordinate at which halocline is calculated
            Returns:
                    halocline_depth: depth in meters of maximum salinity gradient
    '''
    
     # o
        
    halocline = 0
    grid = nc.Dataset('/data/vdo/MEOPAR/NEMO-forcing/grid/mesh_mask201702.nc')
    nemo = nc.Dataset(ncname)
    
    #get the land mask
    col_mask = grid['tmask'][0,:,ts_y,ts_x] 
    
    #get the depths of the watercolumn and filter only cells that have water
    col_depths = grid['gdept_0'][0,:,ts_y,ts_x]
    col_depths = col_depths[col_mask==1] 

### if there is no water, no halocline
    if (len(col_depths) == 0):
        halocline = np.nan
    
    else: 
        #get the salinity of the point, again filtering for where water exists
        col_sal = nemo['vosaline'][0,:,ts_y,ts_x]
        col_sal = col_sal[col_mask==1]

        #get the gradient in salinity
        sal_grad = np.zeros_like(col_sal)

        for i in range(0, (len(col_sal)-1)):
            sal_grad[i] = np.abs((col_sal[i]-col_sal[i+1])/(col_depths[i]-col_depths[i+1]))

        #print(sal_grad)

        loc_max = np.where(sal_grad == np.nanmax(sal_grad))
        loc_max = (loc_max[0][0])

        #halocline is halfway between the two cells
        halocline = col_depths[loc_max] + 0.5*(col_depths[loc_max+1]-col_depths[loc_max])

    
    return halocline

# halocline time series:
def janfebmar_halocline(bio_time,halocline):
    dfhalo=pd.DataFrame({'bio_time':bio_time, 'halo':halocline})
    monthlyhalo=pd.DataFrame(dfhalo.resample('M', on='bio_time').halo.mean())
    monthlyhalo.reset_index(inplace=True)
    jan_halo=monthlyhalo.iloc[0]['halo']
    feb_halo=monthlyhalo.iloc[1]['halo']
    mar_halo=monthlyhalo.iloc[2]['halo']
    return jan_halo, feb_halo, mar_halo

# regression line and r2 value for plots
def reg_r2(driver,bloomdate):
    A = np.vstack([driver, np.ones(len(driver))]).T
    m, c = np.linalg.lstsq(A, bloomdate,rcond=None)[0]
    m=round(m,3)
    c=round(c,2)
    y = m*driver + c
    model, resid = np.linalg.lstsq(A, bloomdate,rcond=None)[:2]
    r2 = 1 - resid / (len(bloomdate) * np.var(bloomdate))
    return y, r2, m, c

# depth of turbocline
def turbo(eddy,time,depth):
    turbo=list()
    for day in eddy: 
        dfed=pd.DataFrame({'depth':depth[:-1], 'eddy':day[1:]}) #do depth T instead of W, depth[:-1], then day[1:]
        dfed=dfed.iloc[1:] # dropping surface values
        dfed[:21] #keep top 21 (25m depth)
        for i, row in dfed.iterrows():
            try:
                if dfed['eddy'].iloc[i]<0.001:
                    turbo.append(dfed.at[i,'depth'])
                    break
            except IndexError:
                turbo.append(np.nan)
                print('turbocline depth not found')
    dfturbo=pd.DataFrame({'time':time, 'turbo':turbo})
    monthlyturbo=pd.DataFrame(dfturbo.resample('M', on='time').turbo.mean())
    monthlyturbo.reset_index(inplace=True)
    jan_turbo=monthlyturbo.iloc[0]['turbo']
    feb_turbo=monthlyturbo.iloc[1]['turbo']
    mar_turbo=monthlyturbo.iloc[2]['turbo']
    return jan_turbo, feb_turbo, mar_turbo

def density_diff(sal,temp,time):
    p=0
    depthrange={5:5,10:10,15:15,19:20,20:25,21:30}
    density_diffs=dict()
    for ind,depth in depthrange.items():
        dsal=pd.DataFrame(sal)
        #isal=np.array(dsal[[depth]]).flatten()
        dtemp=pd.DataFrame(temp)
        #itemp=np.array(dsal[[depth]]).flatten()
        surfacedens=gsw.rho(dsal.iloc[:,0],dtemp.iloc[:,0],p)  # get the surface density
        idens=gsw.rho(dsal.iloc[:,ind],dtemp.iloc[:,ind],p)  # get the density at that depth
        densdiff=idens-surfacedens                               # get the dailiy desnity difference
        df=pd.DataFrame({'time':time, 'densdiff':densdiff})  # average over months
        monthlydiff=pd.DataFrame(df.resample('M', on='time').densdiff.mean())
        monthlydiff.reset_index(inplace=True)
        density_diffs[f'Jan {depth}m']=monthlydiff.iloc[0]['densdiff']
        density_diffs[f'Feb {depth}m']=monthlydiff.iloc[1]['densdiff']
        density_diffs[f'Mar {depth}m']=monthlydiff.iloc[2]['densdiff']
    return density_diffs


def avg_eddy(eddy,time,ij,ii):
    k1=15 # 15m depth is index 15 (actual value is 15.096255)
    k2=22 # 30m depth is index 22 (actual value is 31.101034)
    with xr.open_dataset('/data/vdo/MEOPAR/NEMO-forcing/grid/mesh_mask201702.nc') as mesh:
            tmask=np.array(mesh.tmask[0,:,ij,ii])
            e3t_0=np.array(mesh.e3t_0[0,:,ij,ii])
            e3t_k1=np.array(mesh.e3t_0[:,k1,ij,ii])
            e3t_k2=np.array(mesh.e3t_0[:,k1,ij,ii])
    # vertical sum of microzo in mmol/m3 * vertical grid thickness in m:
    inteddy=list()
    avgeddyk1=list()
    avgeddyk2=list()
    for dailyeddy in eddy:
        eddy_tgrid=(dailyeddy[1:]+dailyeddy[:-1])
        eddy_e3t=eddy_tgrid*e3t_0[:-1]
        avgeddyk1.append(np.sum(eddy_e3t[:k1]*tmask[:k1])/np.sum(e3t_0[:k1]))
        avgeddyk2.append(np.sum(eddy_e3t[:k2]*tmask[:k2])/np.sum(e3t_0[:k2]))

    df=pd.DataFrame({'time':time, 'eddyk1':avgeddyk1,'eddyk2':avgeddyk2})
    monthlyeddyk1=pd.DataFrame(df.resample('M', on='time').eddyk1.mean())
    monthlyeddyk2=pd.DataFrame(df.resample('M', on='time').eddyk2.mean())
    monthlyeddyk1.reset_index(inplace=True)
    monthlyeddyk2.reset_index(inplace=True)
    jan_eddyk1=monthlyeddyk1.iloc[0]['eddyk1']
    feb_eddyk1=monthlyeddyk1.iloc[1]['eddyk1']
    mar_eddyk1=monthlyeddyk1.iloc[2]['eddyk1']
    jan_eddyk2=monthlyeddyk2.iloc[0]['eddyk2']
    feb_eddyk2=monthlyeddyk2.iloc[1]['eddyk2']
    mar_eddyk2=monthlyeddyk2.iloc[2]['eddyk2']
    return jan_eddyk1, feb_eddyk1, mar_eddyk1,jan_eddyk2,feb_eddyk2,mar_eddyk2


        

