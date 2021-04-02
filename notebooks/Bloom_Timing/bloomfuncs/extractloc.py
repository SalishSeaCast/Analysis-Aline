import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
import datetime as dt
from salishsea_tools import evaltools as et, places, viz_tools, visualisations
import xarray as xr
import pandas as pd
import pickle
import os
import bloomdrivers


def extract_loc(basedir,start,end,ij,ii,jw,iw,savepath):
    with xr.open_dataset('/data/vdo/MEOPAR/NEMO-forcing/grid/mesh_mask201702.nc') as mesh:
        tmask=np.array(mesh.tmask)
        gdept_1d=np.array(mesh.gdept_1d)
        e3t_0=np.array(mesh.e3t_0)
    nam_fmt='nowcast'
    flen=1 # files contain 1 day of data each
    ftype= 'ptrc_T' # load bio files
    tres=24 # 1: hourly resolution; 24: daily resolution  
    flist=et.index_model_files(start,end,basedir,nam_fmt,flen,ftype,tres)
    flist3 = et.index_model_files(start,end,basedir,nam_fmt,flen,"grid_T",tres)
    fliste3t = et.index_model_files(start,end,basedir,nam_fmt,flen,"carp_T",tres)

    ik=0
    with xr.open_mfdataset(flist['paths']) as bio:
        bio_time=np.array(bio.time_centered[:])
        no3_alld=np.array(bio.nitrate.isel(y=ij,x=ii)) 
        diat_alld=np.array(bio.diatoms.isel(y=ij,x=ii))
        flag_alld=np.array(bio.flagellates.isel(y=ij,x=ii))
        cili_alld=np.array(bio.ciliates.isel(y=ij,x=ii))
        microzoo_alld=np.array(bio.microzooplankton.isel(y=ij,x=ii))
        mesozoo_alld=np.array(bio.mesozooplankton.isel(y=ij,x=ii))
        
        with xr.open_mfdataset(fliste3t['paths']) as carp:
            intdiat=np.array(np.sum(bio.diatoms.isel(y=ij,x=ii)*carp.e3t.isel(y=ij,x=ii),1)) # depth integrated diatom
            intphyto=np.array(np.sum((bio.diatoms.isel(y=ij,x=ii)+bio.flagellates.isel(y=ij,x=ii)\
                            +bio.ciliates.isel(y=ij,x=ii))*carp.e3t.isel(y=ij,x=ii),1))
            spar=np.array(carp.PAR.isel(deptht=ik,y=ij,x=ii))
            intmesoz=np.array(np.sum(bio.mesozooplankton.isel(y=ij,x=ii)*carp.e3t.isel(y=ij,x=ii),1))
            intmicroz=np.array(np.sum(bio.microzooplankton.isel(y=ij,x=ii)*carp.e3t.isel(y=ij,x=ii),1))
    
    with xr.open_mfdataset(flist3['paths']) as grid:
        grid_time=np.array(grid.time_centered[:])
        temp=np.array(grid.votemper.isel(deptht=ik,y=ij,x=ii)) #surface temperature
        salinity=np.array(grid.vosaline.isel(deptht=ik,y=ij,x=ii)) #surface salinity

    opsdir='/results/forcing/atmospheric/GEM2.5/operational'

    flist2=et.index_model_files(start,end,opsdir,nam_fmt='ops',flen=1,ftype='None',tres=24)
    with xr.open_mfdataset(flist2['paths']) as winds:
        u_wind=np.array(winds.u_wind.isel(y=jw,x=iw))
        v_wind=np.array(winds.v_wind.isel(y=jw,x=iw))
        twind=np.array(winds.time_counter)
        solar=np.array(winds.solar.isel(y=jw,x=iw))

    allvars=(bio_time,diat_alld,no3_alld,flag_alld,cili_alld,microzoo_alld,mesozoo_alld,
              intdiat,intphyto,spar,intmesoz,intmicroz,
            grid_time,temp,salinity,u_wind,v_wind,twind,solar)
    pickle.dump(allvars,open(savepath,'wb'))


def runS3_1812_2015():
    # you could edit this to accept dates/locations as input and loop through them to extract all files

    start=dt.datetime(2016,1,1)
    end=dt.datetime(2016,3,31)
    year=str(start.year)
    modver='201812'
    loc='S3'
    ij,ii=places.PLACES['S3']['NEMO grid ji']
    # GEM2.5 grid ji is atm forcing grid for ops files
    jw,iw=places.PLACES['S3']['GEM2.5 grid ji']
    
    savedir='/ocean/aisabell/MEOPAR/extracted_files'
    #savedir='/ocean/eolson/MEOPAR/'
    
    fname=f'testJanToMarch_TimeSeries_{year}_{loc}_{modver}.pkl'
    savepath=os.path.join(savedir,fname)
    loc='S3'
    
    basedir='/results/SalishSea/nowcast-green.201812/'

    extract_loc(basedir,start,end,ij,ii,jw,iw,savepath)

if __name__ == "__main__":
    runS3_1812_2015()
