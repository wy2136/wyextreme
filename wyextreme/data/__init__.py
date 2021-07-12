#!/usr/bin/env python
# Wenchang Yang (wenchang@princeton.edu)
# Sun Jul 11 16:59:25 EDT 2021
if __name__ == '__main__':
    from misc.timer import Timer
    tt = Timer(f'start {__file__}')
import sys, os.path, os, glob, datetime
import xarray as xr, numpy as np, pandas as pd, matplotlib.pyplot as plt
#more imports
#
if __name__ == '__main__':
    tt.check('end import')
#
#start from here
idir = os.path.dirname(__file__)

ifile = os.path.join(idir, 'TXx_era5_index.nc')
TXx_ERA5 = xr.open_dataarray(ifile)

ifile = os.path.join(idir, 'GISS_GMST_4yrRollingMean.nc')
GMST = xr.open_dataarray(ifile)

ifile = os.path.join(idir, 'TXx_FLOR_CTL1860_0101-1000.nc')
TXx_FLOR_CTL1860 = xr.open_dataarray(ifile)

ifile = os.path.join(idir, 'TXx_FLOR_CTL1990_0101-1000.nc')
TXx_FLOR_CTL1990 = xr.open_dataarray(ifile)


 
 
if __name__ == '__main__':
    #from wyconfig import * #my plot settings
    
    #savefig
    if 'savefig' in sys.argv:
        figname = __file__.replace('.py', f'.png')
        wysavefig(figname)
    tt.check(f'**Done**')
    plt.show()
    
