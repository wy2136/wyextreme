#!/usr/bin/env python
# Wenchang Yang (wenchang@princeton.edu)
# Wed Jun 30 17:02:45 EDT 2021
if __name__ == '__main__':
    from misc.timer import Timer
    tt = Timer(f'start {__file__}')
import sys, os.path, os, glob, datetime
import xarray as xr, numpy as np, pandas as pd, matplotlib.pyplot as plt
#more imports
from scipy.optimize import minimize
from scipy.stats import rankdata, genextreme as gev
from numpy import exp, log
from tqdm import tqdm
import multiprocessing as mp
#
if __name__ == '__main__':
    tt.check('end import')
#
#start from here
# empirical cdf, ep and rp
def smp_cdf(data):
    """empirical cdf""" 
    xx = np.sort(da)
    yy = rankdata(xx)/xx.size
    return xx, yy
def smp_exceedance_prob(data):
    """empirical exceedance probability"""
    xx = np.sort(data)
    ep = 1- (rankdata(xx)-1)/xx.size
    return xx, ep
def plot_smp_exceedance_prob(data, ax=None, **kws):
    """"plot empirical exceedance probability"""
    if ax is None:
        fig, ax = plt.subplots()
    ls = kws.pop('ls', 'none')
    marker = kws.pop('marker', 'o')
    fillstyle = kws.pop('fillstyle', 'none')
    alpha = kws.pop('alpha', 0.5)
    xx, ep = smp_exceedance_prob(data)
    ax.plot(xx, ep, ls=ls, marker=marker, fillstyle=fillsytle, alpha=alpha, **kws)
    ax.set_yscale('log')
def smp_return_period(data):
    """empirical return period"""
    xx, ep = smp_exceedance_prob(data)
    rp = 1/ep
    return xx, rp
def plot_smp_return_period(data, ax=None, **kws):
    """plot empirical return period"""
    if ax is None:
        fig, ax = plt.subplots()
    ls = kws.pop('ls', 'none')
    marker = kws.pop('marker', 'o')
    fillstyle = kws.pop('fillstyle', 'none')
    alpha = kws.pop('alpha', 0.5)
    xx, rp = smp_return_period(data)
    ax.plot(rp, xx, ls=ls, marker=marker, fillstyle=fillstyle, alpha=alpha, **kws) 
    ax.set_xscale('log')

#theoretical cdf, ep and rp
#cdf
def gev_cdf(xx, mu, sigma, xi):
    """GEV cdf. 
        xx: sample(s)
        mu: location
        sigma: scale
        xi: shape
    Ref: https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution
    """
    s = (xx - mu)/sigma
    if xi == 0:
        return exp( -exp(-s) )
    else: #xi != 0
        # xi*s > -1: exp( -(1+xi*s)**(-1/xi) )
        # xi*s <=-1; xi<0: 1
        # xi*s <=-1; xi>0: 0
        return (xi*s>-1)*exp( -(1+xi*s)**(-1/xi) ) + (xi*s<=-1)*(xi<0)
def plot_cdf(mu=0, sigma=1, xi=0, user_defined=True, ax=None, **kws):
    if ax is None:
        fig, ax = plt.subplots()
    #mu, sigma, xi = 0, 1, -0.2
    xx = np.linspace( gev.ppf(0.001, c=-xi, loc=mu, scale=sigma), 
        gev.ppf(0.999, c=-xi, loc=mu, scale=sigma), 1000)
    if user_defined:
        ax.plot(xx, gev_cdf(xx, mu=mu, sigma=sigma, xi=xi), **kws)
    else:
        ax.plot(xx, gev.cdf(xx, c=-xi, loc=mu, scale=sigma), **kws)

#pdf
def gev_pdf(xx, mu, sigma, xi):
    """GEV pdf.
        xx: sample(s)
        mu: location
        sigma: scale
        xi: shape
    Ref: https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution
    """
    s = (xx-mu)/sigma
    if xi == 0:
        tx = exp(-s)
    else:
        tx = (1+xi*s)**(-1/xi)
    return (xi*s>-1)*(1/sigma) * (tx**(1+xi)) * exp(-tx)
def plot_pdf(mu=0, sigma=1, xi=0, user_defined=True, ax=None, **kws):
    if ax is None:
        fig,ax = plt.subplots()
    #mu, sigma, xi = 0, 1, -0.2
    xx = np.linspace( gev.ppf(0.001, c=-xi, loc=mu, scale=sigma), 
        gev.ppf(0.999, c=-xi, loc=mu, scale=sigma), 1000)
    if user_defined:
        ax.plot(xx, gev_pdf(xx, mu=mu, sigma=sigma, xi=xi), **kws)
    else:
        ax.plot(xx, gev.pdf(xx, c=-xi, loc=mu, scale=sigma), **kws) 

#exceedance probability
def gev_exceedance_prob(xx, mu, sigma, xi):
    """GEV exceedance probability"""
    return 1 - gev_cdf(xx, mu, sigma, xi)
def plot_exceedance_prob(mu=0, sigma=1, xi=0, lower=0.001, upper=0.999, ax=None, **kws):
    if ax is None:
        fig,ax = plt.subplots()
    xx = np.linspace( gev.ppf(1-upper, c=-xi, loc=mu, scale=sigma), 
        gev.ppf(1-lower, c=-xi, loc=mu, scale=sigma), 1000)
    ep = gev_exceedance_prob(xx, mu, sigma, xi) 
    ax.plot(xx, ep, **kws)
    ax.set_yscale('log')

#return period
def gev_return_period(xx, mu, sigma, xi):
    """GEV return period"""
    #return 1/( 1 - gev_cdf(xx, mu, sigma, xi) )
    return 1/gev_exceedance_prob(xx, mu, sigma, xi)
def plot_return_period(mu=0, sigma=1, xi=0, lower=1/.99, upper=1000, ax=None, **kws):
    if ax is None:
        fig,ax = plt.subplots()
    xx = np.linspace( gev.ppf(1-1/lower, c=-xi, loc=mu, scale=sigma), 
        gev.ppf(1-1/upper, c=-xi, loc=mu, scale=sigma), 1000)
    rp = gev_return_period(xx, mu, sigma, xi) 
    ax.plot(rp, xx, **kws)
    ax.set_xscale('log')

#return value from return period
def gev_return_period_inverse(rp, mu, sigma, xi):
    """GEV return period inverse (get return value from return period)"""
    return gev.ppf(1-1/rp, c=-xi, loc=mu, scale=sigma) 

def validate_gev(mu=0, sigma=1, xi=0, kind='pdf', ax=None):
    if ax is None:
        fig,ax = plt.subplots()
    if kind == 'cdf':
        plot_cdf(mu, sigma, xi, ax=ax, user_defined=False, label=f'{mu=};{sigma=};{xi=}: genextreme')
        plot_cdf(mu, sigma, xi, ax=ax, user_defined=True, ls='--', label=f'{mu=};{sigma=};{xi=}: WY')
        ax.set_ylabel('GEV cdf')
    elif kind == 'pdf':
        plot_pdf(mu, sigma, xi, ax=ax, user_defined=False, label=f'{mu=};{sigma=};{xi=}: genextreme')
        plot_pdf(mu, sigma, xi, ax=ax, user_defined=True, ls='--', label=f'{mu=};{sigma=};{xi=}: WY')
        ax.set_ylabel('GEV pdf')
        
    ax.legend()

if __name__ == '__main__':
    from wyconfig import * #my plot settings
    plt.close('all')
    fig, axes = plt.subplots(2, 1, figsize=(6,5))
    mu, sigma = 0, 1
    xis = (-0.5, 0, 0.5)
    ax = axes[0]
    for xi in xis:
        validate_gev(mu, sigma, xi, 'cdf', ax)
    ax.set_xlim(None, 15)

    ax = axes[1]
    for xi in xis:
        validate_gev(mu, sigma, xi, 'pdf', ax)
    ax.set_xlim(None, 15)

    #savefig
    if 'savefig' in sys.argv:
        figname = __file__.replace('.py', f'.png')
        wysavefig(figname)
    tt.check(f'**Done**')
    plt.show()
    
