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
from scipy.stats import genextreme
from numpy import exp, log
from tqdm import tqdm
from .shared import plot_smp_return_period, plot_return_period
#
if __name__ == '__main__':
    tt.check('end import')
#
#start from here
#negative log likelihood
def negLogLikelihood(params, data):
    """GEV negative log likelihood.
        params: (mu, sigma, xi)
        xx: sample(s)
    Ref: https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution
    """
    mu, sigma, xi = params
    s = (data - mu)/sigma
    if xi == 0:
        return data.size*log(sigma) + np.sum(s + exp(-s))
    else:
        return -np.sum( log(xi*s>-1) ) + data.size*log(sigma) + np.sum( (1+1/xi)*log(1+xi*s) ) + np.sum( (1+xi*s)**(-1/xi) )
def fit(data, **kws):
    """GEV fit using maximum likelihood"""
    method = kws.pop('method', 'Nelder-Mead')
    xi_bounds = kws.pop('xi_bounds', (None, None))
    bounds = ( (None,None),  (0,None),  xi_bounds )
    x0_default = (data.mean(), data.std(), 0)
    x0 = kws.pop('x0', x0_default)

    return minimize(negLogLikelihood, x0, args=(data,), method=method, bounds=bounds, **kws)
def plot_fit(data, fit_kws=None, ax=None, **kws):
    if ax is None:
        fig,ax = plt.subplots()
    if fit_kws is None:
        fit_kws = {}
    upper = kws.pop('upper', data.size*10)
    #empirical return periods
    plot_smp_return_period(data, ax=ax, **kws)
    #fit by scipy.stats.genextreme
    c, mu, sigma = genextreme.fit(data)
    xi = -c
    print('genextreme fit:'.ljust(16), f'{mu=:.4g}; {sigma=:.4g};  {xi=:.4g}')
    #fit by user defined likelihood function
    r = fit(data, **fit_kws)
    if r.success:
        mu, sigma, xi = r.x
        print('wy fit:'.ljust(16), f'{mu=:.4g}; {sigma=:.4g}; {xi=:.4g}')
        plot_return_period(mu, sigma, xi, upper=upper, ax=ax, **kws)
    else:
        print(f'{r = }')
        print('[failed]:', r.message)
    ax.set_xlabel('return period')
    try:
        ax.set_ylabel(data.name)
    except:
        pass
    print()
    
    return r

def fit_bootstrap(data, nmc=100, mc_seed=None, **kws):
    """GEV fit bootstrap. 
        data: input array-like data to fit
        nmc: size of Monte Carlo samples
        seed: np.random seed (default is 0)
    """
    rng = np.random.default_rng(mc_seed)
    datamc = np.random.choice(data, size=(nmc, data.size))
    params = np.zeros(shape=(nmc, 3)) + np.nan
     
    for ii in tqdm(range(nmc)):
        r = fit(datamc[ii,:], **kws)
        if r.success:
            params[ii,:] = r.x
        else:
            print(f'mc = {ii};', r.message)
    #with mp.Pool(processes=min(40, mp.cpu_count(), nmc)) as p:
    #    p.map(func_bs, range(nmc))
    mu = xr.DataArray(params[:,0], dims='mc')
    sigma = xr.DataArray(params[:,1], dims='mc')
    xi = xr.DataArray(params[:,2], dims='mc')

    ds = xr.Dataset(dict(mu=mu, sigma=sigma, xi=xi))
    return ds 
def plot_fit_bootstrap(data, bsfit=None, nmc=100, mc_seed=None, ci=95, upper_rp=None, ax=None, fit_kws=None, **kws):
    if upper_rp is None:
        upper_rp = data.size*10
    if ax is None:
        fig,ax = plt.subplots()
    if fit_kws is None:
        fit_kws = {}
    #direct fit plot
    r = plot_fit(data, fit_kws=fit_kws, ax=ax, upper=upper_rp, **kws)
    mu, sigma, xi = r.x
    #bootstrap
    if bsfit is None: #do the bootstrap fit
        ds = fit_bootstrap(data, nmc=nmc, mc_seed=mc_seed, **fit_kws)
    else: #already done the bootstrap fit: use the result directly
        ds = bsfit
    ci_bounds = [(1-ci/100)/2, (1+ci/100)/2]
    for ii,daname in enumerate(('mu', 'sigma', 'xi')):
        q = ds[daname].quantile(ci_bounds, dim='mc')
        print(f'{daname} and {ci}% CI:'.rjust(20), f'{r.x[ii]:.4g}({q[0].item():.4g}, {q[1].item():.4g})')
    print()
    #confidence interval of the return value
    lower, upper = 1, np.log10(upper_rp) #return period bounds
    rp = np.logspace(lower, upper, 100)
    yy = [gev_return_period_inverse(rp, mu, sigma, xi)
        for mu,sigma,xi in zip(ds.mu.values, ds.sigma.values, ds.xi.values)]
    yy = xr.DataArray(yy, dims=('mc', 'rp')).assign_coords(rp=rp)
    yy.quantile(ci_bounds, dim='mc').plot(x='rp', ls='--', lw=1, hue='quantile', add_legend=False, **kws)

    ax.set_xlabel('return period')
    ax.set_ylabel('return value')

    return ds
def test(mu=None, sigma=None, xi=None, nmc=100, mc_seed=None, ci=95,nsmp=100):
    #specify params
    rng = np.random.default_rng()
    if mu is None:
        mu = rng.uniform(-10, 10)
    if sigma is None:
        sigma = rng.uniform(0, 10)
    if xi is None:
        xi = rng.uniform(-1, 1)
    true_values = mu,sigma,xi
    print('true params:'.ljust(16), f'{mu=:.4g}; {sigma=:.4g}; {xi=:.4g}')
    #generate data
    data = genextreme.rvs(-xi, loc=mu, scale=sigma, size=nsmp)
    #validate
    #fit_bootstrap plot
    fig,ax = plt.subplots()
    ds = plot_fit_bootstrap(data, nmc=nmc, mc_seed=mc_seed, ci=ci, ax=ax, color='C0')#, upper_rp=data.size*40)
    #fit summary
    r = fit(data)
    mu,sigma,xi = r.x
    ci_bounds = [(1-ci/100)/2, (1+ci/100)/2]
    s = ''
    danames = ('mu', 'sigma', 'xi')
    pnames = ('$\\mu$:', '$\\sigma$:', '$\\xi$:')
    for ii,(daname,pname, tv) in enumerate(zip(danames, pnames, true_values)):
        q = ds[daname].quantile(ci_bounds, dim='mc')
        s += pname + f' {r.x[ii]:.4g}(true:{tv:.4g}); {ci}% CI: ({q[0].item():.4g}, {q[1].item():.4g})\n'
    ax.text(1, 0, s, transform=ax.transAxes, ha='right', va='bottom', fontsize='small', alpha=0.5)
    print(s) 

if __name__ == '__main__':
    from wyconfig import * #my plot settings
    plt.close('all')
    if len(sys.argv)<=1:
        test()
    elif len(sys.argv)>1 and sys.argv[1]=='test': #e.g. python -m wyextreme.gev test xi=-0.1
        kws = dict(mu=None, sigma=None, xi=None, nmc=100, mc_seed=None, ci=95, nsmp=100)
        if len(sys.argv)>2:
            for s in sys.argv[2:]:
                key,v = s.split('=')
                v = int(v) if key in ('nmc', 'mc_seed', 'nsmp') else float(v)
                if key in kws: kws[key] = v
        test(**kws)
    elif len(sys.argv)>2: # two input data files to compare
        da0 = xr.open_dataarray(sys.argv[1])
        da1 = xr.open_dataarray(sys.argv[2])
        fig, ax = plt.subplots()
        plot_fit_bootstrap(da0, ax=ax, color='C0')
        plot_fit_bootstrap(da1, ax=ax, color='C1')
    elif len(sys.argv)>1:#read input data file and do the analysis; otherwise
        ifile = sys.argv[1]
        da = xr.open_dataarray(ifile)
        plot_fit(da)
         

    #savefig
    if 'savefig' in sys.argv:
        figname = __file__.replace('.py', f'.png')
        wysavefig(figname)
    tt.check(f'**Done**')
    plt.show()
    
