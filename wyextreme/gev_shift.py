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
from .shared import plot_smp_return_period, plot_return_period, gev_return_period, gev_return_period_inverse
#
if __name__ == '__main__':
    tt.check('end import')
#
#start from here
#negative log likelihood
def _negLogLikelihood(params, data, datacv, xi_bounds=None):
    """GEV shift negative log likelihood.
        params: (mu0, sigma, xi, alpha)
        data: sample(s)
        datacv: co-variate
    Ref: https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution
    """
    if xi_bounds is None or xi_bounds==(None, None):
        xi_bounds = (-np.inf, np.inf)
    mu0, sigma, xi, alpha = params
    mu = mu0 + alpha*datacv 
    s = (data - mu)/sigma
    if xi < xi_bounds[0] or xi > xi_bounds[1]:#effectively set xi bounds
            return np.inf
    elif xi == 0:
        return data.size*log(sigma) + np.sum(s + exp(-s))
    else:
        return -np.sum( log(xi*s>-1) ) + data.size*log(sigma) + np.sum( (1+1/xi)*log(1+xi*s) ) + np.sum( (1+xi*s)**(-1/xi) )
def fit(data, datacv, **kws):
    """GEV fit using maximum likelihood"""
    method = kws.pop('method', 'Nelder-Mead')
    #method = ['L-BFGS-B', 'TNC', 'SLSQP', 'Powell', 'COBYLA'][-1] #these methods generally don't work
    xi_bounds = kws.pop('xi_bounds', (None, None))
    #xi_bounds = kws.pop('xi_bounds', (-0.3, 0.3))
    bounds = ( (None, None),  (0, None),  xi_bounds, (None, None) )
    alpha_guess = np.corrcoef(data, datacv)[0,1]*data.std().item()/datacv.std().item()
    mu0_guess = data.mean().item() - alpha_guess*datacv.mean().item()
    sigma_guess = (data - alpha_guess*datacv).std().item()
    xi_guess = -0.1
    x0_default = (mu0_guess, sigma_guess, xi_guess, alpha_guess)
    x0 = kws.pop('x0', x0_default)
    #print('initial params:'.ljust(16), f'{mu0_guess}; {sigma_guess=}; {xi_guess=}; {alpha_guess=}')

    r = minimize(_negLogLikelihood, x0, args=(data,datacv, xi_bounds), method=method, bounds=bounds, **kws)
    if not r.success:
        print(f'{r = }')
        print('[failed]:', r.message)
    return r
def plot_fit(data, datacv, cv_level=None, fit_result=None, ax=None, fit_kws=None, **kws):
    if cv_level is None:#co-variate level, e.g. value in some specific year
        cv_level = ('max co-variate', datacv.max().item())
    if ax is None:
        fig,ax = plt.subplots()
    if fit_kws is None:
        fit_kws = {}
    upper = kws.pop('upper', data.size*100)
    label = kws.pop('label', cv_level[0])
    #fit by user defined likelihood function
    if fit_result is None: #do the fit
        r = fit(data, datacv, **fit_kws)
    else: #already done the fit: use the result directly
        r = fit_result
    if r.success:
        mu0, sigma, xi, alpha = r.x
        print('wy fit params:'.ljust(16), f'{mu0=:.4g}; {sigma=:.4g}; {xi=:.4g}; {alpha=:.4g}') 
        #empirical/fit return periods
        plot_smp_return_period(data-alpha*datacv+alpha*cv_level[1], ax=ax, **kws)
        plot_return_period(mu0+alpha*cv_level[1], sigma, xi, upper=upper, ax=ax, label=label, **kws)

    ax.set_xlabel('return period')
    try:
        ax.set_ylabel(data.name)
    except:
        pass
    print()

    return r

def fit_bootstrap(data, datacv, nmc=100, mc_seed=0, **kws):
    """GEV shift fit bootstrap. 
        data: input array-like data to fit
        datacv: covariate
        nmc: size of Monte Carlo samples
        seed: np.random seed (default is 0)
    """
    if isinstance(data, xr.DataArray):
        data = data.values
    if isinstance(datacv, xr.DataArray):
        datacv = datacv.values
    rng = np.random.default_rng(mc_seed)
    #xxmc = np.random.choice(data, size=(nmc, data.size))
    mci = rng.choice(data.size, size=(nmc, data.size))
    data_mc = data[mci]
    datacv_mc = datacv[mci]
    params = np.zeros(shape=(nmc, 4)) + np.nan
    #bootstrap
    for ii in tqdm(range(nmc)):
        r = fit(data_mc[ii,:], datacv_mc[ii,:], **kws)
        if r.success:
            params[ii,:] = r.x
        else:
            print(f'mc = {ii};', r.message)
    #with mp.Pool(processes=min(40, mp.cpu_count(), nmc)) as p:
    #    p.map(func_bs, range(nmc))
    mu0 = xr.DataArray(params[:,0], dims='mc')
    sigma = xr.DataArray(params[:,1], dims='mc')
    xi = xr.DataArray(params[:,2], dims='mc')
    alpha = xr.DataArray(params[:,3], dims='mc')

    ds = xr.Dataset(dict(mu0=mu0, sigma=sigma, xi=xi, alpha=alpha))
    #best fit
    r = fit(data, datacv, **kws)
    ds['mu0_best'] = xr.DataArray(r.x[0])
    ds['sigma_best'] = xr.DataArray(r.x[1])
    ds['xi_best'] = xr.DataArray(r.x[2])
    ds['alpha_best'] = xr.DataArray(r.x[3])

    return ds 
def plot_fit_bootstrap(data, datacv, cv_level=None, bsfit=None, nmc=100, mc_seed=0, ci=95, upper_rp=None, ax=None, fit_kws=None, **kws):
    if cv_level is None:
        cv_level = ('max co-variate', datacv.max().item())
    if upper_rp is None:
        upper_rp = data.size*100
    if ax is None:
        fig,ax = plt.subplots()
    if fit_kws is None:
        fit_kws = {}
    #direct fit plot
    r = plot_fit(data, datacv, cv_level=cv_level, ax=ax, fit_kws=fit_kws, upper=upper_rp, **kws)
    mu0, sigma, xi, alpha = r.x
    #bootstrap
    if bsfit is None: #do the bootstrap fit
        ds = fit_bootstrap(data, datacv, nmc=nmc, mc_seed=mc_seed, **fit_kws)
    else: #already done the bootstrap fit: use the result directly
        ds = bsfit
    ci_bounds = [(1-ci/100)/2, (1+ci/100)/2]
    for ii,daname in enumerate(('mu0', 'sigma', 'xi', 'alpha')):
        q = ds[daname].quantile(ci_bounds, dim='mc')
        print(f'{daname} and {ci}% CI:'.rjust(20), f'{r.x[ii]:.4g}({q[0].item():.4g}, {q[1].item():.4g})')
    print()
    #confidence interval of the return value
    mu_shift = mu0 + alpha*cv_level[1]
    lower, upper = 1, np.log10(upper_rp) #return period bounds
    rp = np.logspace(lower, upper, 100)
    yy = [gev_return_period_inverse(rp, mu0+alpha*cv_level[1], sigma, xi) 
        for mu0,sigma,xi,alpha in zip(ds.mu0.values, ds.sigma.values, ds.xi.values, ds.alpha.values)]
    yy = xr.DataArray(yy, dims=('mc', 'rp')).assign_coords(rp=rp)
    yy.quantile(ci_bounds, dim='mc').plot(x='rp', ls='--', lw=1, hue='quantile', add_legend=False, **kws)

    ax.set_xlabel('return period')
    ax.set_ylabel('return value')

    return ds
def plot_covariate(data, datacv, fit_result=None, ax=None, fit_kws=None, **kws):
    if ax is None:
        fig,ax = plt.subplots()
    if fit_kws is None:
        fit_kws = {}
    ax.plot(datacv, data, ls='none', marker='o', fillstyle='none', alpha=0.5, **kws)
    if fit_result is None:
        r = fit(data, datacv, **fit_kws)
    else:
        r = fit_result
    if r.success:
        mu0,sigma,xi,alpha = r.x
        print('wy fit params:'.ljust(16), f'{mu0=:.4g}; {sigma=:.4g}; {xi=:.4g}; {alpha=:.4g}') 
        ax.axline((datacv[0].item(), mu0+alpha*datacv[0].item()), slope=alpha, **kws)
        ax.axline((datacv[0].item(), mu0+alpha*datacv[0].item()+gev_return_period_inverse(6, 0, sigma, xi)), slope=alpha, lw=1, ls='--', **kws)
        ax.axline((datacv[0].item(), mu0+alpha*datacv[0].item()+gev_return_period_inverse(40, 0, sigma, xi)), slope=alpha, lw=1, ls='--',  **kws)
    ax.set_xlabel('co-variate')
    ax.set_ylabel('return value')

def plot_mu_ci(data, datacv, cv0=None, bsfit=None, nmc=100, mc_seed=0, ci=95, fit_kws=None, ax=None, **kws):
    """add confidence inverval information of mu to the figure generated by the plot_covariate.
        cv0: co-variate value at which the confidence interval info will be added.
    """
    if cv0 is None:
        cv0 = np.array(datacv)[-1]
    if fit_kws is None:
        fit_kws = {}
    if bsfit is None:
        ds = fit_bootstrap(data, datacv, nmc=nmc, mc_seed=mc_seed, **fit_kws)
    else:
        ds = bsfit
    if ax is None:
        ax = plt.gca()
    capsize = kws.pop('capsize', 3)

    mu_cv0 = ds['mu0_best'] + ds['alpha_best'] * cv0
    ci_bound = (1 - ci/100)/2, (1 + ci/100)/2
    mu_cv0_ci = ( ds['mu0'] + ds['alpha'] * cv0 ).quantile(ci_bound, dim='mc')

    x = cv0
    y = mu_cv0
    yerr = np.abs(mu_cv0_ci - mu_cv0)
    yerr = np.array(yerr).reshape(2,1)
    ax.errorbar(x, y, yerr, capsize=capsize, **kws)

    print(f'cv0 = {np.array(cv0)}; mu_cv0 = {np.array(mu_cv0)}; mu_cv0_ci = {np.array(mu_cv0_ci)}')

def fit_all(data, datacv, cv_levels=None, nmc=100, mc_seed=0, ci=95, upper_rp=None, fit_kws=None):
    if cv_levels is None:
        cv_levels = [('min co-variate', datacv.min().item()), ('max co-variate', datacv.max().item())]
    if fit_kws is None:
        fit_kws = {}

    plot_covariate(data, datacv, color='k')
    ds = fit_bootstrap(data, datacv, nmc=nmc, mc_seed=mc_seed, **fit_kws)
    plot_mu_ci(data, datacv, cv0=cv_levels[0][1], bsfit=ds, color='C0')
    plot_mu_ci(data, datacv, cv0=cv_levels[1][1], bsfit=ds, color='C1')
    
    fig,ax = plt.subplots()
    plot_fit_bootstrap(data, datacv, cv_levels[0], bsfit=ds, ci=ci, upper_rp=upper_rp, ax=ax, fit_kws=fit_kws, color='C0')
    if len(cv_levels) > 1:
        for ii,cv_level in enumerate(cv_levels[1:], start=1):
            plot_fit_bootstrap(data, datacv, cv_levels[ii], bsfit=ds, ci=ci, upper_rp=upper_rp, ax=ax, fit_kws=fit_kws, color=f'C{ii}')
    ax.legend()
    
    return ds
    

def makedata(mu0=None, sigma=None, xi=None, alpha=None, datacv=None, nsmp=100, seed=1, ofile=None):
    #specify params
    rng = np.random.default_rng()
    if mu0 is None:
        mu0 = rng.uniform(10, 20)
    if sigma is None:
        sigma = rng.uniform(0, 10)
    if xi is None:
        xi = rng.uniform(-1, 1)
    if alpha is None:
        alpha = rng.uniform(-4, 4)
    true_values = mu0,sigma,xi,alpha
    note = 'true params:'.ljust(16) + f'{mu0=:.4g}; {sigma=:.4g}; {xi=:.4g}; {alpha=:.4g}'
    print(note)
    #specify co-variate
    if datacv is None:
        datacv = np.linspace(0, 2, nsmp)
    #generate data
    rng = np.random.default_rng(seed)#seed to generate genextreme random variables
    genextreme.random_state = rng
    data = genextreme.rvs(-xi, loc=mu0+alpha*datacv, scale=sigma)
    #show data
    plt.plot(datacv, data, ls='none', marker='o', fillstyle='none')

    if ofile is not None:# save data
        ofile_data = 'data_' + ofile
        xr.DataArray(data, dims='year').assign_coords(year=range(1900,1900+data.size)).assign_attrs(note=note).to_dataset(name='mydata').to_netcdf(ofile_data)
        print('[saved]:', ofile_data)
        ofile_datacv = 'datacv_' + ofile
        xr.DataArray(datacv, dims='year').assign_coords(year=range(1900, 1900+datacv.size)).to_dataset(name='mycovariate').to_netcdf(ofile_datacv)
        print('[saved]:', ofile_datacv)
    return data, datacv

        
def test(mu0=None, sigma=None, xi=None, alpha=None, datacv=None, seed=1, nmc=100, mc_seed=0, ci=95, nsmp=100):
    #specify params
    rng = np.random.default_rng()
    if mu0 is None:
        mu0 = rng.uniform(10, 20)
    if sigma is None:
        sigma = rng.uniform(0, 10)
    if xi is None:
        xi = rng.uniform(-1, 1)
    if alpha is None:
        alpha = rng.uniform(-4, 4)
    true_values = mu0,sigma,xi,alpha
    print('true params:'.ljust(16), f'{mu0=:.4g}; {sigma=:.4g}; {xi=:.4g}; {alpha=:.4g}')
    #specify co-variate
    if datacv is None:
        datacv = np.linspace(0, 2, nsmp)
    #generate data
    rng = np.random.default_rng(seed)#seed to generate genextreme random variables
    genextreme.random_state = rng
    data = genextreme.rvs(-xi, loc=mu0+alpha*datacv, scale=sigma)
    #validate
    #co-variate plot
    fig, ax = plt.subplots()
    plot_covariate(data, datacv, ax=ax, color='k')
    #fit_bootstrap plot
    fig,ax = plt.subplots()
    ds = plot_fit_bootstrap(data, datacv, ('initial co-variate', datacv[0]), nmc=nmc, mc_seed=mc_seed, ci=ci, ax=ax, color='C0')#, upper_rp=data.size*40) 
    plot_fit_bootstrap(data, datacv, ('final co-variate', datacv[-1]), bsfit=ds, ax=ax, color='C1')#, upper_rp=data.size*40) 
    ax.legend()
    #fit summary
    r = fit(data, datacv) 
    mu0,sigma,xi,alpha = r.x
    ci_bounds = [(1-ci/100)/2, (1+ci/100)/2]
    s = ''
    danames = ('mu0', 'sigma', 'xi', 'alpha')
    pnames = ('$\\mu_0$:', '$\\sigma$:', '$\\xi$:', '$\\alpha$:')
    for ii,(daname,pname, tv) in enumerate(zip(danames, pnames, true_values)):
        q = ds[daname].quantile(ci_bounds, dim='mc')
        s += pname + f' {r.x[ii]:.4g}(true:{tv:.4g}); {ci}% CI: ({q[0].item():.4g}, {q[1].item():.4g})\n'
    #ax.text(1, 1, s, transform=ax.transAxes, ha='left', va='top', fontsize='small')
    ax.text(1, 0, s, transform=ax.transAxes, ha='right', va='bottom', fontsize='small', alpha=0.5)
    print(s)

if __name__ == '__main__':
    from wyconfig import * #my plot settings
    plt.close('all')
    if len(sys.argv)<=1:
        test()
    elif len(sys.argv)>1 and sys.argv[1]=='test': #e.g. python -m wyextreme.gev_shift test xi=-0.1
        kws = dict(mu0=None, sigma=None, xi=None, alpha=None, seed=1, nmc=100, mc_seed=0, ci=95, nsmp=100)
        if len(sys.argv)>2:
            for s in sys.argv[2:]:
                key,v = s.split('=')
                v = int(v) if key in ('seed', 'nmc', 'mc_seed', 'nsmp') else float(v)
                if key in kws: kws[key] = v
        test(**kws)
    elif len(sys.argv)>1 and sys.argv[1]=='makedata': #e.g. python -m wyextreme.gev_shift make data
        kws = dict(mu0=None, sigma=None, xi=None, alpha=None, nsmp=100, seed=1, ofile=None)
        if len(sys.argv)>2:
            for s in sys.argv[2:]:
                key,v = s.split('=')
                if key in ('seed', 'nsm'):
                    v = int(v)
                elif key in ('ofile',):
                    pass
                else:
                    v = float(v)
                if key in kws: kws[key] = v
        makedata(**kws)
    elif len(sys.argv)>2: # two input data files to compare
        kws = dict(cv_levels=None, nmc=100, mc_seed=0, ci=95, upper_rp=None)
        da0 = xr.open_dataarray(sys.argv[1])
        da1 = xr.open_dataarray(sys.argv[2])
        if 'en' in da0.dims:
            da0 = da0.stack(s=['en', 'year'])
        if 'en' in da1.dims:
            da1 = da1.stack(s=['en', 'year'])
        fit_all(da0, da1, **kws)
        """
        fig, ax = plt.subplots()
        plot_covariate(da0, da1, ax=ax, color='k')
        fig,ax = plt.subplots()
        ds = plot_fit_bootstrap(da0, da1, ('min co-variate', da1.min().item()), ax=ax, color='C0')#, upper_rp=da0.size*40) 
        plot_fit_bootstrap(da0, da1, ('max co-variate', da1.max().item()), bsfit=ds, ax=ax, color='C1')#, upper_rp=da0.size*40) 
        """

    #savefig
    if 'savefig' in sys.argv:
        figname = __file__.replace('.py', f'.png')
        wysavefig(figname)
    tt.check(f'**Done**')
    plt.show()
    
