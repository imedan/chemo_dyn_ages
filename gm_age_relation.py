import numpy as np
from astropy.io import fits
import pandas as pd
from astropy.table import Table, join
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pylab as plt
import os
plt.style.use('%s/mystyle.mplstyle' % os.environ['MPL_STYLES'])

import emcee
from multiprocessing import Pool
import random

from sklearn.neighbors import KernelDensity
from KDEpy import TreeKDE
from sklearn.mixture import GaussianMixture

from p_tqdm import p_map
import os
import time
from galpy.actionAngle import actionAngleStaeckel
from galpy.actionAngle import estimateDeltaStaeckel
from galpy.potential import MWPotential2014
from galpy.orbit import Orbit

from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter, median_filter
from matplotlib.patches import Ellipse, Polygon

import matplotlib.path as mpltPath

from sklearn.decomposition import PCA
from sklearn import preprocessing
from functools import partial

from tqdm import trange


def uvw(ra, de, pmra, pmde, dist, rv):
    """
    Calculate UVW velocity
    """
    DEG = 57.2957764
    x = np.cos(de / DEG) * np.cos(ra / DEG)
    y = np.cos(de / DEG) * np.sin(ra / DEG)
    z=np.sin(de/DEG)

    gx = -0.054877 * x - 0.873437 * y - 0.483835 * z
    gy = +0.494109 * x - 0.444831 * y + 0.746982 * z
    gz = -0.867667 * x - 0.198075 * y + 0.455985 * z

    b = DEG * np.arcsin(gz)
    l = np.zeros(len(b))
    for i in range(len(l)):
        if gy[i] > 0.:
            side = gx[i] / np.cos(b[i] / DEG)
            if side < 1.:
                l[i] = 180.
            elif side > 1.:
                l[i] = 0.
            else:
                l[i] = DEG * np.arccos(side)
        else:
            side = gx[i] / np.cos(b[i] / DEG)
            if side < -1.:
                l[i] = 180.
            elif side > 1.:
                l[i] = 0.
            else:
                l[i] = 360. - DEG * np.arccos(side)
      
    pmx = (-1. * pmra * np.sin(ra / DEG) -
           pmde * np.sin(de / DEG) * np.cos(ra / DEG) +
           rv / 4.74 / dist * np.cos(ra / DEG) * np.cos(de / DEG))
    pmy = (pmra * np.cos(ra / DEG) -
           pmde * np.sin(de / DEG) * np.sin(ra / DEG) +
           rv / 4.74 / dist * np.sin(ra / DEG) * np.cos(de / DEG))
    pmz = pmde * np.cos(de / DEG) + rv / 4.74 / dist * np.sin(de / DEG)

    gpmx = -0.054877 * pmx - 0.873437 * pmy - 0.483835 * pmz
    gpmy = +0.494109 * pmx - 0.444831 * pmy + 0.746982 * pmz
    gpmz = -0.867667 * pmx - 0.198075 * pmy + 0.455985 * pmz

    gx = dist * gx
    gy = dist * gy
    gz = dist * gz

    gu = 4.74 * dist * gpmx
    gv = 4.74 * dist * gpmy
    gw = 4.74 * dist * gpmz
    return gx, gy, gz, gu, gv, gw


def hist_2d_bootstrap(x, y, binx, biny, N):
    """
    bootstrap 2d histogram
    """
    n, binsx, binsy = np.histogram2d(x, y, bins=[binx, biny], density=True)
    ns = np.zeros((N, len(n.ravel())))
    for i in trange(N):
        idx = np.random.randint(len(x),size=len(x))
        n, binsx, binsy = np.histogram2d(x[idx], y[idx], bins=[binx, biny], density=True)
        ns[i, :] = n.T.ravel()
    return np.mean(ns, axis=0), np.std(ns, axis=0)


def est_sigma4(x, theta, ts):    
    y_mod = np.zeros(len(x[0]))
    t_mids = np.array([(ts[i] + ts[i + 1]) / 2 for i in range(len(ts) - 1)])
    for i in range(len(theta)):
        y_mod += theta[i] * x[i + 1]            
    return y_mod


def log_likelihood(theta, x, y, yerr, ts):
    model = est_sigma4(x, theta, ts)
    sigma2 = yerr ** 2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))


def log_prior(theta):
    if np.all(theta > 0) and round(np.sum(theta), 4) == 1.:
        return 0.0
    return -np.inf


def log_probability(theta, x, y, yerr, ts):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr, ts)


def mcmc_GM_fit(MH_norm, W, gms, ts):
    """
    find the probably age distirbution based on GMM fit
    from GALAH and W, [M/H] of data
    """
    binx = np.arange(-1.5, 0.6, 0.1) + 0.5
    binx_mid = np.array([(binx[i] + binx[i + 1]) / 2 for i in range(len(binx) - 1)])
    biny = np.arange(-100, 110, 10) / 100
    biny_mid = np.array([(biny[i] + biny[i + 1]) / 2 for i in range(len(biny) - 1)])

    X, Y = np.meshgrid(binx, biny)
    Xmid, Ymid = np.meshgrid(binx_mid, biny_mid)

    y, yerr = hist_2d_bootstrap(MH_norm + 0.5,
                                W / 100,
                                binx, biny, 1000)

    ev_y = eval("y > 0")

    data_score = np.column_stack((Xmid.ravel()[ev_y], Ymid.ravel()[ev_y]))

    y_comps = []
    for i in range(len(gms)):
        y_comps.append(np.exp(gms[i].score_samples(data_score)))

    pos = np.random.rand(100, len(ts) - 1)

    for i in range(len(pos[:,0])):
        pos[i,:] = pos[i,:] / np.sum(pos[i,:])

    pos += 1e-6 * np.random.randn(100, len(ts) - 1)

    nwalkers, ndim = pos.shape

    with Pool(4) as pool:
        sampler = emcee.EnsembleSampler(nwalkers,
                                        ndim,
                                        log_probability,
                                        args=([Xmid.ravel()[ev_y]] + y_comps,
                                              y[ev_y], yerr[ev_y], ts),
                                        pool=pool)
        sampler.run_mcmc(pos, 10000, progress=True)

    tau = sampler.get_autocorr_time(quiet=True)
    flat_samples = sampler.get_chain(discard=int(3 * max(tau)), thin=int(max(tau) / 2), flat=True)
    return sampler, flat_samples


def bootstrap_mean(x, N):
    if len(x) > 50:
        means = np.zeros(N)
        for i in range(N):
            means[i] = np.nanmean(x[np.random.randint(len(x), size=len(x))])
        return np.mean(means), np.std(means)
    else:
        return np.nan, np.nan
    
def bootstrap_median(x ,N):
    if len(x) > 50:
        means = np.zeros(N)
        for i in range(N):
            means[i] = np.nanmedian(x[np.random.randint(len(x), size=len(x))])
        return np.mean(means), np.std(means)
    else:
        return np.nan, np.nan
    
def bootstrap_std(x, N):
    if len(x) > 50:
        means = np.zeros(N)
        for i in range(N):
            means[i] = np.nanstd(x[np.random.randint(len(x), size=len(x))])
        return np.mean(means), np.std(means)
    else:
        return np.nan, np.nan


def compare_metals(MH_spec, MH_spec_err, MH_photo, ax1, ax2, survey_name):
    """
    compare photometric metallicities to spec.
    and make the comparison plots
    """
    ax1.scatter(MH_photo, MH_spec, c='k', marker='.', label='N = %d' % len(MH_photo))
    ax1.set_xlim((-1, 0.5))
    ax1.set_ylim((-1, 0.5))
    ax1.plot([-1, 0.5], [-1, 0.5], '--', c='r', label='_nolabel_')
    ax1.set_xlabel(r'$[M/H]_{Photo}$')
    ax1.set_ylabel(r'$%s$' % survey_name)
    ax1.legend()
    ax1.grid()


    t2 = np.pi / 4
    rotmat2 = np.array([[np.cos(t2), -np.sin(t2)], 
                        [np.sin(t2),  np.cos(t2)]])

    m2 = MH_spec - MH_photo

    std_tot, std_tot_err = bootstrap_std(m2, 1000)
    mean_spec, mean_spec_err = bootstrap_mean(np.array(MH_spec_err), 1000)

    sys_err = np.sqrt(std_tot_err ** 2 * (std_tot / np.sqrt(std_tot ** 2 - mean_spec ** 2)) ** 2 + 
                      mean_spec_err ** 2 * (mean_spec / np.sqrt(std_tot ** 2 - mean_spec ** 2)) ** 2)

    Nbin, _, _ = ax2.hist(m2, bins=np.linspace(-.4, .4, 50), label=r'$\sigma_{tot}=$'+'%.4f+/-%.4f\n' % (std_tot,std_tot_err)+
                          r'$\sigma_{sys}=$'+'%.4f+/-%.4f' % (np.sqrt(std_tot ** 2 - mean_spec ** 2), sys_err))
    ax2.grid()
    median = bootstrap_median(m2, 1000)
    ax2.axvline(median[0], c='r', linestyle='--', label='Median = %.4f+/-%.4f' % median)
    ax2.legend()
    ax2.set_xlabel(r'$%s - [M/H]_{Photo}$' % survey_name)
    ax2.set_ylabel('N')
    ax2.set_ylim((0, max(Nbin) + 0.5 * max(Nbin)))


def compare_all_spec_surveys(galah_file, apogee_16_file, apogee_14_file,
                             KM_metals, plot_save):
    """
    compare spec metallicities of all surveys
    """
    hdu = fits.open(galah_file)

    galah = Table(hdu[1].data).to_pandas()

    galah = galah.drop_duplicates(subset='source_id', keep='first', ignore_index=True)
    galah = galah.rename(columns={"source_id": "ID"})
    galah = galah.merge(KM_metals[['M_H', 'ID']], on='ID', how='inner')

    hdu=fits.open(apogee_16_file)

    apogee = Table(hdu[1].data)[['APOGEE_ID', 'GAIA_SOURCE_ID', 'M_H', 'M_H_ERR']].to_pandas()
    apogee = apogee.drop_duplicates(subset='GAIA_SOURCE_ID',
                                    keep='first', ignore_index=True)
    apogee = apogee[abs(apogee['M_H'])<1000.].reset_index()
    apogee = apogee.rename(columns={"GAIA_SOURCE_ID": "ID"})
    apogee = apogee.rename(columns={"M_H": "M_H_spec"})
    apogee = apogee.merge(KM_metals[['M_H', 'ID']], on='ID', how='inner')

    hdu=fits.open(apogee_14_file)

    apdr14 = Table(hdu[1].data)[['APOGEE_ID', 'M_H', 'M_H_ERR']].to_pandas()
    apdr14 = apdr14.drop_duplicates(subset='APOGEE_ID',
                                    keep='first', ignore_index=True)
    apdr14 = apdr14[abs(apdr14['M_H'])<1000.].reset_index()
    apdr14 = apdr14.rename(columns={"M_H": "M_H_spec"})
    apdr14 = apdr14.merge(apogee[['ID', 'APOGEE_ID']], on='APOGEE_ID', how='inner')
    apdr14 = apdr14.merge(KM_metals[['M_H', 'ID']], on='ID', how='inner')

    f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(20, 30))

    compare_metals(apdr14['M_H_spec'], apdr14['M_H_ERR'],
                   apdr14['M_H'], ax1, ax2, '[M/H]_{APOGEE_{DR14}}')
    compare_metals(apogee['M_H_spec'], apogee['M_H_ERR'],
                   apogee['M_H'], ax3, ax4, '[M/H]_{APOGEE_{DR16}}')
    compare_metals(galah['fe_h'], galah['e_fe_h'],
                   galah['M_H'], ax5, ax6, '[Fe/H]_{GALAH_{DR3}}')
    plt.savefig(plot_save, dpi=100, bbox_inches='tight')
    plt.show()


class GM_Age_GALAH(object):
    """
    used to find the GM used to define W vs [M/H] for various age bins
    in the GALAH dataset
    """
    def __init__(self, GALAH_path, age_err_limit=0.2, distance_limit=500,
                 Rsun=8100., zsun=21., vphi_sun=248.5):
        self.age_err_limit = age_err_limit
        self.distance_limit = distance_limit
        self.Rsun = Rsun
        self.zsun = zsun
        self.vphi_sun = vphi_sun
        self.GALAH_path = GALAH_path
        self.cat = self.load_GALAH_data()

    def load_GALAH_data(self):
        # match GALAH with GALAH ages (VAC)
        galah_hdu = fits.open(self.GALAH_path + 'GALAH_DR3_main_allstar_v2.fits')
        galah_hdu_ages = fits.open(self.GALAH_path + 'GALAH_DR3_VAC_ages_v2.fits')

        galah = Table(galah_hdu[1].data)
        galah_ages = Table(galah_hdu_ages[1].data)

        galah_join = join(galah[(galah['flag_sp'] == 0) & (galah['flag_fe_h'] == 0)],
                          galah_ages[galah_ages['e_age_bstep']/galah_ages['age_bstep'] < self.age_err_limit],
                          keys='sobject_id',
                          join_type='inner')

        # select GALAH sources within 500 pc
        combined_data = {}
        combined_data['ra_gaia'] = np.array(galah_join['ra'][1000 / galah_join['parallax'] < self.distance_limit])
        combined_data['de_gaia'] = np.array(galah_join['dec'][1000 / galah_join['parallax'] < self.distance_limit])
        combined_data['pmra_gaia'] = np.array(galah_join['pmra'][1000 / galah_join['parallax'] < self.distance_limit])
        combined_data['pmde_gaia'] = np.array(galah_join['pmdec'][1000 / galah_join['parallax'] < self.distance_limit])
        combined_data['plx_gaia'] = np.array(galah_join['parallax'][1000 / galah_join['parallax'] < self.distance_limit])
        combined_data['rv_gaia'] = np.array(galah_join['rv_galah'][1000 / galah_join['parallax'] < self.distance_limit])
        combined_data['Age'] = np.array(galah_join['age_bstep'][1000 / galah_join['parallax'] < self.distance_limit])
        combined_data['clAge'] = np.array(galah_join['e_age_bstep'][1000 / galah_join['parallax'] < self.distance_limit])
        combined_data['fe_h'] = np.array(galah_join['fe_h'][1000 / galah_join['parallax'] < self.distance_limit])
        cat = pd.DataFrame(combined_data)

        # calculate UVW and Galactic R
        gxc, gyc, gzc, guc, gvc, gwc=uvw(cat['ra_gaia'],
                                         cat['de_gaia'],
                                         1e-3 * cat['pmra_gaia'],
                                         1e-3 * cat['pmde_gaia'],
                                         1000 / cat['plx_gaia'],
                                         cat['rv_gaia'])

        cat['UVel_gaia'] = guc
        cat['VVel_gaia'] = gvc
        cat['WVel_gaia'] = gwc

        cat['zgal_gaia'] = gzc
        cat['zgal_gaia'][cat['zgal_gaia'] < -1000] = np.nan

        cat['UVel_gaia'][cat['UVel_gaia'] < -1000] = np.nan
        cat['VVel_gaia'][cat['VVel_gaia'] < -1000] = np.nan
        cat['WVel_gaia'][cat['WVel_gaia'] < -1000] = np.nan

        gxc = self.Rsun - gxc
        cat['R'] = np.sqrt(gxc**2 + gyc**2)

        cat['zgal_gaia'] += self.zsun

        cat['R'] /= 1000.

        sc = SkyCoord(ra=np.array(cat['ra_gaia']) * u.deg,
                      dec=np.array(cat['de_gaia']) * u.deg)

        l = np.array(sc.galactic.l)
        b = np.array(sc.galactic.b)

        R_G = (cat['R'] * (cat['VVel_gaia'] + self.vphi_sun)) / self.vphi_sun
        phi = np.arcsin((1000 / cat['plx_gaia']) * 1e-3 * np.sin(np.radians(l)) / cat['R'])
        cat['xmix'] = R_G * np.cos(phi)
        
        # do cut to select good disk like stars
        ev = (((cat['clAge']) / (cat['Age']) < self.age_err_limit) &
              (abs(cat['UVel_gaia']) < 200) &
              (abs(cat['VVel_gaia']) < 200) &
              (abs(cat['WVel_gaia']) < 200))
        cat = cat[ev].reset_index(drop=True)
        return cat

    def find_GMMs(self, ages, plot_dir=None):
        gms = []

        self.cat['used_kde'] = 0

        for i in range(len(ages)-1):
            ev2 = eval("(self.cat['Age'] > ages[i]) & (self.cat['Age'] <= ages[i + 1])")

            idx = list(self.cat[ev2].index.values)
            
            X = np.column_stack(((np.array(self.cat.loc[idx,'fe_h'] - (-0.03) * (self.cat.loc[idx,'xmix'] - 8.1)) + 0.5) / 1,
                                  np.array(self.cat.loc[idx,'WVel_gaia']) / 100))
            gm = GaussianMixture(n_components=3).fit(X)
            
            gms.append(gm)
            
            if plot_dir is not None:
                f, (ax1, ax2) = plt.subplots(1, 2,figsize=(20, 10))
                ax1.scatter(self.cat['fe_h'][ev2], self.cat['WVel_gaia'][ev2],
                            marker='.', c='k', alpha=0.3)
                ax1.grid()
                ax1.set_xlim((-1.5, 0.5))
                ax1.set_ylim((-100, 100))
                
                x, y = np.meshgrid(np.linspace(-1.5, 0.5, 100), np.linspace(-100, 100, 100))
                X = np.column_stack(((x.ravel() + 0.5) / 1,
                                      y.ravel() / 100))
                                  
                dens_kde = np.zeros(len(X))
                for d in trange(0, len(X), 100):
                    dens_kde[d: d + 100] = gm.score_samples(X[d: d + 100, :])
                                  
                dens_kde = np.reshape(dens_kde, x.shape)
                
                dens = ax2.contourf(x, y, np.exp(dens_kde), 20)
                
                ax1.set_xlabel(r'[Fe/H] $+0.03 (R_G -8.1)$')
                ax1.set_ylabel(r'W (km/s)')
                ax1.set_title('GALAH: %d' % ages[i] + r'$<\tau\leq$' + '%d Gyr' % ages[i + 1])
                
                ax2.set_xlabel(r'[Fe/H] $+0.03 (R_G -8.1)$')
                ax2.set_ylabel(r'W (km/s)')
                ax2.set_title('Gaussian Mixture: %d' % ages[i] + r'$<\tau\leq$' + '%d Gyr' % ages[i + 1])
                
                plt.savefig('%s/GM_%2d_t_%2d.png' % (plot_dir, ages[i], ages[i + 1]),
                            dpi=100, bbox_inches='tight')
                plt.show()
        self.gms = gms
        self.ts = ages

    def test_GMM_fit(self, Ntests, plot_dir, npeaks=1, plot_mcmc_prog=True, norm_adds=1):
        sig_all_2 = []

        for nadds in range(Ntests):
            idx = []
            
            f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
            
            axs = [ax1, ax2, ax3]
            
            chi_2 = 0
            
            for ax in axs:
                age_num = np.array([50] * (len(self.ts) - 1))
                age_num = age_num / norm_adds
                if nadds > 0:
                    # how to add for 1 peak and 2 peak examples
                    if npeaks == 1:
                        age_num[nadds - 1] += 250 / norm_adds
                    else:
                        age_num[nadds - 1] += 250 / norm_adds
                        age_num[nadds - 1 + 2] += 250 / norm_adds
                age_num = age_num * (2500 / norm_adds) / np.sum(age_num)

                for i in range(len(self.ts) - 1):
                    ev_age = eval("(self.cat['Age']>self.ts[i]) & (self.cat['Age']<self.ts[i+1]) & (self.cat['used_kde'] != 1)")

                    if len(self.cat[ev_age]) > age_num[i]:
                        idx += random.sample(list(self.cat[ev_age].index.values), int(age_num[i]))
                    else:
                        idx += random.sample(list(self.cat[ev_age].index.values), int(len(self.cat[ev_age])))


                n, bins = np.histogram(self.cat.loc[idx, 'Age'], bins=self.ts)

                MH_norm = np.array(self.cat.loc[idx,'fe_h'] - (-0.03) * (self.cat.loc[idx,'xmix'] - 8.1))
                W = np.array(self.cat.loc[idx,'WVel_gaia'])
                sampler, flat_samples = mcmc_GM_fit(MH_norm, W, self.gms, self.ts)

                mcmc = np.percentile(flat_samples, [16, 50, 84],axis=0)
                q = np.diff(mcmc,axis=0)
                
                chi_2 += np.sum((mcmc[1,:] - n / np.sum(n)) ** 2 / q[1, :] ** 2)
                
                sig_all_2 += list((mcmc[1,:] - n / np.sum(n)) / q[1, :])

                t_mids = np.array([(self.ts[i] + self.ts[i + 1]) / 2 for i in range(len(self.ts) - 1)])

                
                ax.scatter(t_mids, n / np.sum(n), c='k', label='Observed')
                ax.errorbar(t_mids, n / np.sum(n), xerr=[(self.ts[1] - self.ts[0]) / 2] * len(t_mids), fmt='none', ecolor='k', capsize=2)
                ax.scatter(t_mids, mcmc[1, :], c='r', label='MCMC Fit')
                ax.errorbar(t_mids, mcmc[1, :], xerr=[(self.ts[1] - self.ts[0]) / 2] * len(t_mids),
                             yerr=[q[0, :], q[1, :]], fmt='none', ecolor='r', capsize=2)
                ax.set_xlabel('Age (Gyr)')
                ax.set_ylabel('Fraction of Population')
                ax.set_xlim((0,14))
                if npeaks == 2:
                    ax.set_ylim((0, 0.5))
                else:
                    ax.set_ylim((0, 0.65))
                ax.grid()
                ax.legend()
            plt.savefig('%s/GM_%d_peak_test_%d.png' % (plot_dir, npeaks, nadds),
                        dpi=100, bbox_inches='tight')
            plt.show()

            if plot_mcmc_prog:
                fig, axes = plt.subplots(len(mcmc[1,:]), figsize=(10, 18*len(mcmc[1,:])/6), sharex=True)
                samples = sampler.get_chain()
                labels = [r'$f_{0<t<2}$',
                          r'$f_{2<t<4}$',
                          r'$f_{4<t<6}$',
                          r'$f_{6<t<8}$',
                          r'$f_{8<t<10}$',
                          r'$f_{10<t<12}$',
                          r'$f_{12<t<14}$']
                for i in range(len(mcmc[1,:])):
                    ax = axes[i]
                    ax.plot(samples[:, :, i], "k", alpha=0.3)
                    ax.set_xlim(0, len(samples))
                    ax.set_ylabel(labels[i])

                axes[-1].set_xlabel("step number")

                plt.show()
        return sig_all_2


def action_angle(ra, dec, distance, pm_ra, pm_dec, v_hel):
    """
    approximate actions
    """
    o = Orbit(vxvv=[ra, dec, distance, pm_ra, pm_dec, v_hel],radec=True)
    delta = 0.45
    aAS = actionAngleStaeckel(pot=MWPotential2014, delta=delta, c=True)
    actions = aAS(o.R() / 8., o.vR() / 220., o.vT() / 220., o.z() / 8., o.vz() / 220., fixed_quad=True)
    return actions[0][0], actions[1][0], actions[2][0]


def bootstrap_hist(x, bins, N):
    """
    bootstrap a 1d histogram
    """
    bin_mids = np.array([(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)])
    ns = np.zeros((N, len(bin_mids)))
    for i in range(N):
        idx = np.random.randint(len(x), size=len(x))
        n, bins = np.histogram(x[idx], bins=bins, density=True)
        ns[i,:] = n
    return np.mean(ns,axis=0), np.std(ns,axis=0)


def LZ0_FeH(t, FeH, LZ0prob):
    """
    get the intial LZ of a star based on age
    and metallicity
    """
    delta_inner = -0.03
    delta_FeH = -0.093
    if LZ0prob / 235 < 3:
        delta_FeH_LZ = delta_inner
    else:
        delta_FeH_LZ = delta_FeH
    ft = (1 - t / 12) ** 0.45
    Fe_H_max = 0.7
    if LZ0prob / 235 < 3:
        b_Fe_H = 0.
        return (FeH  - Fe_H_max * ft - b_Fe_H) * (235. / delta_FeH_LZ)
    else:
        b_Fe_H = (delta_inner - delta_FeH) * 3 
        return (FeH  - Fe_H_max * ft - b_Fe_H) * (235. / delta_FeH_LZ)
    
def FeH_from_LZ0(t, LZ0):
    """
    Get metallicity from intial LZ
    """
    delta_inner = -0.03
    delta_FeH = -0.093
    if LZ0 / 235. < 3.:
        delta_FeH_LZ = delta_inner
    else:
        delta_FeH_LZ = delta_FeH
    ft = (1 - t / 12.) ** 0.45
    Fe_H_max = 0.7
    if LZ0 / 235. < 3.:
        b_Fe_H = 0.
    else:
        b_Fe_H = (delta_inner - delta_FeH) *3 
    return Fe_H_max * ft + b_Fe_H + delta_FeH_LZ * LZ0 / 235


def prob_LZ0(t, LZ, LZ0s):
    """
    Get the most PDF of initial LZ based on
    current LZ and age of star for some
    LZOs
    """
    sig_LZ = 567.06
    t_m = 6.
    sig_t = sig_LZ * np.sqrt(t / t_m)
    Rd = 2.85
    D = sig_LZ**2/(2 * 235 * Rd * t_m)
    intg = np.exp(-(LZ - LZ0s - D * t)**2/(2 * sig_t**2))
    return intg

def bootstrap_Rbirth_LZ(mh, mherr, mhbins, age, ageerr, age_bins,
                        LZ, LZerr, LZbins, R_birth_bins, N_samp, N):
    """
    bootstrap the probably birth radius distribution of a group of stars
    based on metallicity, age and current LZ distributions
    """
    mhbins_mid = np.array([(mhbins[i] + mhbins[i + 1]) / 2 for i in range(len(mhbins) - 1)])
    age_bins_mid = np.array([(age_bins[i] + age_bins[i + 1]) / 2 for i in range(len(age_bins) - 1)])
    LZbins_mid = np.array([(LZbins[i] + LZbins[i + 1]) / 2 for i in range(len(LZbins) - 1)])

    ns = np.zeros((N, len(R_birth_bins) - 1))
    LZ0s = np.linspace(0, 3000, 20)
    for i in range(N):
        mh_rand = mh + np.random.randn(len(mh)) * mherr
        mh_rand[mh_rand < 0] = 0
        mh_samp = np.random.choice(mhbins_mid, size=N_samp,
                                   p=mh_rand / np.sum(mh_rand))
        age_rand = age + np.random.randn(len(age)) * ageerr
        age_rand[age_rand < 0] = 0
        age_samp = np.random.choice(age_bins_mid, size=N_samp,
                                    p=age_rand / np.sum(age_rand))
        LZ_rand = LZ + np.random.randn(len(LZ)) * LZerr
        LZ_rand[LZ_rand < 0] = 0
        LZ_samp = np.random.choice(LZbins_mid, size=N_samp,
                                   p=LZ_rand / np.sum(LZ_rand))

        LZ0_samp = np.zeros(N_samp)
        for j in range(N_samp):
            probs = prob_LZ0(age_samp[j], LZ_samp[j], LZ0s)
            LZ0_samp[j] = LZ0_FeH(age_samp[j], mh_samp[j], LZ0s[np.argmax(probs)])

        R_birth = LZ0_samp / 235

        ns[i, :], _ = np.histogram(R_birth, bins=R_birth_bins, density=True)
    return np.mean(ns, axis=0), np.std(ns, axis=0)


def loop(KM_metals, i):
    """
    multiprocessing loops for calculating actions
    """
    ra = KM_metals.loc[i,'RA'] # [deg]
    dec = KM_metals.loc[i,'DEC'] # [deg]
    distance = 1/KM_metals.loc[i,'plx'] #kpc
    pm_ra = KM_metals.loc[i,'pmra'] # [mas/yr]
    pm_dec = KM_metals.loc[i,'pmde'] # [mas/yr]
    v_hel = KM_metals.loc[i,'rv'] # [km /s]
    JR,LZ,JZ = action_angle(ra, dec, distance, pm_ra, pm_dec, v_hel)
    return [JR, LZ, JZ]


class KM_metals(object):
    """
    Find probable age distributions for KM stars in SN
    """
    def __init__(self,  metals_file, gaia_file,
                 Rsun=8100., zsun=21., vphi_sun=248.5,
                 metal_err_cut=0.3):
        self.metals_file = metals_file
        self.gaia_file = gaia_file
        self.Rsun = Rsun
        self.zsun = zsun
        self.vphi_sun = vphi_sun
        self.metal_err_cut = metal_err_cut

        # load the data
        self.KM_metals = self.load_KM_data()

        # define the polygon regions used in the analysis
        self.polygons = self.init_polygons()

        # set names of kinematic groups
        self.names = ['Sirius', 'Coma', 'Horn?', 'Hyades',
                      'Pleiades', 'Herc 1', 'Herc 2', 'Herc 3',
                      '??? (Herc 1)', '??? (Herc 2)']

    def load_KM_data(self):
        df1 = pd.read_csv(self.metals_file, delim_whitespace=True)

        gaia1 = pd.read_csv(self.gaia_file,
                            usecols=[0, 2, 4, 5, 7, 9, 11, 12, 13, 14, 15],
                            names=['RA', 'DEC', 'ID', 'plx', 'pmra',
                                   'pmde', 'G', 'BP', 'RP', 'rv', 'rv_err'],
                            skiprows=1)
        
        # grab the Gaia data
        df1 = df1.rename(columns={"Gaia_ID": "ID"})
        df1 = df1.merge(gaia1, on='ID', how='inner')

        del gaia1

        KM_metals = df1[(df1['M_H_std'] * 1.95 < self.metal_err_cut)]

        # get UVW velocity and galactic components

        ev_uvw = (KM_metals['rv'] != 9999.99) & (~np.isnan(KM_metals['rv']))
        KM_metals = KM_metals[ev_uvw].reset_index(drop=True)
        gx, gy, gz, gu, gv, gw = uvw(np.array(KM_metals['RA']),
                                     np.array(KM_metals['DEC']),
                                     1e-3 * np.array(KM_metals['pmra']),
                                     1e-3 * np.array(KM_metals['pmde']),
                                     1 / (np.array(KM_metals['plx']) * 1e-3),
                                     np.array(KM_metals['rv']))
        sc = SkyCoord(ra=np.array(KM_metals['RA']) * u.deg,
                      dec=np.array(KM_metals['DEC']) * u.deg)

        l = np.array(sc.galactic.l)
        b = np.array(sc.galactic.b)

        KM_metals['dist'] = 1 / (KM_metals['plx'] * 1e-3)
        KM_metals['gu'] = gu
        KM_metals['gv'] = gv
        KM_metals['gw'] = gw
        KM_metals['gx'] = gx
        KM_metals['gy'] = gy
        KM_metals['gz'] = gz

        gx = self.Rsun - gx
        KM_metals['R'] = np.sqrt(gx ** 2 + gy ** 2)

        KM_metals['gz'] += self.zsun

        KM_metals['R'] /= 1000.

        R_G = (KM_metals['R'] * (KM_metals['gv'] + self.vphi_sun)) / self.vphi_sun
        phi = np.arcsin(KM_metals['dist'] * 1e-3 * np.sin(np.radians(l)) / KM_metals['R'])
        KM_metals['xmix'] = R_G * np.cos(phi)

        # r = p_map(partial(loop, KM_metals), KM_metals.index.to_list(), num_cpus=4)

        # r = np.array(r)
        # KM_metals['J_R'] = r[:,0]
        # KM_metals['L_Z'] = r[:,1]
        # KM_metals['J_Z'] = r[:,2]

        return KM_metals

    def init_polygons(self):
        """
        function to store the polygons used to initially
        define kinematic groups
        """
        polygons = []
        #sirius stream
        ellipse = Polygon(xy=np.column_stack(([-3, 15, 46.5, 32.5, -23.2],
                                              [8.6, 8.6, 7.9, 7.6, 8.4])),
                          edgecolor='r', fc='None', lw=2)
        polygons.append(ellipse)
        #coma stream
        ellipse = Polygon(xy=np.column_stack(([-23.2, 32.5, 27.5, 24.5, -34.3],
                                              [8.4, 7.6, 7.2, 7.1, 8])),
                                edgecolor='r', fc='None', lw=2)
        polygons.append(ellipse)
        #horn???
        ellipse = Polygon(xy=np.column_stack(([32.5, 27.5, 54,59.5, 46.2],
                                              [7.6, 7.2, 7, 7.2, 7.5])),
                          edgecolor='r', fc='None', lw=2)
        polygons.append(ellipse)
        #hyades
        ellipse = Polygon(xy=np.column_stack(([-34.3, -12, -31.2, -53.4, -55.2],
                                              [8, 7.65, 7.2, 7.3, 7.7])),
                          edgecolor='r', fc='None', lw=2)
        polygons.append(ellipse)
        #pleides
        ellipse = Polygon(xy=np.column_stack(([-12, 12, 0, -31.2],
                                              [7.65, 7.29, 6.8, 7.2])),
                          edgecolor='r', fc='None', lw=2)
        polygons.append(ellipse)
        #hercules 1
        ellipse = Polygon(xy=np.column_stack(([0, -11.8, -54.5, -54.5, -31.2],
                                              [6.8, 6.7, 6.7, 7, 7.2])),
                          edgecolor='r', fc='None', lw=2)
        polygons.append(ellipse)
        #hercules 2
        ellipse = Polygon(xy=np.column_stack(([-11.8, 17.3, 17.3, -65, -65, -54.5],
                                              [6.7, 6.6, 6.3, 6.3, 6.7, 6.7])),
                          edgecolor='r', fc='None', lw=2)
        polygons.append(ellipse)
        #hercules 3
        ellipse = Polygon(xy=np.column_stack(([17.3, -57.7, -51, 40],
                                              [6.3, 6, 5.5, 6.1])),
                          edgecolor='r', fc='None', lw=2)
        polygons.append(ellipse)
        #???
        ellipse = Polygon(xy=np.column_stack(([-54.5, -54.5, -73.8, -74.8],
                                              [6.7, 7, 7.1, 6.7])),
                          edgecolor='r', fc='None', lw=2)
        polygons.append(ellipse)
        #???
        ellipse = Polygon(xy=np.column_stack(([-65, -65, -74.8, -94, -93],
                                              [6.3, 6.7, 6.7, 6.7, 6.3])),
                          edgecolor='r', fc='None', lw=2)
        polygons.append(ellipse)
        return polygons

    def assign_kinematic_groups(self, plot_groups=True):
        """
        Assign the sample to the various kinemtic groups
        """
        # identify stars within polygons
        self.KM_metals['group'] = -1
        for n, (polygon, name) in enumerate(zip(self.polygons, self.names)):
            path = mpltPath.Path(polygon.xy)
            ev = path.contains_points(np.column_stack((np.array(self.KM_metals['gu']),
                                                       np.array(self.KM_metals['xmix']))))
            self.KM_metals['group'][ev] = n

        # perform the PCA analysis to get the ellipse fits
        self.KM_metals['group_pca'] = -1
        self.KM_metals['group_pca_sig'] = -1

        # dict with individual dfs for each group
        self.stream_dfs = {}

        # keep running list of all stars within 2 sigma
        # of kinematic group center
        self.id2 = []

        # keep the means and stds of the resulting PCA
        self.mean1 = []
        self.mean2 = []
        self.std1 = []
        self.std2 = []
        self.angles = []

        for i in range(len(self.names)):
            group = i
            ev_group = self.KM_metals['group'] == group
            X = np.column_stack((self.KM_metals['gu'][ev_group],
                                 self.KM_metals['xmix'][ev_group]))
            pca = PCA(n_components=2)
            trans = pca.fit_transform(X)
            
            # get mean/std in each direction
            mean1 = np.median(trans[:,0])
            std1 = np.std(trans[:,0])

            mean2 = np.median(trans[:,1])
            std2 = np.std(trans[:,1])
            
            # transform the entire sample
            X = np.column_stack((self.KM_metals['gu'], self.KM_metals['xmix']))
            trans = pca.transform(X)
            
            if plot_groups:
                f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
                ax1.scatter(trans[:,0], trans[:,1], c='k', marker='.')
                ax2.scatter(self.KM_metals['gu'], self.KM_metals['xmix'],
                            c='k', marker='.', alpha=0.1)
            # assign all in group as being within 3 sigma
            sig_cut = 3.
            ev = (trans[:,0] - mean1) ** 2 / (std1 * sig_cut) ** 2 + (trans[:,1] - mean2) ** 2 / (std2 * sig_cut) ** 2 < 1
            self.stream_dfs['%d' % i] = self.KM_metals[ev].copy().reset_index(drop=True)
            self.stream_dfs['%d' % i]['group_pca'] = i
            X = np.column_stack((self.stream_dfs['%d' % i]['gu'],
                                 self.stream_dfs['%d' % i]['xmix']))
            trans = pca.transform(X)

            # save all sigma groups
            c_sig = ['r', 'g', 'b']
            for sig_cut, c in zip(range(3, 0, -1), c_sig):
                ev = (trans[:,0] - mean1) ** 2 / (std1 * sig_cut) ** 2 + (trans[:,1] - mean2) ** 2 / (std2 * sig_cut) ** 2 < 1
                self.stream_dfs['%d' % i]['group_pca_sig'][ev] = sig_cut
                if plot_groups:
                    ax1.scatter(trans[:,0][ev], trans[:,1][ev], c=c, marker='.')
                    ax2.scatter(self.stream_dfs['%d' % i]['gu'][ev],
                                self.stream_dfs['%d' % i]['xmix'][ev],
                                c=c, marker='.', alpha=0.1)
            if plot_groups: 
                ax1.grid()
                ax2.grid()
                ax2.set_xlim((-100, 100))
                ax2.set_ylim((5, 10))
                plt.title(self.names[i])
                plt.show()
            self.id2 += list(self.stream_dfs['%d' % i]['ID'][self.stream_dfs['%d' % i]['group_pca_sig'] <= 2])

            comps = []
            As = []
            for length, vector in zip(pca.explained_variance_, pca.components_):
                v = vector * 3 * np.sqrt(length)
                comps.append(np.sqrt(v[0] ** 2 + v[1] ** 2))
                As.append(np.arctan(v[1] / v[0]) * (180 / np.pi))
            self.mean1.append(pca.inverse_transform([mean1, mean2])[0])
            self.mean2.append(pca.inverse_transform([mean1, mean2])[1])
            self.std1.append(comps[0])
            self.std2.append(comps[1])
            self.angles.append(As[0])

    def estimate_age_distribution(self, gms, plot_dir, ts):
        """
        estimate the age distributions for all kinematic groups
        (compared to the background)
        """
        t_mids = np.array([(ts[i] + ts[i + 1]) / 2 for i in range(len(ts) - 1)])

        Rb_bins = np.arange(0, 22, 2)  
        Rb_bins_mid = np.array([(Rb_bins[i] + Rb_bins[i + 1]) / 2 for i in range(len(Rb_bins) - 1)])

        # save all age distributions
        self.all_age_dists = {}
        self.all_age_dists_errs = {}

        for group in trange(len(self.names)):
            xmin = np.min(self.stream_dfs['%d' % group]['xmix'][self.stream_dfs['%d' % group]['group_pca_sig'] <= 2])
            xmax = np.max(self.stream_dfs['%d' % group]['xmix'][self.stream_dfs['%d' % group]['group_pca_sig'] <= 2])

            self.all_age_dists[group] = []
            self.all_age_dists_errs[group] = []

            ev1 = (self.KM_metals['xmix'] >= xmin) & (self.KM_metals['xmix'] <= xmax)
            ev2 = self.KM_metals['ID_GAIA'].isin(self.id2)

            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

            # find the age distributions for 1 sigma, 2 sigma and background
            ev_groups = [self.stream_dfs['%d' % group]['group_pca_sig'] == 1,
                         self.stream_dfs['%d' % group]['group_pca_sig'] == 2,
                         ev1 & ~ev2]
            colors = ['k', 'b', 'r']
            labels = [r' $[(U,x_{mix}) \leq 1 \sigma]$',
                      r' $[1 \sigma < (U,x_{mix}) \leq 2 \sigma]$',
                      r'Background']
            df_type = ['stream', 'stream', 'KM']
            for ev_group, color, label in zip(ev_groups, colors, labels):
                if df_type == 'stream':
                    data = self.stream_dfs['%d' % group]
                else:
                    data = self.KM_metals
                xs = np.array(data['M_H'][ev_group] -
                              (-0.03) * (data['xmix'][ev_group] - 8.1) - 0.188)
                ys = np.array(data['gw'][ev_group])
                sampler, flat_samples = mcmc_GM_fit(xs, ys, gms, ts)

                mcmc = np.percentile(flat_samples, [16, 50, 84], axis=0)
                q = np.diff(mcmc, axis=0)

                self.all_age_dists[group].append(mcmc[1,:])
                self.all_age_dists_errs[group].append(q[1,:])

                ev_sig = self.KM_metals['ID'].isin(data['ID'][ev_group])
                self.KM_metals.loc[ev_sig, 'age_0'] += mcmc[1,0]
                self.KM_metals.loc[ev_sig, 'age_1'] += mcmc[1,1]
                self.KM_metals.loc[ev_sig, 'age_2'] += mcmc[1,2]
                self.KM_metals.loc[ev_sig, 'age_3'] += mcmc[1,3]
                self.KM_metals.loc[ev_sig, 'age_4'] += mcmc[1,4]
                self.KM_metals.loc[ev_sig, 'age_5'] += mcmc[1,5]
                self.KM_metals.loc[ev_sig, 'age_6'] += mcmc[1,6]

                # determine the Rbirth distribution
                mh_bins = np.arange(-1.5, 0.6, 0.1)
                mh_mids = np.array([(mh_bins[i] + mh_bins[i + 1]) / 2 for i in range(len(mh_bins) - 1)])
                mh_stream1, mh_err_stream1 = bootstrap_hist(np.array(data['M_H'][ev_group]),
                                                            mh_bins, 1000)

                LZ_bins = np.arange(850, 2600, 10)
                LZ_mids = np.array([(LZ_bins[i] + LZ_bins[i + 1]) / 2 for i in range(len(LZ_bins) - 1)])
                LZ_stream1, LZ_err_stream1 = bootstrap_hist(220 * 8 * np.array(data['L_Z'][ev_group]),
                                                            LZ_bins, 1000)

                Rb_stream1, RBerr_stream1 = bootstrap_Rbirth_LZ(mh_stream1, mh_err_stream1, mh_bins, mcmc[1,:], q[1,:],
                                                                ts, LZ_stream1, LZ_err_stream1, LZ_bins,
                                                                Rb_bins,
                                                                len(self.stream_dfs['%d' % group]['L_Z'][ev_group]), 1000)

                # plot the results
                ax1.scatter(t_mids, mcmc[1,:], c=color, label=self.names[group] + label)
                ax1.hist(ts[:-1], bins=ts, weights=mcmc[1,:], facecolor=color, label='_nolabel_', alpha=0.3)
                ax1.errorbar(t_mids, mcmc[1,:],
                             yerr=[q[0,:], q[1,:]], xerr=[(ts[1] - ts[0]) / 2] * len(t_mids),
                             fmt='none', ecolor=color, capsize=2)

                ax2.scatter(Rb_bins_mid, Rb_stream1, c=color, label=self.names[group] + label)
                ax2.hist(Rb_bins[:-1], bins=Rb_bins, weights=Rb_stream1, facecolor=color, alpha=0.3, label='_nolabel_')
                ax2.errorbar(Rb_bins_mid, Rb_stream1, xerr=[1] * len(Rb_stream1),
                             yerr=RBerr_stream1, fmt='none', ecolor=color, capsize=2)

            ax1.grid()
            ax1.set_xlabel('Age (Gyr)')
            ax1.set_ylabel('Fraction of Population')
            ax1.set_xlim((0, 14))
            ax1.set_ylim((0, 0.9))
            ax1.legend(prop={'size': 12.5})

            ax2.grid()
            ax2.set_xlabel(r'$R_{Birth}$')
            ax2.set_ylabel('N (Normalized)')
            ax2.set_xlim((0, 20))
            ax2.set_ylim((0, 0.22))
            ax2.legend(prop={'size': 12.5})

            plt.savefig('%s/%s_age_fit_sigma_regions_same_scale.png' % (plot_dir, self.names[group].replace(' ', '_')),
                        bbox_inches='tight', dpi=100)
            plt.show()
