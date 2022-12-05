import numpy as np
from astropy.io import fits
import pandas as pd
from astropy.table import Table, join
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pylab as plt
import os
plt.style.use('%s/mystyle.mplstyle' % os.environ['MPL_STYLES'])


from multiprocessing import Pool
import emcee
import random

from sklearn.mixture import GaussianMixture

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

from tqdm.notebook import trange

import pickle

from scipy.spatial import cKDTree

import mpl_scatter_density


def uvw(ra, de, pmra, pmde, dist, rv):
    """
    Calculate UVW velocity

    Parameters
    ----------
    ra:
        Right Ascension in degrees

    dec:
        Declination in degrees

    pmra:
        Proper motion in RA directionn in as/yr

    pmde:
        Proper motion in Decl. directionn in as/yr

    dist:
        Distance in pc

    rv:
        Radial velocity in km/s

    Returns
    -------
    gx:
        Galactic X distance (from Sun) in m

    gy:
        Galactic Y distance (from Sun) in m

    gz:
        Galactic Z distance (from Sun) in m

    gu:
        U velocity in km/s

    gv:
        V Velocity in km/s

    gw:
        W velocity in km/s
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

    Parameters
    ----------
    x: np.array
        Data in x-direction

    y: np.array
        Data in y-direction

    binx: np.array
        Bins in x-direction

    biny: np.array
        Bins in y-direction

    N: int
        Number of iterations for bootstrap
    """
    n, binsx, binsy = np.histogram2d(x, y, bins=[binx, biny], density=True)
    ns = np.zeros((N, len(n.ravel())))
    for i in range(N):
        idx = np.random.randint(len(x),size=len(x))
        n, binsx, binsy = np.histogram2d(x[idx], y[idx], bins=[binx, biny], density=True)
        ns[i, :] = n.T.ravel()
    return np.mean(ns, axis=0), np.std(ns, axis=0)


def est_sigma4(x, theta, ts):
    """
    Calculate the total W vs [M/H] distriubtion based on
    fraction in each age bin

    Parameters
    ----------
    x: list
        input where first index is the X values of each bin in 2D hist
        and second index is a list of the gaussian mixture model for each
        age bin

    theta: list
        fraction for each age bin

    ts: np.array
        bin edges for the age bins

    Returns
    -------
    y_mod: np.array
        Resulting modeled distirbution
    """ 
    y_mod = np.zeros(len(x[0]))
    t_mids = np.array([(ts[i] + ts[i + 1]) / 2 for i in range(len(ts) - 1)])
    for i in range(len(theta)):
        y_mod += theta[i] * x[i + 1]            
    return y_mod


def log_likelihood(theta, x, y, yerr, ts):
    """
    Calculate the log likelihood

    Parameters
    ----------
    theta: list
        fraction for each age bin

    x: list
        input where first index is the X values of each bin in 2D hist
        and second index is a list of the gaussian mixture model for each
        age bin

    y: np.array
        1D array (so raveled 2D hist) of the observed data

    yerr: np.array
        1D array (so raveled 2D hist) of the errors on observed data

    ts: np.array
        bin edges for the age bins

    Returns
    -------
    log_like: float
        Resulting log likelihood
    """
    model = est_sigma4(x, theta, ts)
    sigma2 = yerr ** 2
    log_like = -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))
    return log_like


def log_prior(theta):
    """
    returns either 0 or inf based on if prior is contrained or not
    """
    if np.all(theta > 0) and round(np.sum(theta), 4) == 1.:
        return 0.0
    return -np.inf


def log_probability(theta, x, y, yerr, ts):
    """
    Calculatate the model probability based on prior and
    log likelihood

    Parameters
    ----------
    theta: list
        fraction for each age bin

    x: list
        input where first index is the X values of each bin in 2D hist
        and second index is a list of the gaussian mixture model for each
        age bin

    y: np.array
        1D array (so raveled 2D hist) of the observed data

    yerr: np.array
        1D array (so raveled 2D hist) of the errors on observed data

    ts: np.array
        bin edges for the age bins 
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr, ts)


def mcmc_GM_fit(MH_norm, W, gms, ts, progress=True):
    """
    find the probably age distirbution based on GMM fit
    from GALAH and W, [M/H] of data

    Parameters
    ----------
    MH_norm: np.array
        The metallicity corrected for radial gradient

    W: np.array
        W velocities

    gms: list
        Gaussian Mixture models for each age bin

    ts: np.array
        bin edges for the age bins

    progress: boolean
        Display progress of MCMC

    Returns
    -------
    sampler:
        The sampler from emcee

    flat_samples:
        The flattened samples for each walker after trimming
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

    sampler = emcee.EnsembleSampler(nwalkers,
                                    ndim,
                                    log_probability,
                                    args=([Xmid.ravel()[ev_y]] + y_comps,
                                          y[ev_y], yerr[ev_y], ts))
    sampler.run_mcmc(pos, 10000, progress=progress)

    tau = sampler.get_autocorr_time(quiet=True)
    flat_samples = sampler.get_chain(discard=int(3 * max(tau)), thin=int(max(tau) / 2), flat=True)
    return sampler, flat_samples


def bootstrap_mean(x, N):
    """
    bootstrap the mean of sample x over N iterations
    """
    if len(x) > 50:
        means = np.zeros(N)
        for i in range(N):
            means[i] = np.nanmean(x[np.random.randint(len(x), size=len(x))])
        return np.mean(means), np.std(means)
    else:
        return np.nan, np.nan
    
def bootstrap_median(x ,N):
    """
    bootstrap the median of sample x over N iterations
    """
    if len(x) > 50:
        means = np.zeros(N)
        for i in range(N):
            means[i] = np.nanmedian(x[np.random.randint(len(x), size=len(x))])
        return np.mean(means), np.std(means)
    else:
        return np.nan, np.nan
    
def bootstrap_std(x, N):
    """
    bootstrap the standard deviation of sample x over N iterations
    """
    if len(x) > 50:
        means = np.zeros(N)
        for i in range(N):
            means[i] = np.nanstd(x[np.random.randint(len(x), size=len(x))])
        return np.mean(means), np.std(means)
    else:
        return np.nan, np.nan


def compare_metals(MH_spec, MH_spec_err, MH_photo, MH_photo_err,
                   MG, ax1, ax2, ax3, survey_name):
    """
    compare photometric metallicities to spec.
    and make the comparison plots
    
    Parameters
    ----------
    MH_spec:
        spectroscopic metallicities

    MH_spec_err:
        Errors in spectroscopic metallicities

    MH_photo:
        Corresponding photometric metallicities

    MH_photo:
        Errors in Corresponding photometric metallicities

    MG:
        absolute mag of star

    ax1:
        matplotlib axis for 1-to-1 plot

    ax2:
        matplotlib axis for relative error histogram

    ax3:
        matplotlib axis for error as func of MG
    
    survey_name: str
        Label to use for spectroscopic metallicity axis
    """
    ax1.scatter_density(MH_photo, MH_spec, dpi=20)
    ax1.scatter(100, 100, marker='.', alpha=0, label='N = %d' % len(MH_photo))
    ax1.set_xlim((-1, 0.6))
    ax1.set_ylim((-1, 0.6))
    ax1.plot([-1, 0.5], [-1, 0.5], '--', c='r', label='_nolabel_')
    ax1.set_xlabel(r'$[M/H]_{Photo}$')
    ax1.set_ylabel(r'$%s$' % survey_name)
    ax1.legend()
    # ax1.grid()

    m2 = (MH_spec - MH_photo) / np.sqrt(MH_photo_err **2 + MH_spec_err**2)

    Nbin, _, _ = ax2.hist(m2, bins=np.linspace(-5, 5, 50))
    ax2.grid()
    median = bootstrap_median(np.array(m2), 1000)
    sigma = bootstrap_std(np.array(m2), 1000)
    xcont = np.linspace(-5, 5, 1000)
    ax2.plot(xcont, np.nanmax(Nbin) * np.exp(-0.5 * (xcont - median[0]) ** 2), '--', c='r',
             label=r'Median = %.4f$\pm$%.4f,' % median)
             # label=r'Median = %.4f$\pm$%.4f,' % median + '\n' +
             #       r'$\sigma$ = %.4f$\pm$%.4f,' % sigma)
    ax2.legend()
    ax2.set_xlabel(r'$\frac{%s - [M/H]_{Photo}}{\sqrt{\sigma_{%s}^2 + \sigma_{[M/H]_{Photo}}^2}}$' % (survey_name, survey_name))
    ax2.set_ylabel('N')
    ax2.set_ylim((0, max(Nbin) + 0.2 * max(Nbin)))

    m2 = MH_spec - MH_photo

    ax3.scatter_density(MG, m2, dpi=20)
    ax3.set_xlabel(r'$M_G$')
    ax3.set_ylabel(r'$%s - [M/H]_{Photo}$' % survey_name)
    ax3.set_ylim((-0.6, 0.6))
    ax3.set_xlim((5, 11))
    ax3.axhline(0, linestyle='--', c='r')


def compare_all_spec_surveys(galah_file, apogee_17_file, apogee_14_file,
                             gaia_mh_file, KM_metals, plot_save):
    """
    compare spec metallicities of all surveys

    Parameters
    ----------
    galah_file: str
        path to Galah file

    apogee_17_file: str
        Path to APOGEE DR17 file

    apogee_14_file: str
        Path to APOGEE DR14 file

    gaia_mh_file: str
        Path to the gaia RVS and BP/RP M/H file

    KM_metals: pd.DataFrame
        photometric metallicity dataframe from KM_Metals()

    plot_save: str
        Name of file to save plot as
    """
    hdu = fits.open(galah_file)

    galah = Table(hdu[1].data).to_pandas()

    galah = galah.drop_duplicates(subset='dr3_source_id', keep='first', ignore_index=True)
    galah = galah.rename(columns={"dr3_source_id": "ID"})
    galah = galah.merge(KM_metals[['M_H', 'M_H_std', 'ID', 'G', 'plx']], on='ID', how='inner')

    hdu = fits.open(apogee_17_file)

    apogee = Table(hdu[1].data)[['APOGEE_ID', 'GAIAEDR3_SOURCE_ID', 'M_H', 'M_H_ERR']].to_pandas()
    apogee = apogee.drop_duplicates(subset='GAIAEDR3_SOURCE_ID',
                                    keep='first', ignore_index=True)
    apogee = apogee[abs(apogee['M_H'])<1000.].reset_index()
    apogee = apogee.rename(columns={"GAIAEDR3_SOURCE_ID": "ID"})
    apogee = apogee.rename(columns={"M_H": "M_H_spec"})
    apogee = apogee.merge(KM_metals[['M_H', 'M_H_std', 'ID', 'G', 'plx']],
                          on='ID', how='inner')

    hdu = fits.open(apogee_14_file)

    apdr14 = Table(hdu[1].data)[['APOGEE_ID', 'M_H', 'M_H_ERR']].to_pandas()
    apdr14 = apdr14.drop_duplicates(subset='APOGEE_ID',
                                    keep='first', ignore_index=True)
    apdr14 = apdr14[abs(apdr14['M_H'])<1000.].reset_index()
    apdr14 = apdr14.rename(columns={"M_H": "M_H_spec"})
    apdr14 = apdr14.merge(apogee[['ID', 'APOGEE_ID']], on='APOGEE_ID', how='inner')
    apdr14 = apdr14.merge(KM_metals[['M_H', 'M_H_std', 'ID', 'G', 'plx']],
                          on='ID', how='inner')

    gaia_mh = pd.read_csv(gaia_mh_file)
    gaia_mh = gaia_mh.merge(KM_metals[['M_H', 'M_H_std', 'ID', 'G', 'plx']],
                            left_on='source_id', right_on='ID')

    f, (axs1, axs2, axs3, axs4, axs5) = plt.subplots(5, 3, figsize=(30, 50),
                                                     subplot_kw={'projection': 'scatter_density'})

    compare_metals(apdr14['M_H_spec'], apdr14['M_H_ERR'],
                   apdr14['M_H'], apdr14['M_H_std'],
                   apdr14['G'] + 5 * np.log10(1e-3 * apdr14['plx']) + 5,
                   axs1[0], axs1[1], axs1[2], '[M/H]_{APOGEE_{DR14}}')
    compare_metals(apogee['M_H_spec'], apogee['M_H_ERR'],
                   apogee['M_H'], apogee['M_H_std'],
                   apogee['G'] + 5 * np.log10(1e-3 * apogee['plx']) + 5,
                   axs2[0], axs2[1], axs2[2], '[M/H]_{APOGEE_{DR17}}')
    compare_metals(galah['fe_h'], galah['e_fe_h'],
                   galah['M_H'], galah['M_H_std'],
                   galah['G'] + 5 * np.log10(1e-3 * galah['plx']) + 5,
                   axs3[0], axs3[1], axs3[2], '[Fe/H]_{GALAH_{DR3}}')
    ev = ~np.isnan(gaia_mh['mh_gspphot'])
    compare_metals(gaia_mh['mh_gspphot'][ev], gaia_mh['mh_gspphot_upper'][ev] - gaia_mh['mh_gspphot'][ev],
                   gaia_mh['M_H'][ev], gaia_mh['M_H_std'][ev],
                   gaia_mh['G'][ev] + 5 * np.log10(1e-3 * gaia_mh['plx'][ev]) + 5,
                   axs4[0], axs4[1], axs4[2], '[M/H]_{GSP-Phot}')
    ev = ~np.isnan(gaia_mh['mh_gspspec'])
    compare_metals(gaia_mh['mh_gspspec'][ev], gaia_mh['mh_gspspec_upper'][ev] - gaia_mh['mh_gspspec'][ev],
                   gaia_mh['M_H'][ev], gaia_mh['M_H_std'][ev],
                   gaia_mh['G'][ev] + 5 * np.log10(1e-3 * gaia_mh['plx'][ev]) + 5,
                   axs5[0], axs5[1], axs5[2], '[M/H]_{GSP-Spec}')
    plt.tight_layout()
    plt.savefig(plot_save, dpi=100, bbox_inches='tight')
    plt.close('all')


class GM_Age_GALAH(object):
    """
    used to find the GM used to define W vs [M/H] for various age bins
    in the GALAH dataset

    Parameters
    ----------
    GALAH_path: str
        path to where GALAH files are stored

    age_err_limit: float
        Maximum age error for ages in GALAH used

    distance_limit: float
        Distance limit to use for GALAH sample

    Rsun: float
        Galactocentric radius of the Sun in m

    zsun: float
        Galactocentric z of the Sun in m

    vphi_sun: float
        V_phi of Sun in km/s
    """
    def __init__(self, GALAH_path, age_err_limit=0.2, distance_limit=500,
                 Rsun=8100., zsun=21., vphi_sun=248.5, vLSR=235.):
        self.age_err_limit = age_err_limit
        self.distance_limit = distance_limit
        self.Rsun = Rsun
        self.zsun = zsun
        self.vphi_sun = vphi_sun
        self.vLSR = vLSR
        self.GALAH_path = GALAH_path
        self.cat = self.load_GALAH_data()

    def load_GALAH_data(self):
        """
        Load the Galah data from the appropriate files and with the
        specified limits
        """
        # match GALAH with GALAH ages (VAC)
        galah_hdu = fits.open(self.GALAH_path + 'GALAH_DR3_main_allstar_v2.fits')
        galah_hdu_ages = fits.open(self.GALAH_path + 'GALAH_DR3_VAC_ages_v2.fits')
        galah_hdu_dr3 = fits.open(self.GALAH_path + 'GALAH_DR3_VAC_GaiaEDR3_v2.fits')

        galah = Table(galah_hdu[1].data)
        galah_ages = Table(galah_hdu_ages[1].data)
        galah_dr3 = Table(galah_hdu_dr3[1].data)

        galah_join = join(galah[(galah['flag_sp'] == 0) & (galah['flag_fe_h'] == 0)],
                          galah_ages[galah_ages['e_age_bstep']/galah_ages['age_bstep'] < self.age_err_limit],
                          keys='sobject_id',
                          join_type='inner')
        galah_join = join(galah_join,
                          galah_dr3,
                          keys='sobject_id',
                          join_type='inner')
        galah_join = galah_join.to_pandas()

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

        R_G = (cat['R'] * (cat['VVel_gaia'] + self.vphi_sun)) / self.vLSR
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
        """
        Calculate Gaussian Mixture models for GALAH data for some
        binning in age

        Parameters
        ----------
        ages:
            Ages of GALAH sources

        plot_dir: str
            Directory to store resulting GM plots
        """
        gms = []

        self.cat['used_kde'] = 0

        for i in range(len(ages)-1):
            ev2 = eval("(self.cat['Age'] > ages[i]) & (self.cat['Age'] <= ages[i + 1])")

            idx = list(self.cat[ev2].index.values)
            
            X = np.column_stack(((np.array(self.cat.loc[idx,'fe_h']) + 0.5) / 1,
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
                
                ax1.set_xlabel(r'[Fe/H]')
                ax1.set_ylabel(r'W (km/s)')
                ax1.set_title('GALAH: %d' % ages[i] + r'$<\tau\leq$' + '%d Gyr' % ages[i + 1])
                
                ax2.set_xlabel(r'[Fe/H]')
                ax2.set_ylabel(r'W (km/s)')
                ax2.set_title('Gaussian Mixture: %d' % ages[i] + r'$<\tau\leq$' + '%d Gyr' % ages[i + 1])
                
                plt.savefig('%s/GM_%2d_t_%2d.png' % (plot_dir, ages[i], ages[i + 1]),
                            dpi=100, bbox_inches='tight')
                plt.close('all')
        self.gms = gms
        self.ts = ages

    def test_GMM_fit(self, Ntests, plot_dir, npeaks=1, plot_mcmc_prog=True, norm_adds=1):
        """
        Test the GM fitting for a subset of GALAH with a known age distribution

        Parameters
        ----------
        Ntests: int
            Number of total tests to conduct. This will always start with
            uniform distribution and then add npeaks from youngest to
            oldest ages

        plot_dir: str
            Directory to store the plots

        npeaks: int
            Number of peaks in test distributions

        plot_mcmc_prog: bool
            Plot the overall progress of emcee

        norm_adds: int
            Dont need to change this

        Returns
        -------
        sig_all_2: list
            Number of sigma all datapoints are from true value
        """
        sig_all_2 = []

        for nadds in range(Ntests):
            idx = []
            
            f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
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

                MH_norm = np.array(self.cat.loc[idx,'fe_h'])
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
            plt.close('all')

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

                plt.close('all')
        return sig_all_2


def action_angle(ra, dec, distance, pm_ra, pm_dec, v_hel):
    """
    approximate actions
    """
    inps = []
    for i in range(len(ra)):
        inps.append([ra[i], dec[i], distance[i], pm_ra[i], pm_dec[i], v_hel[i]])
    o = Orbit(vxvv=inps, radec=True)
    delta = 0.45
    aAS = actionAngleStaeckel(pot=MWPotential2014, delta=delta, c=True)
    actions = aAS(o.R() / 8., o.vR() / 220., o.vT() / 220., o.z() / 8., o.vz() / 220., fixed_quad=True)
    return actions[0], actions[1], actions[2]


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


def assign_equal_number_grid(x, y):
    """
    assign the equal number griding
    """
    xnorm = (x - x.min()) / (x.max() - x.min())
    ynorm = (y - y.min()) / (y.max() - y.min())
    tree = cKDTree(np.column_stack((xnorm, ynorm))) # let points be an array of shape (n,2)
    groups = []
    depth = 7 # depth of the tree, ie: depth=5 results in 64 bins
    def recurse(node, i=0):
        g = node.greater
        l = node.lesser
        if(i == depth):
            groups.append(g.data_points)
            groups.append(l.data_points)
            return
        recurse(g, i=i+1)
        recurse(l, i=i+1)

    recurse(tree.tree)

    return groups


def LZ0_FeH(t, FeH):
    """
    get the intial LZ of a star based on age
    and metallicity
    """
    delta_inner = -0.03
    delta_FeH = -0.093
    ft = (1 - t / 12) ** 0.45
    Fe_H_max = 0.7
    LZ0s = np.zeros(len(t))
    ev_divide = (FeH > Fe_H_max * ft + 3 * delta_inner)
    LZ0s[ev_divide] = (235 / delta_inner) * (FeH[ev_divide] - Fe_H_max * ft[ev_divide])
    LZ0s[~ev_divide] = (235 / delta_FeH) * (FeH[~ev_divide] - Fe_H_max * ft[~ev_divide] - 3 * (delta_inner - delta_FeH))
    return LZ0s
    ### this is the old version of the inversion thats wrong
    # if LZ0prob / 235 < 3:
    #     delta_FeH_LZ = delta_inner
    # else:
    #     delta_FeH_LZ = delta_FeH
    # ft = (1 - t / 12) ** 0.45
    # Fe_H_max = 0.7
    # if LZ0prob / 235 < 3:
    #     b_Fe_H = 0.
    #     return (FeH  - Fe_H_max * ft - b_Fe_H) * (235. / delta_FeH_LZ)
    # else:
    #     b_Fe_H = (delta_inner - delta_FeH) * 3 
    #     return (FeH  - Fe_H_max * ft - b_Fe_H) * (235. / delta_FeH_LZ)
    
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
                        R_birth_bins, N_samp, N):
    """
    bootstrap the probably birth radius distribution of a group of stars
    based on metallicity, age and current LZ distributions
    """
    mhbins_mid = np.array([(mhbins[i] + mhbins[i + 1]) / 2 for i in range(len(mhbins) - 1)])
    age_bins_mid = np.array([(age_bins[i] + age_bins[i + 1]) / 2 for i in range(len(age_bins) - 1)])

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

        LZ0_samp = LZ0_FeH(age_samp, mh_samp)

        R_birth = LZ0_samp / 248.5

        ns[i, :], _ = np.histogram(R_birth, bins=R_birth_bins)
        ns[i, :] /= np.sum(ns[i, :])
    return np.mean(ns, axis=0), np.std(ns, axis=0)


def loop(KM_metals, cs, i):
    """
    multiprocessing loops for calculating actions
    """
    maxx = i + cs - 1
    if maxx > len(KM_metals) - 1:
        maxx = len(KM_metals) - 1
    ra = np.array(KM_metals.loc[i: maxx,'RA']) # [deg]
    dec = np.array(KM_metals.loc[i: maxx,'DEC']) # [deg]
    distance = np.array(1/KM_metals.loc[i: maxx,'plx']) #kpc
    pm_ra = np.array(KM_metals.loc[i: maxx,'pmra']) # [mas/yr]
    pm_dec = np.array(KM_metals.loc[i: maxx,'pmde']) # [mas/yr]
    v_hel = np.array(KM_metals.loc[i: maxx,'rv']) # [km /s]
    JR,LZ,JZ = action_angle(ra, dec, distance, pm_ra, pm_dec, v_hel)
    return [JR, LZ, JZ]


class KM_metals(object):
    """
    Find probable age distributions for KM stars in SN
    """
    def __init__(self,  metals_file, gaia_file, gaia_dr2_match_file,
                 Rsun=8100., zsun=21., vphi_sun=248.5, vLSR=235.,
                 metal_err_cut=0.3):
        self.metals_file = metals_file
        self.gaia_file = gaia_file
        self.gaia_dr2_match_file = gaia_dr2_match_file
        self.Rsun = Rsun
        self.zsun = zsun
        self.vphi_sun = vphi_sun
        self.vLSR = vLSR
        self.metal_err_cut = metal_err_cut

        # load the data
        self.KM_metals = self.load_KM_data()

        # define the polygon regions used in the analysis
        self.polygons = self.init_polygons()

        # set names of kinematic groups
        self.names = ['A1/A2', r'$\gamma$Leo', 'Sirius','Coma Berenices','Dehnen98/Wolf630','Hyades','Pleiades',
                      'Hercules 1', 'Hercules 2', 'Hercules 3 (HR1614)', 
                      'g24 (Herc. 1)', 'g28 (Herc. 2)']

    def load_KM_data(self):
        df1 = pd.read_csv(self.metals_file, delim_whitespace=True)

        gaia1 = pd.read_csv(self.gaia_file,
                            usecols=[0, 2, 4, 5, 7, 9, 11, 12, 13, 14, 15, 18],
                            names=['RA', 'DEC', 'ID', 'plx', 'pmra',
                                   'pmde', 'G', 'BP', 'RP', 'rv', 'rv_err', 'ruwe'],
                            skiprows=1)

        gaia_dr2 = pd.read_csv(self.gaia_dr2_match_file,
                               names=['ID', 'dr2_ID', 'ang_dist'],
                               skiprows=1,
                               dtype={'ID': 'Int64', 'dr2_ID': 'Int64', 'ang_dist': 'float64'})
        gaia_dr2 = gaia_dr2.sort_values(by=['ID', 'ang_dist'])
        gaia_dr2 = gaia_dr2.drop_duplicates(subset='ID', keep='first')
        gaia1 = gaia1.merge(gaia_dr2, on='ID', how='left')
        
        # grab the Gaia data
        df1 = df1.rename(columns={"Gaia_ID": "ID"})
        df1 = df1.merge(gaia1, on='ID', how='inner')

        del gaia1

        KM_metals = df1[(df1['M_H_std'] * 1.95 < self.metal_err_cut) & (df1['ruwe'] <= 1.4)]

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

        R_G = (KM_metals['R'] * (KM_metals['gv'] + self.vphi_sun)) / self.vLSR
        phi = np.arcsin(KM_metals['dist'] * 1e-3 * np.sin(np.radians(l)) / KM_metals['R'])
        KM_metals['xmix'] = R_G * np.cos(phi)

        cs = 10000
        with Pool(processes=4) as pool:
            r = pool.map(partial(loop, KM_metals, cs), range(0, len(KM_metals), cs))
        r1 = []
        r2 = []
        r3 = []
        for ri in r:
            r1 += list(ri[0])
            r2 += list(ri[1])
            r3 += list(ri[2])

        r = np.column_stack((r1, r2, r3))
        KM_metals['J_R'] = r[:,0]
        KM_metals['L_Z'] = r[:,1]
        KM_metals['J_Z'] = r[:,2]

        return KM_metals

    def init_polygons(self):
        """
        function to store the polygons used to initially
        define kinematic groups
        """
        polygons = []
        # A1/A2
        ellipse = Polygon(xy=np.column_stack(([-23.2, -60, -60, 30, 15, -3],
                                              np.array([8.5, 8.75, 9.25, 9.25, 8.7, 8.7]) * 248.5 / 235)),
                          edgecolor='r', fc='None', lw=2)
        polygons.append(ellipse)
        # gamma Leo
        ellipse = Polygon(xy=np.column_stack(([15, 60, 60, 46.5],
                                              np.array([8.7, 8.7, 8.2, 7.9]) * 248.5 / 235)),
                          edgecolor='r', fc='None', lw=2)
        polygons.append(ellipse)
        #sirius stream
        ellipse = Polygon(xy=np.column_stack(([-3, 15, 46.5, 32.5, -23.2],
                                              np.array([8.7, 8.7, 7.9, 7.6, 8.5]) * 248.5 / 235)),
                          edgecolor='r', fc='None', lw=2)
        polygons.append(ellipse)
        #coma stream
        ellipse = Polygon(xy=np.column_stack(([-23.2, 32.5, 27.5, 24.5, -34.3],
                                              np.array([8.5, 7.6, 7.2, 7.1, 8]) * 248.5 / 235)),
                                edgecolor='r', fc='None', lw=2)
        polygons.append(ellipse)
        #horn???
        ellipse = Polygon(xy=np.column_stack(([32.5, 27.5, 54,59.5, 46.2],
                                              np.array([7.6, 7.2, 7, 7.2, 7.5]) * 248.5 / 235)),
                          edgecolor='r', fc='None', lw=2)
        polygons.append(ellipse)
        #hyades
        ellipse = Polygon(xy=np.column_stack(([-34.3, -12, -31.2, -53.4, -55.2],
                                              np.array([8, 7.65, 7.2, 7.3, 7.7]) * 248.5 / 235)),
                          edgecolor='r', fc='None', lw=2)
        polygons.append(ellipse)
        #pleides
        ellipse = Polygon(xy=np.column_stack(([-12, 12, 0, -31.2],
                                              np.array([7.65, 7.29, 6.8, 7.2]) * 248.5 / 235)),
                          edgecolor='r', fc='None', lw=2)
        polygons.append(ellipse)
        #hercules 1
        ellipse = Polygon(xy=np.column_stack(([0, -11.8, -54.5, -54.5, -31.2],
                                              np.array([6.8, 6.7, 6.7, 7, 7.2]) * 248.5 / 235)),
                          edgecolor='r', fc='None', lw=2)
        polygons.append(ellipse)
        #hercules 2
        ellipse = Polygon(xy=np.column_stack(([-11.8, 17.3, 17.3, -65, -65, -54.5],
                                              np.array([6.7, 6.6, 6.3, 6.3, 6.7, 6.7]) * 248.5 / 235)),
                          edgecolor='r', fc='None', lw=2)
        polygons.append(ellipse)
        #hercules 3
        ellipse = Polygon(xy=np.column_stack(([17.3, -57.7, -51, 40],
                                              np.array([6.3, 6, 5.5, 6.1]) * 248.5 / 235)),
                          edgecolor='r', fc='None', lw=2)
        polygons.append(ellipse)
        #???
        ellipse = Polygon(xy=np.column_stack(([-54.5, -54.5, -73.8, -74.8],
                                              np.array([6.7, 7, 7.1, 6.7]) * 248.5 / 235)),
                          edgecolor='r', fc='None', lw=2)
        polygons.append(ellipse)
        #???
        ellipse = Polygon(xy=np.column_stack(([-65, -65, -74.8, -94, -93],
                                              np.array([6.3, 6.7, 6.7, 6.7, 6.3]) * 248.5 / 235)),
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

        # file to write table data
        f = open('summary_table_data.txt', 'w')

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
                plt.close('all')
            self.id2 += list(self.stream_dfs['%d' % i]['ID'][self.stream_dfs['%d' % i]['group_pca_sig'] <= 2])

            comps = []
            As = []
            line = [self.names[i]]
            for length, vector in zip(pca.explained_variance_, pca.components_):
                v = vector * 3 * np.sqrt(length)
                comps.append(np.sqrt(v[0] ** 2 + v[1] ** 2))
                As.append(np.arctan(v[1] / v[0]) * (180 / np.pi))
            # write the line for the table
            line.append(pca.inverse_transform([mean1,mean2])[0])
            line.append(pca.inverse_transform([mean1,mean2])[1])
            line.append(comps[0])
            line.append(comps[1])
            line.append(As[0]*-1)
            line.append(len(self.stream_dfs['%d' % i][self.stream_dfs['%d' % i]['group_pca_sig']==1.]))
            line.append(len(self.stream_dfs['%d' % i][self.stream_dfs['%d' % i]['group_pca_sig']==2.]))
            line.append(len(self.stream_dfs['%d' % i][self.stream_dfs['%d' % i]['group_pca_sig']==3.]))
            f.write('%s & %.3f & %.3f & %.3f & %.3f & %.3f & %d & %d & %d \n' % tuple(line))
            self.mean1.append(pca.inverse_transform([mean1, mean2])[0])
            self.mean2.append(pca.inverse_transform([mean1, mean2])[1])
            self.std1.append(comps[0])
            self.std2.append(comps[1])
            self.angles.append(As[0])
        f.close()
