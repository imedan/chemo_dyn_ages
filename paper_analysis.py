import numpy as np
from gm_age_relation import (GM_Age_GALAH, KM_metals, 
                             compare_all_spec_surveys,
                             assign_equal_number_grid,
                             mcmc_GM_fit, bootstrap_hist,
                             bootstrap_Rbirth_LZ)
from scipy.optimize import curve_fit
from astropy.coordinates import SkyCoord
import astropy.units as u


def bootstrap_median(x ,N):
    """
    bootstrap the median of sample x over N iterations
    """
    if len(x) > 10:
        means = np.zeros(N)
        for i in range(N):
            means[i] = np.nanmedian(x[np.random.randint(len(x), size=len(x))])
        return np.mean(means), np.std(means)
    else:
        return np.nan, np.nan


def line(x, A, B):
    return A * x + B


def bootstrap_mean(x ,N):
    """
    bootstrap the median of sample x over N iterations
    """
    if len(x) > 5:
        means = np.zeros(N)
        for i in range(N):
            means[i] = np.nanmean(x[np.random.randint(len(x), size=len(x))])
        return np.mean(means), np.std(means)
    else:
        return np.nan, np.nan


if __name__ == '__main__':
    # initialze the GALAH training
    # need to download galah dr3 files (GALAH_DR3_main_allstar_v2.fits,
    # GALAH_DR3_VAC_ages_v2.fits and GALAH_DR3_VAC_GaiaEDR3_v2.fits)
    # from here: https://cloud.datacentral.org.au/teamdata/GALAH/public/GALAH_DR3/
    GM = GM_Age_GALAH('galah_data/')

    # get the GM models from GALAH
    GM.find_GMMs(np.arange(0, 16, 2), plot_dir='paper_plots')

    # do the 2 peak tests
    sig_2 = GM.test_GMM_fit(6, 'paper_plots', npeaks=2, plot_mcmc_prog=False)

    # do the 1 peak tests
    sig_1 = GM.test_GMM_fit(8, 'paper_plots', npeaks=1, plot_mcmc_prog=False)

    # initialize the class with the K/M dwarfs

    # gaia_file can be queried from the archive via:
    # SELECT g3.ra, g3.ra_error, g3.dec, g3.dec_error, g3.source_id, g3.parallax,
    #        g3.parallax_error, g3.pmra, g3.pmra_error, g3.pmdec, g3.pmdec_error,
    #        g3.phot_g_mean_mag, g3.phot_bp_mean_mag, g3.phot_rp_mean_mag, g3.radial_velocity,
    #        g3.radial_velocity_error, g3.rv_template_teff, g3.rv_template_fe_h, g3.ruwe
    # FROM gaiadr3.gaia_source as g3
    # WHERE g3.phot_g_mean_mag <= 14 AND g3.phot_rp_mean_mag <= 14 AND g3.phot_bp_mean_mag -
    #       g3.phot_rp_mean_mag > 0.98 AND g3.phot_bp_mean_mag - g3.phot_rp_mean_mag < 2.39 AND
    #       g3.parallax > 0 AND g3.phot_g_mean_mag + 5 * log10(0.001 * g3.parallax) + 5 > 4

    # gaia_dr2_match_file can be queried from the archive via:
    # SELECT g3.source_id, gneigh.dr2_source_id, gneigh.angular_distance
    # FROM gaiadr3.gaia_source as g3
    # JOIN gaiadr3.dr2_neighbourhood AS gneigh ON g3.source_id = gneigh.dr3_source_id
    # WHERE g3.phot_g_mean_mag <= 14 AND g3.phot_rp_mean_mag <= 14 AND g3.phot_bp_mean_mag -
    #       g3.phot_rp_mean_mag > 0.98 AND g3.phot_bp_mean_mag - g3.phot_rp_mean_mag < 2.39 AND
    #       g3.parallax > 0 AND g3.phot_g_mean_mag + 5 * log10(0.001 * g3.parallax) + 5 > 4
    KM = KM_metals(metals_file='gaia_DR3_photo_metallicities.txt',
                   gaia_file='gaia_file/K_dwarf_RV_DR3-result.csv',
                   gaia_dr2_match_file='gaia_file/K_dwarf_RV_DR3_w_DR2_match-result.csv')

    # comapre photo metallicities to those in spectroscopic surveys
    # download galah dr3 from: https://cloud.datacentral.org.au/teamdata/GALAH/public/GALAH_DR3/
    # download apogee dr16 from: https://dr16.sdss.org/sas/dr16/apogee/spectro/aspcap/r12/l33/allStarLite-r12-l33.fits
    # download apogee dr14 data from: https://data.sdss.org/sas/dr16/apogee/spectro/aspcap/r12/l33/allStar-r12-l33.fits
    galah_file = 'galah_data/GALAH_DR3_main_allstar_v2.fits'
    apogee_16_file = '../500_pc_KM_rv_cross_match/APOGEE_DR16_allStarLite-r12-l33.fits'
    apogee_14_file = '../500_pc_KM_rv_cross_match/APOGEE_DR14_allStar-l31c.2.fits'
    compare_all_spec_surveys(galah_file, apogee_16_file, apogee_14_file,
                             KM.KM_metals, 'paper_plots/compare_all_spec_metals.png')

    # assign the kinemtic groups
    KM.assign_kinematic_groups(plot_groups=False)

    # get the gridding in x_mix vs U
    x = np.array(KM.KM_metals['gu'])
    y = np.array(KM.KM_metals['xmix'])
    groups = assign_equal_number_grid(x, y)

    # calculate the age distribution on the grid
    ts = np.arange(0, 16, 2)

    bounds = np.zeros((len(groups), 4))
    age_frac = np.zeros((len(groups), len(ts) - 1))
    age_err = np.zeros((len(groups), len(ts) - 1))

    for i, group in enumerate(groups):
        xmin = min(group[:, 0] * (x.max() - x.min()) + x.min())
        xmax = max(group[:, 0] * (x.max() - x.min()) + x.min())
        ymin = min(group[:, 1] * (y.max() - y.min()) + y.min())
        ymax = max(group[:, 1] * (y.max() - y.min()) + y.min())
        
        bounds[i, :] = np.array([xmin, xmax, ymin, ymax])
        
        ev_group = ((KM.KM_metals['gu'] >= xmin) & (KM.KM_metals['gu'] <= xmax) &
                    (KM.KM_metals['xmix'] >= ymin) & (KM.KM_metals['xmix'] <= ymax))
        xs = (np.array(KM.KM_metals['M_H'])[ev_group] -
              (-0.03) * (np.array(KM.KM_metals['xmix'])[ev_group] - 8.1) - 0.188)
        ys = np.array(KM.KM_metals['gw'])[ev_group]
        sampler, flat_samples = mcmc_GM_fit(xs, ys, GM.gms, ts, progress=False)

        mcmc = np.percentile(flat_samples, [16, 50, 84], axis=0)
        q = np.diff(mcmc, axis=0)
        
        age_frac[i, :] = mcmc[1, :]
        age_err[i, :] = q[1, :]

    # save the results
    np.save('age_grid_kde/bounds.npy', bounds)
    np.save('age_grid_kde/age_frac.npy', age_frac)
    np.save('age_grid_kde/age_err.npy', age_err)

    # calculate the birth radii distributions
    Rb_bins = np.arange(0, 22, 2)  
    Rb_bins_mid = np.array([(Rb_bins[i] + Rb_bins[i + 1]) / 2 for i in range(len(Rb_bins) - 1)])

    Rb_frac = np.zeros((len(bounds), len(Rb_bins_mid)))
    Rb_err = np.zeros((len(bounds), len(Rb_bins_mid)))

    for i in range(len(bounds)):
        xmin = bounds[i][0]
        xmax = bounds[i][1]
        ymin = bounds[i][2]
        ymax = bounds[i][3]
        
        ev_group = ((KM.KM_metals['gu'] >= xmin) & (KM.KM_metals['gu'] <= xmax) &
                    (KM.KM_metals['xmix'] >= ymin) & (KM.KM_metals['xmix'] <= ymax))
        
        mh_bins = np.arange(-1.5, 0.6, 0.1)
        mh_mids = np.array([(mh_bins[i] + mh_bins[i + 1]) / 2 for i in range(len(mh_bins) - 1)])
        mh_stream1, mh_err_stream1 = bootstrap_hist(np.array(KM.KM_metals['M_H'])[ev_group],
                                                    mh_bins, 1000)

        Rb_frac[i, :], Rb_err[i, :] = bootstrap_Rbirth_LZ(mh_stream1, mh_err_stream1, mh_bins,
                                                          age_frac[i, :], age_err[i, :],
                                                          ts, Rb_bins,
                                                          len(np.array(KM.KM_metals['L_Z'])[ev_group]), 1000)

    # save the results
    np.save('age_grid_kde/Rb_frac.npy', Rb_frac)
    np.save('age_grid_kde/Rb_err.npy', Rb_err)

    # calculate the bending and breathing params
    sc = SkyCoord(ra=np.array(KM.KM_metals['RA']) * u.deg,
                  dec=np.array(KM.KM_metals['DEC'])* u.deg)
    l = np.array(sc.galactic.l)
    b = np.array(sc.galactic.b)

    A = np.zeros(len(bounds))
    B = np.zeros(len(bounds))

    Aerr = np.zeros(len(bounds))
    Berr = np.zeros(len(bounds))

    zs = np.linspace(-500, 500, 34)
    z_mids = np.array([(zs[i] + zs[i+1]) / 2 for i in range(len(zs) - 1)])

    for i in range(len(bounds)):
        xmin = bounds[i][0]
        xmax = bounds[i][1]
        ymin = bounds[i][2]
        ymax = bounds[i][3]
        
        ev_group = ((KM.KM_metals['gu'] >= xmin) & (KM.KM_metals['gu'] <= xmax) &
                    (KM.KM_metals['xmix'] >= ymin) & (KM.KM_metals['xmix'] <= ymax) &
                    (l > 50) & (l < 200))

        w_means = np.zeros(len(zs) - 1)
        w_stds = np.zeros(len(zs) - 1)
        for j in range(len(zs) - 1):
            ev_z = (KM.KM_metals['gz'] > zs[j]) & (KM.KM_metals['gz'] <= zs[j + 1])
            w_means[j], w_stds[j] = bootstrap_median(
                np.array(KM.KM_metals['gw'][ev_z & ev_group] + 7.25), 1000)
        
        popt, pcov = curve_fit(line, z_mids[~np.isnan(w_means)] / 1000,
                               w_means[~np.isnan(w_means)], p0=(2, -1),
                               sigma=w_stds[~np.isnan(w_means)], absolute_sigma=True)
        A[i] = popt[0]
        Aerr[i] = np.sqrt(np.diag(pcov))[0]
        B[i] = popt[1]
        Berr[i] = np.sqrt(np.diag(pcov))[1]

    # get the average params with age
    A_means = np.zeros(len(age_frac[0, :]))
    A_stds = np.zeros(len(age_frac[0, :]))
    for i in range(len(age_frac[0, :])):
        A_means[i], A_stds[i] = bootstrap_mean(abs(A[age_frac[:, i] > 0.5]),
                                               1000)

    B_means = np.zeros(len(age_frac[0, :]))
    B_stds = np.zeros(len(age_frac[0, :]))
    for i in range(len(age_frac[0, :])):
        B_means[i], B_stds[i] = bootstrap_mean(abs(B[age_frac[:, i] > 0.5]),
                                               1000)
    
