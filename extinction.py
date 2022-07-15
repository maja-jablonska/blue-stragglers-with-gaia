from dustmaps.bayestar import BayestarWebQuery
from uncertainties import ufloat
from uncertainties.umath import log10
from astropy.coordinates import SkyCoord
import astropy.units as u
from typing import Tuple
import numpy as np
import pandas as pd

bq = BayestarWebQuery()

# Extinction vectors from https://arxiv.org/pdf/1905.02734.pdf for 2MASS

# Multiply the E(B_V) by 3.1 and use the following relationships for Gaia bands:
# G: 0.83627
# G_BP: 1.08337
# G_RP: 0.63439


GAIA_EXTINCTION_VECTOR = {
    'G': 0.83627,
    'G_BP': 1.08337,
    'G_RP': 0.63439
}


def absolute_magnitude(g_mean_mag: np.float32,
                       g_mean_mag_err: np.float32,
                       g_ext: np.float32,
                       parallax: np.float32,
                       parallax_err: np.float32) -> Tuple[np.float32, np.float32]:
    value = (ufloat(g_mean_mag, g_mean_mag_err)+
             5*log10(ufloat(parallax, parallax_err))-10-g_ext)
    return value.nominal_value, value.std_dev


def color(bp_mean_mag: np.float32,
          bp_mean_mag_err: np.float32,
          bp_extinction: np.float32,
          rp_mean_mag: np.float32,
          rp_mean_mag_err: np.float32,
          rp_extinction: np.float32) -> Tuple[np.float32, np.float32]:
    value = ((ufloat(bp_mean_mag, bp_mean_mag_err)-bp_extinction)-
             (ufloat(rp_mean_mag, rp_mean_mag_err)-rp_extinction))
    return value.nominal_value, value.std_dev


def gaia_extinction(b_v: np.float32, passband: str) -> np.float32:
    try:
        return 3.1*GAIA_EXTINCTION_VECTOR[passband]*b_v
    except KeyError:
        raise ValueError(f'Passband must be one of Gaia passbands: {GAIA_PASSBANDS}!')

        
def extinction_coefficient(ra: np.float32, dec: np.float32, distance: np.float32) -> np.float32:
    return bq.query(coords=SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), distance=distance*u.kpc), mode='best')


def add_colors_and_abs_mag(sources: pd.DataFrame) -> pd.DataFrame:
    sources['BP_err'] = 2.5/(np.log(10)*sources['phot_bp_mean_flux_over_error'])
    sources['RP_err'] = 2.5/(np.log(10)*sources['phot_rp_mean_flux_over_error'])
    sources['G_err'] = 2.5/(np.log(10)*sources['phot_g_mean_flux_over_error'])

    sources = sources.rename(columns={'phot_g_mean_mag': 'G',
                                      'phot_bp_mean_mag': 'BP',
                                      'phot_rp_mean_mag': 'RP'})
    
    sources['B(E_V)'] = extinction_coefficient(sources.ra.values,
                                               sources.dec.values,
                                               sources.parallax.values)
    sources['A_G'] = gaia_extinction(sources['B(E_V)'].values, 'G')
    sources['A_BP'] = gaia_extinction(sources['B(E_V)'].values, 'G_BP')
    sources['A_RP'] = gaia_extinction(sources['B(E_V)'].values, 'G_RP')
    
    sources['color'], sources['color_err'] = np.vectorize(color)(sources['BP'].values,
                                                                 sources['BP_err'].values,
                                                                 sources['A_BP'].values,
                                                                 sources['RP'].values,
                                                                 sources['RP_err'].values,
                                                                 sources['A_RP'].values)
    
    sources['mag_abs'], sources['mag_abs_err'] = np.vectorize(absolute_magnitude)(sources['G'].values,
                                                                                  sources['G_err'].values,
                                                                                  sources['A_G'].values,
                                                                                  sources['parallax'].values,
                                                                                  sources['parallax_error'].values)
    return sources


def correct_flux_excess_factor(bp_rp, phot_bp_rp_excess_factor):
    """
    Calculate the corrected flux excess factor for the input Gaia EDR3 data.
    
    Parameters
    ----------
    
    bp_rp: float, numpy.ndarray
        The (BP-RP) colour listed in the Gaia EDR3 archive.
    phot_bp_rp_excess_factor: float, numpy.ndarray
        The flux excess factor listed in the Gaia EDR3 archive.
        
    Returns
    -------
    
    The corrected value for the flux excess factor, which is zero for "normal" stars.
    
    Example
    -------
    
    phot_bp_rp_excess_factor_corr = correct_flux_excess_factor(bp_rp, phot_bp_rp_flux_excess_factor)
    """
    
    if np.isscalar(bp_rp) or np.isscalar(phot_bp_rp_excess_factor):
        bp_rp = np.float64(bp_rp)
        phot_bp_rp_excess_factor = np.float64(phot_bp_rp_excess_factor)
    
    if bp_rp.shape != phot_bp_rp_excess_factor.shape:
        raise ValueError('Function parameters must be of the same shape!')
        
    do_not_correct = np.isnan(bp_rp)
    bluerange = np.logical_not(do_not_correct) & (bp_rp < 0.5)
    greenrange = np.logical_not(do_not_correct) & (bp_rp >= 0.5) & (bp_rp < 4.0)
    redrange = np.logical_not(do_not_correct) & (bp_rp > 4.0)
    
    correction = np.zeros_like(bp_rp)
    correction[bluerange] = 1.154360 + 0.033772*bp_rp[bluerange] + 0.032277*np.power(bp_rp[bluerange], 2)
    correction[greenrange] = 1.162004 + 0.011464*bp_rp[greenrange] + 0.049255*np.power(bp_rp[greenrange], 2) \
        - 0.005879*np.power(bp_rp[greenrange], 3)
    correction[redrange] = 1.057572 + 0.140537*bp_rp[redrange]
    
    return phot_bp_rp_excess_factor - correction
