from cross_match import PANSTARRS1
from dustmaps.bayestar import BayestarWebQuery
from uncertainties import ufloat
from uncertainties.umath import log10
from astropy.coordinates import ICRS, SkyCoord
import astropy.units as u
from typing import Tuple
import numpy as np
import pandas as pd

bq = BayestarWebQuery()

# Extinction vectors from https://arxiv.org/pdf/1905.02734.pdf for 2MASS and PanStarrs1

# Multiply the E(B_V) by 3.1 and use the following relationships for Gaia bands:
# G: 0.83627
# G_BP: 1.08337
# G_RP: 0.63439

# PanStarrs1
# g: 3.518
# r: 2.617
# i: 1.971
# z: 1.549
# y: 1.263

# 2MASS
# J: 0.7927
# H: 0.4690
# K: 0.3026


EXTINCTION_VECTOR = {
    'G': 0.83627,
    'BP': 1.08337,
    'RP': 0.63439,
    'g': 3.518,
    'r': 2.617,
    'i': 1.971,
    'z': 1.549,
    'y': 1.263,
    'J': 0.7927,
    'H': 0.4690,
    'K': 0.3026
}

GAIA_PASSBANDS = ['G', 'BP', 'RP']
PANSTARRS1_PASSBANDS = ['g', 'r', 'i', 'z', 'y']
TWOMASS_PASSBANDS = ['J', 'H', 'K']
ALL_PASSBANDS = GAIA_PASSBANDS+PANSTARRS1_PASSBANDS+TWOMASS_PASSBANDS


def extinction(b_v: np.float32, passband: str) -> np.float32:
    if passband in GAIA_PASSBANDS:
        return 3.1*EXTINCTION_VECTOR[passband]*b_v*0.981
    elif passband in PANSTARRS1_PASSBANDS or passband in TWOMASS_PASSBANDS:
        return 3.1*EXTINCTION_VECTOR[passband]*b_v
    else:
        raise ValueError(f'Passband must be one of passbands: {ALL_PASSBANDS}!')

        
def extinction_coefficient(ra: np.float32, dec: np.float32, distance: np.float32) -> np.float32:
    return bq.query(coords=SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg),
                                    distance=distance*u.kpc, frame=ICRS), mode='best')


def add_color(sources: pd.DataFrame, color1_name: str, color2_name: str) -> pd.DataFrame:
    sources[f'{color1_name}-{color2_name}'] = sources[color1_name]-sources[color2_name]\
                                              -sources[f'A_{color1_name}']+sources[f'A_{color2_name}']
    sources[f'{color1_name}-{color2_name}_error'] = np.sqrt(np.power(sources[f'{color1_name}_error'].values, 2)+np.power(sources[f'{color2_name}_error'].values, 2))
    return sources


def mag_abs(sources: pd.DataFrame, color_name: str) -> pd.DataFrame:
        #G = g - 10. + 5.*np.log10(par) - A_G  
    sources[f'{color_name}_abs'] = sources[color_name] - 10. + 5.*np.log10(sources['parallax'].values) - sources[f'A_{color_name}'].values
    
    # eG = np.sqrt(eg*eg + (5./np.log(10)*pare/par)**2.)
    sources[f'{color_name}_abs_error'] = np.sqrt(np.power(sources[f'{color_name}_error'].values, 2) + np.power((5./np.log(10))*(1/sources['parallax_over_error'].values), 2))
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


def add_colors_and_abs_mag(sources: pd.DataFrame) -> pd.DataFrame:
    sources['BP_error'] = (2.5/(np.log(10)*sources['phot_bp_mean_flux_over_error'])).astype(np.float32)
    sources['RP_error'] = (2.5/(np.log(10)*sources['phot_rp_mean_flux_over_error'])).astype(np.float32)
    sources['G_error'] = (2.5/(np.log(10)*sources['phot_g_mean_flux_over_error'])).astype(np.float32)

    sources = sources.rename(columns={'phot_g_mean_mag': 'G',
                                      'phot_bp_mean_mag': 'BP',
                                      'phot_rp_mean_mag': 'RP'})
    
    sources['E(B_V)'] = extinction_coefficient(sources.ra.values,
                                               sources.dec.values,
                                               1/sources.parallax.values)
    
    for passband in ALL_PASSBANDS:
    
        sources[f'A_{passband}'] = extinction(sources['E(B_V)'].values, passband)
    

    sources = add_color(sources, 'BP', 'RP')
    sources = mag_abs(sources, 'G')
    
    sources['excess'] = correct_flux_excess_factor(sources['BP-RP'], sources['phot_bp_rp_excess_factor'])
    
    return sources



