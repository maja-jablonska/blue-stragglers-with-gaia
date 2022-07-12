from dustmaps.bayestar import BayestarWebQuery
from uncertainties import ufloat
from uncertainties.umath import log10
from astropy.coordinates import SkyCoord
import astropy.units as u
from typing import Tuple
import numpy as np

bq = BayestarWebQuery()

# Extinction vectors from https://arxiv.org/pdf/1905.02734.pdf for 2MASS
    

EXTINCTION_VECTOR = {
    'H': 0.4690,
    'J': 0.7927,
    'Ks': 0.3026
}


def absolute_magnitude(g_mean_mag: np.float32,
                       g_mean_mag_err: np.float32,
                       g_ext: np.float32,
                       g_ext_err: np.float32,
                       parallax: np.float32,
                       parallax_err: np.float32) -> Tuple[np.float32, np.float32]:
    value = (ufloat(g_mean_mag, g_mean_mag_err)+
             5*log10(ufloat(parallax, parallax_err))-10-ufloat(g_ext, g_ext_err))
    return value.nominal_value, value.std_dev


def color(bp_mean_mag: np.float32,
          bp_mean_mag_err: np.float32,
          rp_mean_mag: np.float32,
          rp_mean_mag_err: np.float32,
          reddening: np.float32,
          reddening_err: np.float32) -> Tuple[np.float32, np.float32]:
    value = (ufloat(bp_mean_mag, bp_mean_mag_err)-ufloat(rp_mean_mag, rp_mean_mag_err)-
             ufloat(reddening, reddening_err))
    return value.nominal_value, value.std_dev


def extinction_in_passband(reddening_function: np.float32, passband: str) -> np.float32:
    return EXTINCTION_VECTOR[passband]*reddening_function


def gaia_B_RP_extinction(reddening_function: np.float32) -> Tuple[np.float32, np.float32]:
    h_ext = extinction_in_passband(reddening_function, 'H')
    ks_ext = extinction_in_passband(reddening_function, 'Ks')
    
    # TODO: add uncert
    # sig = 0.2361
    
    return 0.1836+8.456*(h_ext-ks_ext)-3.781*(h_ext-ks_ext)**2, 0.2361


def gaia_g_extinction(reddening_function: np.float32) -> Tuple[np.float32, np.float32]:
    h_ext = extinction_in_passband(reddening_function, 'H')
    ks_ext = extinction_in_passband(reddening_function, 'Ks')
    
    # sig = 0.08553
    
    return 0.5594+11.09*(h_ext-ks_ext)+3.040*(h_ext-ks_ext)**2+ks_ext, 0.08553


def extinction_coefficient(ra: np.float32, dec: np.float32, distance: np.float32) -> np.float32:
    return bq.query(coords=SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), distance=distance*u.kpc), mode='best')
