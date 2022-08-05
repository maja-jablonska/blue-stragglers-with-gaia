from numpy import column_stack
from astroquery.gaia import Gaia
import numpy as np
import pandas as pd
from typing import List

from astropy.coordinates import ICRS, SkyCoord
import astropy.units as u

Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
Gaia.ROW_LIMIT = -1 # Fetch all results


def format_coord(coord: float) -> str:
    two_dec: str = '{0:.2f}'.format(coord)
    with_sign = '+'+two_dec if coord>0 else two_dec
    return with_sign


def get_sources_by_random_index(rows: int = 1000, start_index: int = 0) -> pd.DataFrame:
    job = Gaia.launch_job(f'''
        SELECT source_id, ra, dec, parallax, parallax_error, 
        pmra, pmra_error, pmdec, pmdec_error 
        FROM gaiadr3.gaia_source 
        WHERE random_index > {start_index} AND random_index < {start_index+rows}
    ''', output_format='csv')

    return job.get_results().to_pandas()


def gaia_cone_search_5d(ra: float,
                        dec: float,
                        parallax: float,
                        pmra: float,
                        pmdec: float,
                        radvel: float,
                        radius: float,
                        min_parallax: float,
                        max_parallax: float) -> pd.DataFrame:
    """
        Perform a cone search in Gaia DR3 with proper motion propagation.
        :param ra: right ascension of the cluster center [degree]
        :param dec: declination of the cluster center [degree]
        :param parallax: parallax of the cluster center [mas]
        :param pmra: proper motion of the cluster center [mas/yr]
        :param pmdec: proper motion of the cluster center [mas/yr]
        :param radvel: radial velocity of the cluster center [km/s]
        :param radius: radius of the area we wish to search in [degrees]
        :param min_parallax: minimum parallax for sources [mas]
        :param max_parallax: maximum parallax for sources [mas]
        :return table of 5D data for objects in the area
    """
    
    ra_str: str = format_coord(ra)
    dec_str: str = format_coord(dec)
    
    print(f"Executing cone search for ra={ra_str} and dec={dec_str} with radius of {radius}...")
    
    query: str = f'''
        SELECT *
        FROM gaiadr3.gaia_source 
        WHERE 1 = CONTAINS( 
                POINT('ICRS', ra, dec), 
                CIRCLE('ICRS',
                    COORD1(EPOCH_PROP_POS({ra}, {dec}, {parallax}, {pmra}, {pmdec}, {radvel}, 2000, 2016.0)),
                    COORD2(EPOCH_PROP_POS({ra}, {dec}, {parallax}, {pmra}, {pmdec}, {radvel}, 2000, 2016.0)),
                {radius})) 
        AND parallax > {min_parallax} AND parallax < {max_parallax} 
    '''
        
    print('Executing query:')
    print(query)
    
    job = Gaia.launch_job_async(query, output_format='csv')

    print(f"Query finished!")
    
    return job.get_results().to_pandas()


def gaia_download_data(source_ids: np.ndarray) -> pd.DataFrame:
    """
        Download all columns from Gaia DR3 for given source ids.
        :param source_ids (np.ndarray): source_ids to download Gaia DR3 data for
        :return table of all Gaia DR3 columns
    """
    
    print(f"Executing Gaia query for {len(source_ids)} sources...")
    
    query: str = f'''
        SELECT *
        FROM gaiadr3.gaia_source 
        WHERE source_id IN ({', '.join([str(o) for o in source_ids])})
    '''
        
    print('Executing query...')
    
    job = Gaia.launch_job_async(query, output_format='csv')

    print(f"Query finished!")
    
    return job.get_results().to_pandas()


def __gaia_dr3_cross_match(source_ids: np.array, survey_name: str, max_ang_sep: float, verbose: bool) -> pd.DataFrame:
    
    query: str = f"""
      SELECT
      {survey_name}_best_neighbour.source_id, 
      {survey_name}_best_neighbour.original_ext_source_id, 
      {survey_name}_best_neighbour.angular_distance, 
      {survey_name}_best_neighbour.xm_flag 
      FROM gaiadr3.{survey_name}_best_neighbour 
      WHERE source_id IN ({', '.join([str(o) for o in source_ids])})
    """
    
    if verbose:
        print('Executing query:')
        print(query)
    
    job = Gaia.launch_job_async(query, output_format='csv')
    result: pd.DataFrame = job.get_results().to_pandas()
        
    result = result[(result.xm_flag<16) & (result.angular_distance<=max_ang_sep)]
    print(f'{survey_name} cross-match: {len(result)} sources')
    
    return result.drop(columns=[
        'xm_flag', 'angular_distance'
    ]).rename(columns={
        'original_ext_source_id': f'{survey_name}_id'
    })


def panstarrs1_cross_match(source_ids: np.array, verbose: bool = False) -> pd.DataFrame:
    """Return cross-matched sources for DR3 sources

    Args:
        source_ids (np.array): Gaia DR3 source ids
        verbose (bool): Print the executed query

    Returns:
        pd.DataFrame: Gaia DR3 results with best neighbours from panstarrs1_best_neighbour
    """

    return __gaia_dr3_cross_match(source_ids, 'panstarrs1', 1., verbose)
        



def allwise_cross_match(source_ids: np.array, verbose: bool = False) -> pd.DataFrame:
    """Return cross-matched sources for DR3 sources

    Args:
        source_ids (np.array): Gaia DR3 source ids
        verbose (bool): Print the executed query

    Returns:
        pd.DataFrame: Gaia DR3 results with best neighbours from panstarrs1_best_neighbour
    """

    return __gaia_dr3_cross_match(source_ids, 'allwise', 3., verbose)


def apassdr9_cross_match(source_ids: np.array, verbose: bool = False) -> pd.DataFrame:
    """Return cross-matched sources for DR3 sources

    Args:
        source_ids (np.array): Gaia DR3 source ids
        verbose (bool): Print the executed query

    Returns:
        pd.DataFrame: Gaia DR3 results with best neighbours from panstarrs1_best_neighbour
    """

    return __gaia_dr3_cross_match(source_ids, 'apassdr9', 3., verbose)


def twomass_cross_match(source_ids: np.array, verbose: bool = False) -> pd.DataFrame:
    """Return cross-matched sources for DR3 sources

    Args:
        source_ids (np.array): Gaia DR3 source ids
        verbose (bool): Print the executed query

    Returns:
        pd.DataFrame: Gaia DR3 results with best neighbours from panstarrs1_best_neighbour
    """
    
    return __gaia_dr3_cross_match(source_ids, 'tmass_psc_xsc', 3., verbose).rename(columns={
        'tmass_psc_xsc_id': 'twomass_id'
    })

def ravedr5_cross_match(source_ids: np.array, verbose: bool = False) -> pd.DataFrame:
    """Return cross-matched sources for DR3 sources

    Args:
        source_ids (np.array): Gaia DR3 source ids
        verbose (bool): Print the executed query

    Returns:
        pd.DataFrame: Gaia DR3 results with best neighbours from panstarrs1_best_neighbour
    """
    
    return __gaia_dr3_cross_match(source_ids, 'ravedr5', 1., verbose)


def ravedr6_cross_match(source_ids: np.array, verbose: bool = False) -> pd.DataFrame:
    """Return cross-matched sources for DR3 sources

    Args:
        source_ids (np.array): Gaia DR3 source ids
        verbose (bool): Print the executed query

    Returns:
        pd.DataFrame: Gaia DR3 results with best neighbours from panstarrs1_best_neighbour
    """
    
    return __gaia_dr3_cross_match(source_ids, 'ravedr6', 1., verbose)


def sdssdr13_cross_match(source_ids: np.array, verbose: bool = False) -> pd.DataFrame:
    """Return cross-matched sources for DR3 sources

    Args:
        source_ids (np.array): Gaia DR3 source ids
        verbose (bool): Print the executed query

    Returns:
        pd.DataFrame: Gaia DR3 results with best neighbours from panstarrs1_best_neighbour
    """
    
    return __gaia_dr3_cross_match(source_ids, 'sdssdr13', 1., verbose)


def skymapperdr2_cross_match(source_ids: np.array, verbose: bool = False) -> pd.DataFrame:
    """Return cross-matched sources for DR3 sources

    Args:
        source_ids (np.array): Gaia DR3 source ids
        verbose (bool): Print the executed query

    Returns:
        pd.DataFrame: Gaia DR3 results with best neighbours from panstarrs1_best_neighbour
    """
    
    return __gaia_dr3_cross_match(source_ids, 'skymapperdr2', 1., verbose)


def urat1_cross_match(source_ids: np.array, verbose: bool = False) -> pd.DataFrame:
    """Return cross-matched sources for DR3 sources

    Args:
        source_ids (np.array): Gaia DR3 source ids
        verbose (bool): Print the executed query

    Returns:
        pd.DataFrame: Gaia DR3 results with best neighbours from panstarrs1_best_neighbour
    """
    
    return __gaia_dr3_cross_match(source_ids, 'urat1', 1., verbose)


def cluster_sources_with_cross_match(ra: float,
                                     dec: float,
                                     parallax: float,
                                     pmra: float,
                                     pmdec: float,
                                     radvel: float,
                                     radius: float,
                                     min_parallax: float,
                                     max_parallax: float,
                                     max_neighbour_angular_separation: float = 0.01) -> pd.DataFrame:
    """Fetch sources by a cone search from Gaia DR3 and cross-match with panstarrs1, 2mass and wise

    Args:
        ra (float): right ascension of the cluster center [degrees]
        dec (float): declination of the cluster center [degrees]
        parallax (float): parallax of the cluster center [mas]
        pmra (float): proper motion of the cluster center [mas/yr]
        pmdec (float): proper motion of the cluster center [mas/yr]
        radvel (float): radial velocity of the cluster center [km/s]
        radius (float): radius to search around the cluster center [degrees]
        min_parallax (float): minimum parallax for sources [mas]
        max_parallax (float): maximum parallax for the sources [mas]
        max_neighbour_angular_separation (float): maximum angular separation for the neighbour to be a cross-match [arcseconds]

    Returns:
        pd.DataFrame: Gaia DR3 sources with IDs from panstarrs1, 2mass and wise
    """
    
    gaia_sources: pd.DataFrame = gaia_cone_search_5d(ra, dec, parallax, pmra, pmdec, radvel, radius, min_parallax, max_parallax)
    panstarrs1_sources: pd.DataFrame = panstarrs1_cone_search_best_neighbour(ra, dec, parallax, pmra, pmdec, radvel, radius, min_parallax, max_parallax)
    allwise_sources: pd.DataFrame = allwise_cone_search_best_neighbour(ra, dec, parallax, pmra, pmdec, radvel, radius, min_parallax, max_parallax)
    twomass_sources: pd.DataFrame = twomass_cone_search_best_neighbour(ra, dec, parallax, pmra, pmdec, radvel, radius, min_parallax, max_parallax)
    
    # Filter by the angular distance
    panstarrs1_sources = panstarrs1_sources[panstarrs1_sources['panstarrs1_angular_distance']<max_neighbour_angular_separation]
    allwise_sources = allwise_sources[allwise_sources['allwise_angular_distance']<max_neighbour_angular_separation]
    twomass_sources = twomass_sources[twomass_sources['twomass_angular_distance']<max_neighbour_angular_separation]
    
    # pd.merge(product,customer,on='Product_ID',how='left')
    gaia_sources: pd.DataFrame = pd.merge(gaia_sources, panstarrs1_sources, on='source_id', how='left')
    gaia_sources: pd.DataFrame = pd.merge(gaia_sources, allwise_sources, on='source_id', how='left')
    gaia_sources: pd.DataFrame = pd.merge(gaia_sources, twomass_sources, on='source_id', how='left')
    
    return gaia_sources
    
    
def gaia_get_dr2_in_dr3(dr2: int) -> pd.DataFrame:
    
    query: str = f'''
    SELECT dr2_source_id, dr3_source_id, angular_distance, magnitude_difference, proper_motion_propagation
    FROM gaiadr3.dr2_neighbourhood
    WHERE dr2_source_id = {dr2}
    '''
        
    print('Executing query:')
    print(query)
    
    job = Gaia.launch_job(query, output_format='csv')
    
    return job.get_results().to_pandas()


def get_photogeometric_distances(ra: float,
                                 dec: float,
                                 parallax: float,
                                 pmra: float,
                                 pmdec: float,
                                 radvel: float,
                                 radius: float,
                                 min_parallax: float,
                                 max_parallax: float) -> pd.DataFrame:
    
    query: str = f'''
      SELECT 
      source_id, ra, dec,
      r_med_geo, r_lo_geo, r_hi_geo,
      r_med_photogeo, r_lo_photogeo, r_hi_photogeo,
      phot_bp_mean_mag-phot_rp_mean_mag AS bp_rp,
      phot_g_mean_mag - 5 * LOG10(r_med_geo) + 5 AS qg_geo,
      phot_g_mean_mag - 5 * LOG10(r_med_photogeo) + 5 AS gq_photogeo
      FROM (
        SELECT * FROM gaiaedr3.gaia_source
            WHERE 1 = CONTAINS( 
            POINT('ICRS', ra, dec), 
            CIRCLE('ICRS',
                COORD1(EPOCH_PROP_POS({ra}, {dec}, {parallax}, {pmra}, {pmdec}, {radvel}, 2000, 2016.0)),
                COORD2(EPOCH_PROP_POS({ra}, {dec}, {parallax}, {pmra}, {pmdec}, {radvel}, 2000, 2016.0)),
            {radius})) 
        AND parallax > {min_parallax} AND parallax < {max_parallax} 
      ) AS edr3
      JOIN external.gaiaedr3_distance using(source_id)
      WHERE ruwe<1.4 
    '''
        
    print('Executing query:')
    print(query)
    
    job = Gaia.launch_job_async(query, output_format='csv')
    
    return job.get_results().to_pandas() 


def download_dr3_lightcurve(source_ids: np.ndarray) -> List[pd.DataFrame]:
    retrieval_type = 'ALL'          # Options are: 'EPOCH_PHOTOMETRY', 'MCMC_GSPPHOT', 'MCMC_MSC', 'XP_SAMPLED', 'XP_CONTINUOUS', 'RVS', 'ALL'
    data_structure = 'INDIVIDUAL'   # Options are: 'INDIVIDUAL', 'COMBINED', 'RAW'
    data_release   = 'Gaia DR3'     # Options are: 'Gaia DR3' (default), 'Gaia DR2'

    
    lightcurves: List[pd.DataFrame] = []

    datalink = Gaia.load_data(ids=source_ids,
                              data_release = data_release,
                              retrieval_type= 'EPOCH_PHOTOMETRY',
                              data_structure = data_structure,
                              verbose = False, output_file = None)
    dl_keys  = [inp for inp in datalink.keys()]
    dl_keys.sort()

    print(f'len{dl_keys} lightcurves found.')
    for dl_key in dl_keys:
        print(f'\tDownloading {dl_key}')
        lightcurves.append(datalink[dl_key][0].to_table().to_pandas())
        
    return lightcurves


def vari_class(source_ids: np.ndarray) -> pd.DataFrame:
    query = f'''
        SELECT source_id, in_vari_rrlyrae, in_vari_cepheid, in_vari_planetary_transit,
        in_vari_short_timescale, in_vari_long_period_variable,
        in_vari_eclipsing_binary, in_vari_rotation_modulation,
        in_vari_ms_oscillator, in_vari_agn,
        in_vari_microlensing, in_vari_compact_companion 
        FROM gaiadr3.vari_summary 
        WHERE vari_summary.source_id IN ({', '.join([str(si) for si in source_ids])})
    '''
    
    job = Gaia.launch_job_async(query, output_format='csv')
    
    return job.get_results().to_pandas()


def vari_short_timescale(source_ids: np.ndarray) -> pd.DataFrame:
    query = f'''
        SELECT * 
        FROM gaiadr3.vari_short_timescale 
        WHERE vari_short_timescale.source_id IN ({', '.join([str(si) for si in source_ids])})
    '''
    
    job = Gaia.launch_job_async(query, output_format='csv')
    
    return job.get_results().to_pandas()