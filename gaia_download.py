from tracemalloc import start
from astroquery.gaia import Gaia
import pandas as pd

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
                        radius: float,
                        min_parallax: float,
                        max_parallax: float) -> pd.DataFrame:
    """
        :param ra: right ascension of the cluster center [degree]
        :param dec: declination of the cluster center [degree]
        :param radius: radius of the area we wish to search in [degrees]
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
            CIRCLE('ICRS', {ra_str}, {dec_str}, {radius})) 
        AND parallax > {min_parallax} AND parallax < {max_parallax} 
    '''
        
    print('Executing query:')
    print(query)
    
    job = Gaia.launch_job_async(query, output_format='csv')

    print(f"Query finished!")
    
    return job.get_results().to_pandas()
    
    
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
            CIRCLE('ICRS', {ra}, {dec}, {radius})) 
        AND parallax > {min_parallax} AND parallax < {max_parallax} 
      ) AS edr3
      JOIN external.gaiaedr3_distance using(source_id)
      WHERE ruwe<1.4 
    '''
        
    print('Executing query:')
    print(query)
    
    job = Gaia.launch_job_async(query, output_format='csv')
    
    return job.get_results().to_pandas() 