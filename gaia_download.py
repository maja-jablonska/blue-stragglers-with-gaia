from numpy import column_stack
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


def panstarrs1_cone_search_best_neighbour(ra: float,
                                          dec: float,
                                          parallax: float,
                                          pmra: float,
                                          pmdec: float,
                                          radvel: float,
                                          radius: float,
                                          min_parallax: float,
                                          max_parallax: float) -> pd.DataFrame:
    """Perform a cone search around cluster center with proper motion propagation.

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

    Returns:
        pd.DataFrame: Gaia DR3 cone search results with best neighbours from panstarrs1_best_neighbour
    """
    
    query: str = f'''
      SELECT
      panstarrs1_best_neighbour.source_id,
      panstarrs1_best_neighbour.original_ext_source_id,
      panstarrs1_best_neighbour.angular_distance
      FROM gaiadr3.gaia_source JOIN gaiadr3.panstarrs1_best_neighbour ON gaia_source.source_id=panstarrs1_best_neighbour.source_id
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
    
    return job.get_results().to_pandas().rename(columns={
        'original_ext_source_id': 'panstarrs1_id',
        'angular_distance': 'panstarrs1_angular_distance'
    })


def allwise_cone_search_best_neighbour(ra: float,
                                       dec: float,
                                       parallax: float,
                                       pmra: float,
                                       pmdec: float,
                                       radvel: float,
                                       radius: float,
                                       min_parallax: float,
                                       max_parallax: float) -> pd.DataFrame:
    """Perform a cone search around cluster center with proper motion propagation.

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

    Returns:
        pd.DataFrame: Gaia DR3 cone search results with best neighbours from allwise_best_neighbour
    """
    
    query: str = f'''
      SELECT 
      allwise_best_neighbour.source_id, 
      allwise_best_neighbour.original_ext_source_id, 
      allwise_best_neighbour.angular_distance 
      FROM gaiadr3.gaia_source JOIN gaiadr3.allwise_best_neighbour ON gaia_source.source_id=allwise_best_neighbour.source_id
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
    
    return job.get_results().to_pandas().rename(columns={
        'original_ext_source_id': 'allwise_id',
        'angular_distance': 'allwise_angular_distance'
    })


def twomass_cone_search_best_neighbour(ra: float,
                                       dec: float,
                                       parallax: float,
                                       pmra: float,
                                       pmdec: float,
                                       radvel: float,
                                       radius: float,
                                       min_parallax: float,
                                       max_parallax: float) -> pd.DataFrame:
    """Perform a cone search around cluster center with proper motion propagation.

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

    Returns:
        pd.DataFrame: Gaia DR3 cone search results with best neighbours from tmass_psc_xsc_best_neighbour
    """
    
    query: str = f'''
      SELECT 
      tmass_psc_xsc_best_neighbour.source_id, 
      tmass_psc_xsc_best_neighbour.original_ext_source_id, 
      tmass_psc_xsc_best_neighbour.angular_distance 
      FROM gaiadr3.gaia_source JOIN gaiadr3.tmass_psc_xsc_best_neighbour ON gaia_source.source_id=tmass_psc_xsc_best_neighbour.source_id
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
    
    return job.get_results().to_pandas().rename(columns={
        'original_ext_source_id': 'twomass_id',
        'angular_distance': 'twomass_angular_distance'
    })


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