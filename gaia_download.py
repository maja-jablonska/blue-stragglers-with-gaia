from tracemalloc import start
from astroquery.gaia import Gaia
import pandas as pd

from astropy.coordinates import ICRS, SkyCoord
import astropy.units as u

Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
Gaia.ROW_LIMIT = -1 # Fetch all results

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
#     results = Gaia.cone_search_async(SkyCoord(coords, frame=ICRS, unit=u.deg),
#                                      u.Quantity(radius, u.deg))
    job = Gaia.launch_job_async(f'''
        SELECT source_id, ra, dec, parallax, parallax_error, 
        pmra, pmra_error, pmdec, pmdec_error 
        FROM gaiadr3.gaia_source 
        WHERE 1 = CONTAINS( 
            POINT({ra}, {dec}), 
            CIRCLE(ra, dec, {radius})) 
        AND parallax > {min_parallax} AND parallax < {max_parallax} 
    ''', output_format='csv')

    print(f"Query finished!")
    
    return job.get_results().to_pandas()
    