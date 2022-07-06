from tracemalloc import start
from astroquery.gaia import Gaia
import pandas as pd

def get_sources_by_random_index(rows: int = 1000, start_index: int = 0) -> pd.DataFrame:
    job = Gaia.launch_job(f'''
        SELECT source_id, ra, dec, parallax, parallax_error, 
        pmra, pmra_error, pmdec, pmdec_error 
        FROM gaiadr3.gaia_source 
        WHERE random_index > {start_index} AND random_index < {start_index+rows}
    ''', output_format='csv')

    return job.get_results().to_pandas()
