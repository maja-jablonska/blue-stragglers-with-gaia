import pyvo as vo
import pandas as pd
import numpy as np
from astroquery.esasky import ESASky


PANSTARRS1 = vo.dal.TAPService("http://vao.stsci.edu/PS1DR2/tapservice.aspx")


def query_panstarrs1(obj_ids: np.array) -> pd.DataFrame:
    job = PANSTARRS1.run_async(f"""
        SELECT objID, RAMean, DecMean, nDetections,
        gMeanPSFMag, gMeanPSFMagErr,
        rMeanPSFMag, rMeanPSFMagErr,
        iMeanPSFMag, iMeanPSFMagErr,
        zMeanPSFMag, zMeanPSFMagErr,
        yMeanPSFMag, yMeanPSFMagErr 
        FROM dbo.MeanObjectView
        WHERE objID IN ({', '.join([str(o) for o in obj_ids])})
        """)
    TAP_results = job.to_table()
    return TAP_results.to_pandas()


def query_twomass(obj_ids: np.array) -> pd.DataFrame:
    # dec	err_ang	err_maj	err_min	ext_key	h_m	h_msigcom	j_date	j_m	j_msigcom	ks_m	ks_msigcom	name	ra
    COLUMNS = ['name', 'h_m', 'h_msigcom', 'j_m', 'j_msigcom', 'ks_m', 'ks_msigcom']
    try:
        return ESASky.query_ids_catalogs(source_ids=obj_ids, catalogs=["TwoMASS"])[0].to_pandas()[COLUMNS]
    except:
        return pd.DataFrame(columns=COLUMNS, data=[])


def query_allwise(obj_ids: np.array) -> pd.DataFrame:
    # w1mpro	w1mpro_error	
    COLUMNS = ['name', 'w1mpro', 'w1mpro_error', 'w2mpro', 'w2mpro_error',
               'w3mpro', 'w3mpro_error', 'w4mpro', 'w4mpro_error']
    try:
        return ESASky.query_ids_catalogs(source_ids=obj_ids, catalogs=["ALLWise"])[0].to_pandas()[COLUMNS]
    except:
        return pd.DataFrame(columns=COLUMNS, data=[])
