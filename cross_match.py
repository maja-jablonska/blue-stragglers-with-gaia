import pyvo as vo
import pandas as pd
import numpy as np
from astroquery.esasky import ESASky


PANSTARRS1 = vo.dal.TAPService("http://vao.stsci.edu/PS1DR2/tapservice.aspx")
IRSA = vo.dal.TAPService('https://irsa.ipac.caltech.edu/TAP')


# Refer to https://irsa.ipac.caltech.edu/data/WISE/CatWISE/gator_docs/catwise_colDescriptions.html#apflag

CATWISE_FLAGS = {
    0: 'no contamination',
    1: 'source confusion',
    2: 'bad or fatal pixels',
    4: 'non-zero bit flag tripped',
    8: 'corruption',
    16: 'saturation',
    32: 'upper limit'
}


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
    job = IRSA.run_async(f"""
           SELECT source_name, w1mag, w1sigm, w1flg, w2mag, w2sigm, w2flg 
           FROM catwise_2020
           WHERE source_name IN ({', '.join([f"'{str(o)}'" for o in obj_ids])})
    """)
    TAP_results = job.to_table()
    return TAP_results.to_pandas()
