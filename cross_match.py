import pyvo as vo
import pandas as pd
import numpy as np
from astroquery.esasky import ESASky
from gaia_download import panstarrs1_cross_match, allwise_cross_match, twomass_cross_match


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


def query_panstarrs1(obj_ids: np.array = []) -> pd.DataFrame:
    
    if len(obj_ids) == 0:
        return pd.DataFrame(data=[],
                            columns=['objID',
                                     'g', 'g_error',
                                     'r', 'r_error',
                                     'i', 'i_error',
                                     'z', 'z_error',
                                     'y', 'y_error'])
    
    job = PANSTARRS1.run_async(f"""
        SELECT objID,
        gMeanPSFMag, gMeanPSFMagErr,
        rMeanPSFMag, rMeanPSFMagErr,
        iMeanPSFMag, iMeanPSFMagErr,
        zMeanPSFMag, zMeanPSFMagErr,
        yMeanPSFMag, yMeanPSFMagErr 
        FROM dbo.MeanObjectView
        WHERE objID IN ({', '.join([str(o) for o in obj_ids])})
        """)
    TAP_results = job.to_table()
    return TAP_results.to_pandas().rename(columns={
        'gMeanPSFMag': 'g',
        'gMeanPSFMagErr': 'g_error',
        'rMeanPSFMag': 'r',
        'rMeanPSFMagErr': 'r_error',
        'iMeanPSFMag': 'i',
        'iMeanPSFMagErr': 'i_error',
        'zMeanPSFMag': 'z',
        'zMeanPSFMagErr': 'z_error'
    })


def add_panstarrs1(sources: pd.DataFrame) -> pd.DataFrame:
    source_ids = sources.source_id.values
    panstarrs1_ids = panstarrs1_cross_match(source_ids)
    panstarrs1_photo = query_panstarrs1(panstarrs1_ids.panstarrs1_id.values)
    panstarrs_data = pd.merge(right=panstarrs1_ids,
                              left=panstarrs1_photo,
                              right_on='panstarrs1_id',
                              left_on='objID').drop(columns=['objID'])
    return pd.merge(right=sources, left=panstarrs_data, how='right', on='source_id')
    

def query_twomass(obj_ids: np.array = []) -> pd.DataFrame:
    COLUMNS = ['name', 'h_m', 'h_msigcom', 'j_m', 'j_msigcom', 'ks_m', 'ks_msigcom']
    RENAMED_COLUMNS = ['name', 'H', 'H_error', 'J', 'J_error', 'K', 'K_error']
    
    if len(obj_ids) == 0:
        return pd.DataFrame(data=[],
                            columns=RENAMED_COLUMNS)
    
    try:
        return ESASky.query_ids_catalogs(
            source_ids=[str(oi) for oi in obj_ids],
            catalogs=["TwoMASS"])[0].to_pandas()[COLUMNS].rename(columns={
                'h_m': 'H',
                'h_msigcom': 'H_error',
                'j_m': 'J',
                'j_msigcom': 'J_error',
                'ks_m': 'K',
                'ks_msigcom': 'K_error'
            })
    except:
        return pd.DataFrame(columns=RENAMED_COLUMNS, data=[])
    
    
def add_twomass(sources: pd.DataFrame) -> pd.DataFrame:
    source_ids = sources.source_id.values
    twomass_ids = twomass_cross_match(source_ids)
    twomass_photo = query_twomass(twomass_ids.twomass_id.values)
    twomass_data = pd.merge(right=twomass_ids,
                            left=twomass_photo,
                            right_on='twomass_id',
                            left_on='name').drop(columns=['name'])
    return pd.merge(right=sources, left=twomass_data, how='right', on='source_id')


def query_allwise(obj_ids: np.array = []) -> pd.DataFrame:
    
    # TODO: add photometry flags
    if len(obj_ids) == 0:
        return pd.DataFrame(data=[],
                            columns=['source_name', 'w1', 
                                     'w1_error', 'w1flg',
                                     'w2', 'w2_error', 'w2flg'])
    
    job = IRSA.run_async(f"""
           SELECT source_name, w1mag, w1sigm, w1flg, w2mag, w2sigm, w2flg 
           FROM catwise_2020
           WHERE source_name IN ({', '.join([f"'{str(o)}'" for o in obj_ids])})
    """)
    TAP_results = job.to_table()
    return TAP_results.to_pandas().rename(columns={
        'w1mag': 'w1',
        'w1sigm': 'w1_error',
        'w2mag': 'w2',
        'w2sigm': 'w2_error'
    })
