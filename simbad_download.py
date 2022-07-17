from unittest import result
import pyvo as vo
import numpy as np
import pandas as pd
import re
from typing import Optional, Tuple


def simbad_tap():
    return vo.dal.TAPService("http://simbad.u-strasbg.fr/simbad/sim-tap")


def clean_str(obj_id: str) -> str:
    return ' '.join(obj_id.split())


def fetch_catalog_id(ids: str, catalog_identifier: str, verbose: bool = False):
    try:
        return re.findall(f'(?<={catalog_identifier} )\d+', ids)[0]
    except IndexError:
        if verbose:
            print(f'No {catalog_identifier} id for ids={ids}...')
        return np.nan


def resolve_name(obj_identifier: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    service = simbad_tap()
    try:
        resultset = service.search(f'''select ra, dec, plx_value 
            from basic where main_id='{obj_identifier}'
        ''').to_table().to_pandas().values

        if len(resultset) == 1:
            return resultset[0, 0], resultset[0, 1], resultset[0, 2]
        else:
            return None, None, None
    except Exception as e:
        print(f'Exception while querying: {e}')
        return None, None, None



def fetch_object_children(obj_identifier: str) -> pd.DataFrame:
    service = simbad_tap()
    resultset = service.search(f'''
        SELECT main_id as child, oid, link_bibcode, membership,
        ra, dec, coo_bibcode,
        plx_value, plx_err, plx_bibcode,
        pmra, pmdec, pm_err_maj_prec, pm_bibcode,
        rvz_radvel, rvz_err, rvz_bibcode, ids.ids
        from h_link JOIN ident as p on p.oidref=parent JOIN basic on oid=child JOIN ids on ids.oidref=child
        WHERE p.id = '{obj_identifier}' and (membership >=95 or membership is null);''')
    
    obj_ids = resultset['child'].data
    oids = resultset['oid'].data
    bibcodes = resultset['link_bibcode'].data
    ras = resultset['ra'].data
    decs = resultset['dec'].data
    coo_bibcodes = resultset['coo_bibcode'].data
    plx_values = resultset['plx_value'].data
    plx_errs = resultset['plx_err'].data
    plx_bibcodes = resultset['plx_bibcode'].data
    pmras = resultset['pmra'].data
    pmdecs = resultset['pmdec'].data
    pm_errs = resultset['pm_err_maj_prec'].data
    pm_bibcodes = resultset['pm_bibcode'].data
    radvels = resultset['rvz_radvel'].data
    rvz_errs = resultset['rvz_err'].data
    rvz_bibcodes = resultset['rvz_bibcode'].data
    ids = resultset['ids'].data

    data = np.array([
        np.array(list(map(clean_str, obj_ids))),
        oids.astype(int),
        bibcodes,
        ras.astype(float),
        decs.astype(float),
        coo_bibcodes,
        plx_values.astype(float),
        plx_errs.astype(float),
        plx_bibcodes,
        pmras.astype(float),
        pmdecs.astype(float),
        pm_errs.astype(float),
        pm_bibcodes,
        radvels.astype(float),
        rvz_errs.astype(float),
        rvz_bibcodes,
        ids
    ])

    cluster_children: pd.DataFrame = pd.DataFrame(
        columns=['obj_id', 'oid', 'link_bibcode', 'ra', 'dec', 'coo_bibcode',
                 'parallax', 'parallax_err', 'parallax_bibcode',
                 'pmra', 'pmdec', 'pm_err', 'pm_bibcode',
                 'radvel', 'radvel_err', 'rvz_bibcode', 'ids'],
        data=data.T)
    cluster_children = cluster_children.dropna(subset=['ra', 'dec', 'link_bibcode'])
    return cluster_children

