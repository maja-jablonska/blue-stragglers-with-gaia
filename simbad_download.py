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
        resultset = service.search(f'''select ra, dec, plx_value, pmra, pmdec, rvz_radvel 
            from basic where main_id='{obj_identifier}'
        ''').to_table().to_pandas().values

        if len(resultset) == 1:
            return tuple(resultset[0, :])
        else:
            return None, None, None, None, None, None
    except Exception as e:
        print(f'Exception while querying: {e}')
        return None, None, None, None, None, None


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
    
    cluster_children['EDR3 id'] = np.vectorize(fetch_catalog_id)(cluster_children.ids, 'EDR3')
    cluster_children['DR2 id'] = np.vectorize(fetch_catalog_id)(cluster_children.ids, 'DR2')
    cluster_children['TIC'] = np.vectorize(fetch_catalog_id)(cluster_children.ids, 'TIC')
    
    cluster_children['EDR3 id'] = pd.to_numeric(cluster_children['EDR3 id'], errors='coerce')
    cluster_children['DR2 id'] = pd.to_numeric(cluster_children['DR2 id'], errors='coerce')
    cluster_children['TIC'] = pd.to_numeric(cluster_children['TIC'], errors='coerce')
    
    cluster_children = cluster_children.dropna(subset=['EDR3 id'])
    
    edr_unique = np.unique(cluster_children['EDR3 id'].values)
    reported_counts = {x: len(np.nonzero(cluster_children['EDR3 id'].values==x)[0]) for x in edr_unique}
    cluster_children['reported'] = cluster_children['EDR3 id'].apply(lambda x: reported_counts[x])
    cluster_children['parallax_year'] = cluster_children['parallax_bibcode'].apply(lambda x: x[:4])
    cluster_children['pm_year'] = cluster_children['pm_bibcode'].apply(lambda x: x[:4])
    cluster_children['rvz_year'] = cluster_children['rvz_bibcode'].apply(lambda x: x[:4])
    cluster_children = cluster_children.sort_values(by=['EDR3 id', 'parallax_year', 'pm_year', 'rvz_year'])
    cluster_children = cluster_children.drop_duplicates(subset=['EDR3 id'])
    
    return cluster_children




def title_and_authors(bibcode: str) -> str:
    URL = f'https://ui.adsabs.harvard.edu/abs/{bibcode}/abstract'
    website = requests.get(URL)
    results = BeautifulSoup(website.content, 'html.parser')
    title = ' '.join(results.find('h2', class_='s-abstract-title').text.split())
    authors = [author.text.strip() for author in results.find_all('li', class_='author')]
    return f'{",".join(authors)}:\n {title}'


def count_reportings(children, edr3_id):
    return len(children[children['EDR3 id'].astype(int)==edr3_id])

