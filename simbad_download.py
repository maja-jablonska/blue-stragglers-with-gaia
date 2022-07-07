import pyvo as vo
import numpy as np
import pandas as pd

def clean_str(obj_id: str) -> str:
    return ' '.join(obj_id.split())


def fetch_object_children(obj_identifier: str) -> pd.DataFrame:

    service = vo.dal.TAPService("http://simbad.u-strasbg.fr/simbad/sim-tap")
    resultset = service.search(f'''
        SELECT main_id as child, link_bibcode, membership, ra, dec
        from h_link JOIN ident as p on p.oidref=parent JOIN basic on oid=child
        WHERE p.id = '{obj_identifier}' and (membership >=95 or membership is null);''')
    obj_ids = resultset['child'].data
    bibcodes = resultset['link_bibcode'].data
    ras = resultset['ra'].data
    decs = resultset['dec'].data

    data = np.array([
        np.array(list(map(clean_str, obj_ids))),
        bibcodes,
        ras.astype(float),
        decs.astype(float)
    ])

    cluster_children: pd.DataFrame = pd.DataFrame(columns = ['obj_id', 'link_bibcode', 'ra', 'dec'], data=data.T)
    cluster_children['ra'] = cluster_children['ra'].astype(float)
    cluster_children['dec'] = cluster_children['dec'].astype(float)
    cluster_children = cluster_children.dropna()
    return cluster_children

