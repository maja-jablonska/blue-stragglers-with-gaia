from simbad_download import resolve_name
from gaia_download import cluster_sources_with_cross_match, get_photogeometric_distances
from simbad_download import fetch_object_children, fetch_catalog_id
from extinction import add_colors_and_abs_mag
import pandas as pd
import click
from typing import List, Optional
from astropy.coordinates import ICRS, SkyCoord
import astropy.units as u
from plot_utils import plot_on_aitoff
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from cluster_utils import is_in_cluster_function


def normalize(sources: pd.DataFrame,
              columns: List[str],
              with_errors: bool = False) -> pd.DataFrame:
    scaler = StandardScaler()
    
    s = scaler.fit_transform(sources[columns])
    
    if with_errors:
        err_columns = [f'{c}_error' for c in columns]
        rescaled_errors = sources[err_columns].values*scaler.scale_
        s = np.concatenate([s, rescaled_errors], axis=1)
    return s


@click.command()
@click.argument('cluster_name', type=str)
@click.option('--radius', default=2, help='Radius of cone search in degrees.', type=float)
@click.option('--filepath', default=None, help='Path to the csv file with fetched sources.')
def download_sources_for_cluster(cluster_name: str, radius: float, filepath: Optional[str]):

    # COLUMNS = ['ra', 'dec', 'parallax', 'pmra', 'pmdec']

    # click.secho(f'\nResolving {cluster_name} using Simbad...')
    # cluster_ra, cluster_dec, cluster_parallax, cluster_pmra, cluster_pmdec, cluster_radvel = resolve_name(cluster_name)

    # if not (cluster_ra and cluster_dec and cluster_parallax):
    #     click.secho(f'Couldn\'t resolve name {cluster_name}.', fg='red', bold=True)
    #     return
        
    # click.secho(f'''
    #     Resolved coordinates for {cluster_name}
    #     \t ra={cluster_ra} deg
    #     \t dec={cluster_dec} deg
    #     \t parallax={cluster_parallax} mas
    #     \t pmra={cluster_pmra} mas/yr
    #     \t pmdec={cluster_pmdec} mas/yr
    #     \t radvel={cluster_radvel} km/s
    # ''')
    # min_parallax: float = max(0., cluster_parallax-0.25)
    # max_parallax: float = cluster_parallax+0.25

    # click.secho(f'''
    # Querying {radius} degrees around cluster center, with parallax in range [{min_parallax}, {max_parallax}]
    # ''')

    # sources: pd.DataFrame = cluster_sources_with_cross_match(cluster_ra, cluster_dec, cluster_parallax,
    #                                                          cluster_pmra, cluster_pmdec, cluster_radvel,
    #                                                          radius=radius,
    #                                                          min_parallax=min_parallax,
    #                                                          max_parallax=max_parallax)

    # click.secho(f'Found {len(sources.index)} sources')
    # click.secho(f'{len(sources.index)} sources after filtering')

    # sources = add_colors_and_abs_mag(sources)
    
    # sky_coords = SkyCoord(ra=sources['ra'].values,
    #                       dec=sources['dec'].values,
    #                       unit=(u.deg, u.deg),
    #                       frame=ICRS)
    
    # sources.ra = sky_coords.ra.wrap_at(180 * u.deg).value

    # filepath = filepath if filepath else f'{cluster_name}.csv'
    # sources.to_csv(filepath, index=None)
    # click.secho(f'Saved sources to {filepath}!', fg='green', bold=True)

    # normalized_sources = normalize(sources, COLUMNS, with_errors=True)
    # np.savetxt(filepath.replace('.csv', '_normalized.dat'), normalized_sources)
    # click.secho(f'Saved sources to {filepath.replace(".csv", "_normalized.dat")}!', fg='green', bold=True)

    click.secho(f'Downloading sources from Simbad...')
    literature_sources = fetch_object_children(cluster_name)
    click.secho(f'Found {len(literature_sources.index)}.')
    
    literature_sources['EDR3 id'] = np.vectorize(fetch_catalog_id)(literature_sources.ids, 'EDR3')
    literature_sources['DR2 id'] = np.vectorize(fetch_catalog_id)(literature_sources.ids, 'DR2')
    literature_sources['TIC'] = np.vectorize(fetch_catalog_id)(literature_sources.ids, 'TIC')
    print(literature_sources.columns)
    
    # Drop sources that aren't in EDR3
    literature_sources = literature_sources.dropna(subset=['EDR3 id'])
    print(len(literature_sources.ids))
    
    # is_in_edr3 = is_in_cluster_function(sources)
    
    # # Match the simbad sources with Gaia cone search result
    # from_lit_edr3 = literature_sources[np.vectorize(is_in_edr3)(literature_sources['EDR3 id'].values)]
    
    # click.secho(f'{len(from_lit_edr3.index)} sources both in Simbad and Gaia DR3 cone search result')

    # from_lit_edr3.to_csv(filepath.replace('.csv', '_literature.csv'), index=None)

    # click.secho(f'Saved sources from Simbad to {filepath.replace(".csv", "_literature.csv")}!', fg='green', bold=True)

    # plot_on_aitoff(sources, cluster_name, radius)
    # plt.show()

if __name__ == '__main__':
    download_sources_for_cluster()
