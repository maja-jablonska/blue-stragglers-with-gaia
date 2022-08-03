from simbad_download import resolve_name
from gaia_download import gaia_cone_search_5d
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
import os.path
import os
from tqdm import tqdm


def normalize(cluster_values: np.array) -> pd.DataFrame:
    scaler = StandardScaler()
    return scaler.fit_transform(cluster_values)


def cp_proper_motions(ra: np.float32, 
                      dec: np.float32,
                      par: np.float32,
                      pmra: np.float32,
                      pmdec: np.float32,
                      cp_pmra: np.float32,
                      cp_pmdec: np.float32,
                      cp_par: np.float32) -> np.matrix:
    
    tan_theta = cp_pmra/cp_pmdec
    
    sin_theta = tan_theta/np.sqrt(1+np.power(tan_theta, 2))
    cos_theta = 1/np.sqrt(1+np.power(tan_theta, 2))
    mu_values = np.squeeze(np.asarray(np.matmul(
                np.matrix([[sin_theta, cos_theta],
                           [-cos_theta, sin_theta]]),
                np.matrix([pmra, pmdec]).T)))
    return mu_values/par-np.array([cp_pmra/cp_par, 0.])


matrices = np.vectorize(cp_proper_motions, excluded=['cp_pmra', 'cp_pmdec', 'cp_par'], otypes=[np.ndarray])


@click.command()
@click.argument('cluster_name', type=str)
@click.option('-r', '--radius', default=2, help='Radius of cone search in degrees.', type=float)
@click.option('-f', '--filepath', default=None, help='Path to the csv file with fetched sources.')
@click.option('-oa', '--overwrite-all', is_flag=True, default=False, help='Overwrite all the data if files exist.')
@click.option('-os', '--overwrite-sources', is_flag=True, default=False, help='Overwrite Gaia data files exist.')
@click.option('-ol', '--overwrite-literature', is_flag=True, default=False, help='Overwrite Simbad data if files exist.')
def download_sources_for_cluster(cluster_name: str, radius: float, filepath: Optional[str],
                                 overwrite_all: bool, overwrite_sources: bool, overwrite_literature: bool):
    
    if filepath and not filepath.endswith('.csv'):
        click.secho(f'Please provide a .csv file!', fg='red', bold=True)
        return
    
    PATH_ROOT: str = f'data/{cluster_name}'
    if not filepath and not os.path.exists(PATH_ROOT):
        os.makedirs(PATH_ROOT)
        
    SOURCES_FILEPATH: str = filepath if filepath else f'{PATH_ROOT}/{cluster_name}.csv'
    NORMALIZED_FILEPATH: str = SOURCES_FILEPATH.replace('.csv', '_normalized.dat')
    LITERATURE_FILEPATH: str = SOURCES_FILEPATH.replace('.csv', '_literature.csv')
        
    COLUMNS = ['ra', 'dec', 'parallax', 'pmra', 'pmdec']
    
    if (not os.path.isfile(SOURCES_FILEPATH)) or overwrite_all or overwrite_sources:

        click.secho(f'\nResolving {cluster_name} using Simbad...')
        cluster_ra, cluster_dec, cluster_parallax, cluster_pmra, cluster_pmdec, cluster_radvel = resolve_name(cluster_name)

        if not (cluster_ra and cluster_dec and cluster_parallax):
            click.secho(f'Couldn\'t resolve name {cluster_name}.', fg='red', bold=True)
            return

        click.secho(f'''
            Resolved coordinates for {cluster_name}
            \t ra={cluster_ra} deg
            \t dec={cluster_dec} deg
            \t parallax={cluster_parallax} mas
            \t pmra={cluster_pmra} mas/yr
            \t pmdec={cluster_pmdec} mas/yr
            \t radvel={cluster_radvel} km/s
        ''')
        min_parallax: float = max(0., cluster_parallax-0.25)
        max_parallax: float = cluster_parallax+0.25

        click.secho(f'''
        Querying {radius} degrees around cluster center, with parallax in range [{min_parallax}, {max_parallax}]
        ''')

        sources: pd.DataFrame = gaia_cone_search_5d(cluster_ra, cluster_dec, cluster_parallax,
                                                    cluster_pmra, cluster_pmdec, cluster_radvel,
                                                    radius=radius,
                                                    min_parallax=min_parallax,
                                                    max_parallax=max_parallax)

        click.secho(f'Found {len(sources.index)} sources')
        click.secho(f'{len(sources.index)} sources after filtering')

        sources = add_colors_and_abs_mag(sources)

        sky_coords = SkyCoord(ra=sources['ra'].values,
                              dec=sources['dec'].values,
                              unit=(u.deg, u.deg),
                              frame=ICRS)

        sources.ra = sky_coords.ra.wrap_at(180 * u.deg).value

        sources.to_csv(SOURCES_FILEPATH, index=None)
        click.secho(f'Saved sources to {SOURCES_FILEPATH}!', fg='green', bold=True)
        
    else:
        click.secho(f'{SOURCES_FILEPATH} exists and --overwrite=False. Reading the file.', fg='yellow', bold=True)
        
        try:
            sources = pd.read_csv(SOURCES_FILEPATH)
        except Exception as e:
            click.secho(f'Exception when reading {SOURCES_FILEPATH}: {e}', fg='red', bold=True)
            return
    
    if (not os.path.isfile(NORMALIZED_FILEPATH)) or overwrite_all or overwrite_sources:
        
        galactic_coordinates = SkyCoord(ra=sources.ra.values*u.deg, dec=sources.dec.values*u.deg, 
                                        distance=(1/sources.parallax.values)*u.kpc,
                                        radial_velocity=cluster_radvel*u.km/u.s,
                                        pm_ra_cosdec=sources.pmra.values*u.mas/u.yr,
                                        pm_dec=sources.pmdec.values*u.mas/u.yr,
                                        frame=ICRS)
        
        galactic_cartesian = galactic_coordinates.galactic.cartesian
        
        proper_motions = np.stack(
            matrices(
                sources.ra.values,
                sources.dec.values,
                sources.parallax.values,
                sources.pmra.values,
                sources.pmdec.values,
                cluster_pmra,
                cluster_pmdec,
                cluster_parallax
            )
        )
        
        cluster_values = np.concatenate([
            galactic_cartesian.x.value.reshape((-1, 1)),
            galactic_cartesian.y.value.reshape((-1, 1)),
            galactic_cartesian.z.value.reshape((-1, 1)),
            proper_motions[:, 0].reshape((-1, 1)),
            proper_motions[:, 1].reshape((-1, 1))
        ], axis=1)
    
        normalized_sources = normalize(cluster_values)
        np.savetxt(NORMALIZED_FILEPATH, normalized_sources)
        
        click.secho(f'Saved sources to {NORMALIZED_FILEPATH}!', fg='green', bold=True)
    
    else:
        click.secho(f'{NORMALIZED_FILEPATH} exists and --overwrite=False.', fg='yellow', bold=True)
        
    if (not os.path.isfile(LITERATURE_FILEPATH)) or overwrite_all or overwrite_literature:

        click.secho(f'Downloading sources from Simbad...')
        literature_sources = fetch_object_children(cluster_name)
        click.secho(f'Found {len(literature_sources.index)}.')
    
        literature_sources['EDR3 id'] = np.vectorize(fetch_catalog_id)(literature_sources.ids, 'EDR3')
        literature_sources['DR2 id'] = np.vectorize(fetch_catalog_id)(literature_sources.ids, 'DR2')
        literature_sources['TIC'] = np.vectorize(fetch_catalog_id)(literature_sources.ids, 'TIC')
    
        # Drop sources that aren't in EDR3
        literature_sources = literature_sources.dropna(subset=['EDR3 id'])

        click.secho(f'{len(literature_sources.index)} sources both in Simbad and Gaia DR3')

        literature_sources.to_csv(LITERATURE_FILEPATH, index=None)

        click.secho(f'Saved sources from Simbad to {LITERATURE_FILEPATH}!', fg='green', bold=True)
        
    else:
        click.secho(f'{LITERATURE_FILEPATH} exists and --overwrite=False.', fg='yellow', bold=True)


    plot_on_aitoff(sources, cluster_name, radius)
    plt.show()

if __name__ == '__main__':
    download_sources_for_cluster()
