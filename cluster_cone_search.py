from simbad_download import resolve_name
from gaia_download import gaia_cone_search_5d
from simbad_download import fetch_object_children, fetch_catalog_id
from extinction import add_colors_and_abs_mag
import pandas as pd
import click
from typing import List, Optional
from astropy.coordinates import ICRS, SkyCoord, Distance
import astropy.units as u
from plot_utils import plot_on_aitoff
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import os.path
import os
from tqdm import tqdm
from uncertainties import unumpy


def normalize(cluster_values) -> pd.DataFrame:
    means: np.ndarray = np.mean(unumpy.nominal_values(cluster_values), axis=0).reshape(1, -1)
    stds: np.ndarray = np.std(unumpy.nominal_values(cluster_values), axis=0).reshape(1, -1)
    return (cluster_values-means)/stds


def galactic_coords_with_uncert(ra, ra_err,
                                dec, dec_err,
                                par, par_err):
    ras = np.clip(np.random.normal(scale=ra_err, size=(100,))+ra, a_min=0, a_max=360)
    decs = np.clip(np.random.normal(scale=dec_err, size=(100,))+dec, a_min=-90, a_max=90)
    pars = np.clip(np.random.normal(scale=par_err, size=(100,))+par, a_min=1e-5, a_max=None)
    coords = SkyCoord(ra=ras*u.deg, dec=decs*u.deg, distance=1/pars*u.kpc, frame=ICRS).galactic.cartesian
    xs = coords.x.value
    ys = coords.y.value
    zs = coords.z.value
    return np.array([np.mean(xs), np.mean(ys), np.mean(zs), np.std(xs), np.std(ys), np.std(zs)])


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
@click.option('-pmin', type=float, default=None)
@click.option('-pmax', type=float, default=None)
def download_sources_for_cluster(cluster_name: str, radius: float, filepath: Optional[str],
                                 overwrite_all: bool, overwrite_sources: bool, overwrite_literature: bool,
                                 pmin: float, pmax: float):
    
    if filepath and not filepath.endswith('.csv'):
        click.secho(f'Please provide a .csv file!', fg='red', bold=True)
        return
    
    PATH_ROOT: str = f'data/{cluster_name}'
    if not filepath and not os.path.exists(PATH_ROOT):
        os.makedirs(PATH_ROOT)
        
    SOURCES_FILEPATH: str = filepath if filepath else f'{PATH_ROOT}/{cluster_name}.csv'
    NORMALIZED_FILEPATH: str = SOURCES_FILEPATH.replace('.csv', '_normalized.dat')
    NORMALIZED_UNCERT_FILEPATH: str = SOURCES_FILEPATH.replace('.csv', '_normalized_uncert.dat')
    NORMALIZED_CP_FILEPATH: str = SOURCES_FILEPATH.replace('.csv', '_normalized_cp.dat')
    NORMALIZED_CP_UNCERT_FILEPATH: str = SOURCES_FILEPATH.replace('.csv', '_normalized_cp_uncert.dat')
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
        min_parallax: float = pmin if pmin else max(0., cluster_parallax-0.25)
        max_parallax: float = pmax if pmax else cluster_parallax+0.25

        click.secho(f'''
        Querying {radius} degrees around cluster center, with parallax in range [{min_parallax}, {max_parallax}]
        ''')

        sources: pd.DataFrame = gaia_cone_search_5d(cluster_ra, cluster_dec, cluster_parallax,
                                                    cluster_pmra, cluster_pmdec, cluster_radvel,
                                                    radius=radius,
                                                    min_parallax=min_parallax,
                                                    max_parallax=max_parallax)
        sources = sources[sources.parallax>0]

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
        
        ras = unumpy.uarray(sources.ra.values, sources.ra_error.values)
        decs = unumpy.uarray(sources.dec.values, sources.dec_error.values)
        pmras = unumpy.uarray(sources.pmra.values, sources.pmra_error.values)
        pmdecs = unumpy.uarray(sources.pmdec.values, sources.pmdec_error.values)
        parallaxes = unumpy.uarray(sources.parallax.values, sources.parallax_error.values)
        
        proper_motions = np.stack(
            matrices(
                ras,
                decs,
                parallaxes,
                pmras,
                pmdecs,
                cluster_pmra,
                cluster_pmdec,
                cluster_parallax
            )
        )
        
        cluster_values = sources[['ra', 'dec', 'parallax', 'pmra', 'pmdec']].values
        uncertainties = sources[['ra_error', 'dec_error', 'parallax_error', 'pmra_error', 'pmdec_error']].values
        
        cluster_values_with_uncert = unumpy.uarray(cluster_values.astype(np.float64),
                                                   uncertainties.astype(np.float64))
        
        normalized_sources = normalize(cluster_values_with_uncert)
        
        np.savetxt(NORMALIZED_FILEPATH, unumpy.nominal_values(normalized_sources))
        np.savetxt(NORMALIZED_UNCERT_FILEPATH, unumpy.std_devs(normalized_sources))
        
        galactic_cartesian = np.array([galactic_coords_with_uncert(*x) for x in zip(
                                                         sources.ra.values.flatten(),
                                                         sources.ra_error.values.flatten(),
                                                         sources.dec.values.flatten(),
                                                         sources.dec_error.values.flatten(),
                                                         sources.parallax.values.flatten(),
                                                         sources.parallax_error.values.flatten())])
        
        cluster_values_cp = np.concatenate([
            galactic_cartesian[:, 0].reshape((-1, 1)),
            galactic_cartesian[:, 1].reshape((-1, 1)),
            galactic_cartesian[:, 2].reshape((-1, 1)),
            unumpy.nominal_values(proper_motions)[:, 0].reshape((-1, 1)),
            unumpy.nominal_values(proper_motions)[:, 1].reshape((-1, 1))
        ], axis=1)
        
        cp_uncertainties = np.concatenate([
            galactic_cartesian[:, 3].reshape((-1, 1)),
            galactic_cartesian[:, 4].reshape((-1, 1)),
            galactic_cartesian[:, 5].reshape((-1, 1)),
            unumpy.std_devs(proper_motions)[:, 0].reshape((-1, 1)),
            unumpy.std_devs(proper_motions)[:, 1].reshape((-1, 1))
        ], axis=1)
        
        cluster_values_cp_with_uncert = unumpy.uarray(cluster_values_cp, cp_uncertainties)
    
        normalized_sources_cp = normalize(cluster_values_cp)
        np.savetxt(NORMALIZED_CP_FILEPATH, unumpy.nominal_values(normalized_sources_cp))
        np.savetxt(NORMALIZED_CP_UNCERT_FILEPATH, unumpy.std_devs(normalized_sources_cp))
        
        click.secho(f'Saved sources to {NORMALIZED_FILEPATH} and {NORMALIZED_CP_FILEPATH}!', fg='green', bold=True)
    
    else:
        click.secho(f'{NORMALIZED_FILEPATH} exists and --overwrite=False.', fg='yellow', bold=True)
        
    if (not os.path.isfile(LITERATURE_FILEPATH)) or overwrite_all or overwrite_literature:

        click.secho(f'Downloading sources from Simbad...')
        literature_sources = fetch_object_children(cluster_name)
        click.secho(f'Found {len(literature_sources.index)}.')

        click.secho(f'{len(literature_sources.index)} sources both in Simbad and Gaia DR3')

        literature_sources.to_csv(LITERATURE_FILEPATH, index=None)

        click.secho(f'Saved sources from Simbad to {LITERATURE_FILEPATH}!', fg='green', bold=True)
        
    else:
        click.secho(f'{LITERATURE_FILEPATH} exists and --overwrite=False.', fg='yellow', bold=True)


    plot_on_aitoff(sources, cluster_name, radius)
    plt.show()

if __name__ == '__main__':
    download_sources_for_cluster()
