import numpy as np
import pandas as pd
import gaia_download as gd
import matplotlib.pyplot as plt
import click
from download_isochrone import load_isochrone
from scipy.interpolate import interp1d
import os


@click.command()
@click.argument('cluster_name', type=str)
def select_bss_candidates(cluster_name: str):
    
    click.secho(f'Selecting BSS candidates using the isochrone for {cluster_name}.', fg='yellow', bold=True)
    
    PATH_ROOT: str = f'data/{cluster_name}'
    
    clustered_sources: pd.DataFrame = pd.read_csv(f'{PATH_ROOT}/{cluster_name}_clustered.csv')
    isochrone: np.ndarray = load_isochrone(f'data/{cluster_name}/{cluster_name}_isochrone.dat')
    eq_mass_isochrone = isochrone+np.array([0., -0.75])
    
    sources = clustered_sources[['BP-RP', 'G_abs']].values
    BOUNDS = np.max(isochrone, axis=0)-np.min(isochrone, axis=0)
    
    def closest(source, isochrone):
        closest = isochrone[
        np.argsort(
                np.linalg.norm((source-isochrone), axis=1).flatten()
            )
        ][:5]
        return closest[np.argmin(closest[:, 0])], closest[np.argmax(closest[:, 0])]

    def distance_to_closest(source, isochrone):
        src_rescaled = source/BOUNDS
        isochrone_rescaled = isochrone/BOUNDS
        p1, p2 = closest(src_rescaled, isochrone_rescaled)
        return np.linalg.norm(np.cross((p2-p1), (p1-src_rescaled)))/np.linalg.norm((p2-p1))

    def above_isochrone(source, isochrone):
        closest = isochrone[
            np.argsort(
                    np.linalg.norm((source-isochrone), axis=1).flatten()
                )
        ][:3]
        p1 = closest[np.argmin(closest[:, 0])]
        p2 = closest[np.argmax(closest[:, 0])]
        v1 = (p2[0]-p1[0], p2[1]-p1[1])   # Vector 1
        v2 = (p2[0]-source[0], p2[1]-source[1])   # Vector 2
        xp = v1[0]*v2[1] - v1[1]*v2[0]  # Cross product

        return xp>=0
    
    dists = np.apply_along_axis(
        lambda x: distance_to_closest(x, isochrone), 1,
        sources
    )
    
    above_equal_binary_limit = np.apply_along_axis(
        lambda x: above_isochrone(x, eq_mass_isochrone), 1,
        sources
    )
    
    dist_std = np.nanstd(dists)

    fig = plt.figure(figsize=(10, 8));
    plt.scatter(clustered_sources['BP-RP'], clustered_sources['G_abs'],
                c=dists, cmap='turbo', label='Gaia (BP-RP)')
    plt.colorbar()
    plt.gca().set_ylabel('G_abs', fontsize=16);
    plt.gca().set_xlabel('color', fontsize=16);
    plt.plot(isochrone[:, 0], isochrone[:, 1], label='PARSEC isochrone', color='tomato')
    plt.plot(isochrone[:, 0], isochrone[:, 1]-0.75, label='Equal mass binary limit', color='tomato', linestyle='--')
    plt.gca().invert_yaxis();
    plt.legend(fontsize=15);
    plt.show();
    
    plt.figure(figsize=(10, 8));

    TO_COLOR = np.min(isochrone[:, 0])
    TO_MAG = isochrone[np.argmin(isochrone[:, 0]), 1]

    plt.errorbar(clustered_sources['BP-RP'], clustered_sources['G_abs'],
                 color='skyblue', label='Gaia (BP-RP)', fmt='o', zorder=1)
    plt.gca().set_ylabel('G_abs', fontsize=16);
    plt.gca().set_xlabel('color', fontsize=16);
    plt.plot(isochrone[:, 0], isochrone[:, 1], label='PARSEC isochrone', color='tomato', zorder=3)
    plt.plot(isochrone[:, 0], isochrone[:, 1]-0.75,
             label='Equal mass binary limit',
             color='tomato', zorder=3,
             linestyle='--')
    
    bss_candidates = clustered_sources[((dists>=np.nanstd(dists)) & (clustered_sources['G_abs']<TO_MAG) &
                                        (clustered_sources['BP-RP']<1.2*TO_COLOR))]
    yss_candidates = clustered_sources[(clustered_sources['BP-RP']>1.2*TO_COLOR) & (clustered_sources['G_abs']<TO_MAG) &
                                       (((above_equal_binary_limit) & (dists>=np.nanstd(dists))))]
    plt.errorbar(bss_candidates['BP-RP'], bss_candidates['G_abs'],
                 xerr=bss_candidates['BP-RP_error'], yerr=bss_candidates['G_abs_error'],
                 color='royalblue', fmt='*', zorder=2, label='BSS candidates', markersize=10.)
    plt.errorbar(yss_candidates['BP-RP'], yss_candidates['G_abs'],
                 xerr=yss_candidates['BP-RP_error'], yerr=yss_candidates['G_abs_error'],
                 color='gold', fmt='*', zorder=2, label='YSS candidates', markersize=10.)
    plt.axvline(x=TO_COLOR, color='gray', linestyle='dotted')
    plt.gca().invert_yaxis();
    plt.legend(fontsize=15);
    plt.show();
    
    click.secho(f'{len(bss_candidates)} BSS candidates found and {len(yss_candidates)} YSS candidates found.',
                fg='yellow', bold=True)
    
    BSS_FILENAME: str = f'data/{cluster_name}/{cluster_name}_bss.csv'
    YSS_FILENAME: str = f'data/{cluster_name}/{cluster_name}_yss.csv'
    
    bss_candidates.to_csv(BSS_FILENAME, index=None);
    yss_candidates.to_csv(YSS_FILENAME, index=None);
    click.secho(f'Saved BSS and YSS candidates to {BSS_FILENAME} and {YSS_FILENAME}!', fg='green', bold=True)
    
if __name__ == '__main__':
    select_bss_candidates()