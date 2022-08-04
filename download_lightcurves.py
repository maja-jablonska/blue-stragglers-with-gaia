import pandas as pd
import numpy as np
from astroquery.gaia import Gaia
from typing import List
import matplotlib.pyplot as plt


def download_lightcurves(source_ids: np.ndarray) -> List[pd.DataFrame]:
    retrieval_type = 'ALL'          # Options are: 'EPOCH_PHOTOMETRY', 'MCMC_GSPPHOT', 'MCMC_MSC', 'XP_SAMPLED', 'XP_CONTINUOUS', 'RVS', 'ALL'
    data_structure = 'INDIVIDUAL'   # Options are: 'INDIVIDUAL', 'COMBINED', 'RAW'
    data_release   = 'Gaia DR3'     # Options are: 'Gaia DR3' (default), 'Gaia DR2'

    
    lightcurves: List[pd.DataFrame] = []

    datalink = Gaia.load_data(ids=source_ids,
                              data_release = data_release,
                              retrieval_type= 'EPOCH_PHOTOMETRY',
                              data_structure = data_structure,
                              verbose = False, output_file = None)
    dl_keys  = [inp for inp in datalink.keys()]
    dl_keys.sort()

    print(f'{len(dl_keys)} lightcurves found.')
    for dl_key in dl_keys:
        print(f'\tDownloading {dl_key}')
        lightcurves.append(datalink[dl_key][0].to_table().to_pandas())
        
    return lightcurves


def get_dr3_band_lightcurve(lc: pd.DataFrame, band: str = 'G') -> pd.DataFrame:
    df = lc[lc.band==band][['time', 'flux', 'mag', 'flux_error']]
    df['time'] += 2455197.5
    return df


def plot_dr3_lc(lc: pd.DataFrame):
    plt.figure()
    plt.errorbar(x=lc.time, y=lc.flux, yerr=lc.flux_error, color='black', fmt='o')
    plt.gca().set_ylabel('Flux', fontsize=16);
    plt.gca().set_xlabel('Time [JD]', fontsize=16);
    
    
@click.command()
@click.argument('cluster_name', type=str)
def download_lightcurves(cluster_name: str):
    
    click.secho(f'Downloading lightcurves for BSS and YSS candidates in {cluster_name}...')
    
    PATH_ROOT: str = f'data/{cluster_name}'
        
    bss_candidates: pd.DataFrame = pd.read_csv(f'{PATH_ROOT}/{cluster_name}/bss.csv')
    yss_candidates: pd.DataFrame = pd.read_csv(f'{PATH_ROOT}/{cluster_name}/yss.csv')
        
        
    # 1. Check for stars marked as non-single
    bss_non_single = bss_candidates[bss_candidates.non_single_star!=0]
    print(f'Non single BSS stars: {bss_non_single}')
    
    bss_phot = bss_candidates[bss_candidates.phot_variable_flag=='VARIABLE']
    print(f'Photometrically variable BSS stars: {bss_phot}')
    
    dr3_lightcurves = download_lightcurves(bss_phot.source_id.values)
    for lc in dr3_lightcurves:
        lc.to_csv(f'{PATH_ROOT}/{lc.source_id.values[0]}_lc.csv');
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    