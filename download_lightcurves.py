import pandas as pd
import numpy as np
import lightkurve as lk
from typing import List
import matplotlib.pyplot as plt
from tess import download_lc
from gaia_download import download_dr3_lightcurve
import click
import pickle
from typing import List


@click.command()
@click.argument('cluster_name', type=str)
def download_lightcurves(cluster_name: str):
    
    ROOT_PATH: str = f'data/{cluster_name}'
    
    found = pd.read_csv(f'{ROOT_PATH}/{cluster_name}_found.csv')
    bss_candidates = pd.read_csv(f'{ROOT_PATH}/{cluster_name}_bss.csv')
    yss_candidates = pd.read_csv(f'{ROOT_PATH}/{cluster_name}_yss.csv')
    
    
    bss_candidates = pd.merge(left=bss_candidates, right=found[['EDR3 id', 'TIC']], left_on='source_id',
                              right_on='EDR3 id').drop(columns=['EDR3 id'])
    
    yss_candidates = pd.merge(left=yss_candidates, right=found[['EDR3 id', 'TIC']], left_on='source_id',
                              right_on='EDR3 id').drop(columns=['EDR3 id'])
    
    # Have epoch photometry
    bss_epoch_phot = bss_candidates[bss_candidates.has_epoch_photometry]
    yss_epoch_phot = yss_candidates[yss_candidates.has_epoch_photometry]
    
    bss_lcs = download_dr3_lightcurve(bss_epoch_phot.source_id.values)
    for lc in bss_lcs:
        lc.to_csv(f'{ROOT_PATH}/bss_{lc.source_id.values[0]}_phot.csv')
        
    yss_lcs = download_dr3_lightcurve(yss_epoch_phot.source_id.values)
    for lc in yss_lcs:
        lc.to_csv(f'{ROOT_PATH}/yss_{lc.source_id.values[0]}_phot.csv')
        
    click.secho(f'Downloaded {len(bss_lcs)} DR3 lightcurves for BSS candidates and {len(yss_lcs)} for YSS candidates',
                fg='green', bold=True)
        
        
    # Have TIC
    bss_tic = bss_candidates.dropna(subset=['TIC'])
    yss_tic = yss_candidates.dropna(subset=['TIC'])
    
    for s_id, tic in zip(bss_tic.source_id.values, bss_tic['TIC'].values):
        tess_lcs = download_lc(int(tic))
        with open(f'{ROOT_PATH}/bss_{s_id}_tess.pickle', 'wb') as f:
            pickle.dump(tess_lcs, f)
            
    for s_id, tic in zip(yss_tic.source_id.values, yss_tic['TIC'].values):
        tess_lcs = download_lc(int(tic))
        with open(f'{ROOT_PATH}/yss_{s_id}_tess.pickle', 'wb') as f:
            pickle.dump(tess_lcs, f)
            
            
    click.secho(f'Downloaded {len(bss_tic)} TESS lightcurves for BSS candidates and {len(yss_tic)} for YSS candidates',
                fg='green', bold=True)
    
    
if __name__=='__main__':
    download_lightcurves()