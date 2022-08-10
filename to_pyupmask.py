import click
import numpy as np
from astropy.table import Table
from astropy.io import ascii


@click.command()
@click.argument('cluster_name', type=str)
def to_pyupmask(cluster_name: str):
    
    ROOT_PATH: str = f'data/{cluster_name}'
    PYUPMASK_INPUT_PATH: str = f'pyUPMASK/input'
    
    sources_normalized = np.loadtxt(f'{ROOT_PATH}/{cluster_name}_normalized.dat')
    uncertainties_normalized = np.loadtxt(f'{ROOT_PATH}/{cluster_name}_normalized_uncert.dat')
    
    data = Table()
    data['_x'] = sources_normalized[:, 0]
    data['_y'] = sources_normalized[:, 1]
    data['Plx'] = sources_normalized[:, 2]
    data['pmRA'] = sources_normalized[:, 3]
    data['pmDE'] = sources_normalized[:, 4]
    data['e_Plx'] = uncertainties_normalized[:, 2]
    data['e_pmRA'] = uncertainties_normalized[:, 3]
    data['e_pmDE'] = uncertainties_normalized[:, 4]
    ascii.write(data, f'{PYUPMASK_INPUT_PATH}/{cluster_name}.dat', overwrite=True) 
    
    click.secho(f'Saved to {PYUPMASK_INPUT_PATH}/{cluster_name}.dat', fg='green', bold=True)


if __name__ == '__main__':
    to_pyupmask()
