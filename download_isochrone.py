import click
import mechanize
from mechanize._html import content_parser
import requests as re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional


def load_isochrone(filename: str) -> np.ndarray:
    isochrone_raw: np.ndarray = np.loadtxt(filename, usecols=(28, 29, 30))
    isochrone_raw = isochrone_raw[isochrone_raw[:, 0]<15]
    isochrone: np.ndarray = np.concatenate([(isochrone_raw[:-1, 1]-isochrone_raw[:-1, 2]).reshape(-1, 1),
                                             isochrone_raw[:-1, 0].reshape(-1, 1)], axis=1)
    return isochrone


def plot_isochrone(isochrone: np.ndarray, ax: Optional[plt.axis] = None, color: Optional[str] = None):
    
    if color is None:
        color = 'royalblue'
    
    if ax is None:
        plt.plot(isochrone[:, 0], isochrone[:, 1], color=color, label='isochrone');
        plt.gca().invert_yaxis();
        plt.gca().set_xlabel("$G_{BP}-G_{RP}$", fontsize=16);
        plt.gca().set_ylabel("$G_{abs}$", fontsize=16);
        plt.legend(fontsize=14);
    else:
        ax.plot(isochrone[:, 0], isochrone[:, 1], color=color);
    


@click.command()
@click.argument('cluster_name', type=str)
@click.option('-a', '--age', help='Age in scientific notation as str [e.g. \'6.5e9\']', required=True, type=str)
@click.option('-Z', '--metallicity', help='Metallicity as percentage [e.g. 0.128]', required=True, type=float)
def download_isochrone(cluster_name: str, age: str, metallicity: float):
    
    clustered_sources: pd.DataFrame = pd.read_csv(f'data/{cluster_name}/{cluster_name}_clustered.csv')
    
    click.secho(f'Downloading the isochrone for {cluster_name} with age of {age} and metallicity {metallicity}...')
    
    br = mechanize.Browser()
    br.open("http://stev.oapd.inaf.it/cgi-bin/cmd_3.7")
    br.select_form(action='./cmd_3.7')
    
    br['isoc_agelow'] = str(age)
    br['isoc_zlow'] = str(metallicity)
    br['photsys_file'] = ['YBC_tab_mag_odfnew/tab_mag_gaiaEDR3.dat']
    
    response = br.submit()
    html = content_parser(response.get_data())
    output_href = html.find("body").find("form").find("fieldset").find("p").find("a").attrib['href'].split('./')[-1]
    r = re.get(f'http://stev.oapd.inaf.it/{output_href}', allow_redirects=True)
    
    if r.status_code != 200:
        click.secho('Something went wrong while fetching the file :(', fg='red', bold=True)
        return
    
    FILENAME: str = f'data/{cluster_name}/{cluster_name}_isochrone.dat'
    open(FILENAME, 'wb').write(r.content)
    click.secho(f'Saved the isochrone to {FILENAME}', fg='green', bold=True)
    
    isochrone = load_isochrone(FILENAME)
    plot_isochrone(isochrone)
    
    plt.errorbar(clustered_sources['BP-RP'], clustered_sources['G_abs'],
                 color='skyblue', label='Gaia (BP-RP)', fmt='o', zorder=1)
    plt.show()


if __name__ == '__main__':
    download_isochrone()