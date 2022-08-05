import pandas as pd
import lightkurve as lk
import matplotlib.pyplot as plt

from typing import List
from lightkurve.correctors import DesignMatrix

def download_lc(tic: int, cutout_size: int = 10) -> List[lk.LightCurve]:
    ts = lk.search_tesscut(f'TIC {int(tic)}')
    print(f'Downloaded {len(ts)} tesscuts.')
    
    _, ax = plt.subplots(ncols=3, nrows=len(ts))
    
    light_curves: List[lk.lightcurve] = []
    
    for i, cut in enumerate(ts):
        try:
            tpfs = cut.download(cutout_size=cutout_size)

            aper = tpfs.create_threshold_mask()
            regressors = tpfs.flux[:, ~aper]
            dm = DesignMatrix(regressors, name='regressors')
            uncorrected_lc = tpfs.to_lightcurve(aperture_mask=aper)
            uncorrected_lc.plot(ax=ax[i, 1]);

            corrector = lk.RegressionCorrector(uncorrected_lc)
            corrected_lc = corrector.correct(dm)
            corrected_lc.plot(ax=ax[i, 2])

            tpfs.plot(ax[i, 0], aperture_mask=aper)
            ax[i, 0].set_title('')
            light_curves.append(corrected_lc)
        except:
            ax[i, 0].set_visible(False);
            ax[i, 1].set_visible(False);
            ax[i, 2].set_visible(False);
            continue
            
    plt.show()
    
    return light_curves 