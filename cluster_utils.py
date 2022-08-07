import pandas as pd
from astropy.coordinates import ICRS, SkyCoord
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.metrics import silhouette_score


KAPPA: float = 4.74047
    
    
def dbscan(sources: pd.DataFrame, eps: float = 0.5, min_samples=20) -> np.array:
    dbscan_clust = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_fit = dbscan_clust.fit(sources)

    dbscan_labels = dbscan_fit.labels_
    unique_labels = set(dbscan_labels)
    print(f'Classified into {len(unique_labels)} labels.')
    return dbscan_labels

def label_sources(sources: pd.DataFrame, labels: np.array) -> pd.DataFrame:
    sources_labelled = pd.DataFrame(columns=[*sources.columns, 'label'],
                                    data=np.concatenate([sources,
                                                         labels.reshape((-1, 1))], axis=1))
    sources_labelled['label'] = sources_labelled.label.astype(int)
    return sources_labelled

def check_for_cluster_children(cluster_sources: pd.DataFrame,
                               literature: pd.DataFrame) -> bool:
    indices = np.in1d(literature['EDR3 id'].values, cluster_sources['source_id'].values)
    return literature[indices], literature[~indices]

def get_clustered_and_noise(labelled_sources: pd.DataFrame,
                            labels: np.array,
                            literature: np.array) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    non_noise_labels: np.array = labels[labels!=-1]
    unique_label_count = np.unique(non_noise_labels, return_counts=True)
    
    if len(unique_label_count) == 0:
        return None, None
    
    largest_values: np.ndarray = np.flip(np.argsort(unique_label_count[1]))
    largest_non_noise: np.ndarray = unique_label_count[0][largest_values]
    
    clustered_numbers: pd.DataFrame = np.array(
        [len(
            check_for_cluster_children(
                labelled_sources[labelled_sources['label']==lnn],
                literature
                )[0]
            ) for lnn in largest_non_noise
        ]
    )
    largest_cluster: int = np.max(clustered_numbers)
    best_label: int = largest_non_noise[np.argmax(clustered_numbers)]
    
    clustered: pd.DataFrame = labelled_sources[labelled_sources['label']==best_label]
    noise: pd.DataFrame = labelled_sources[labelled_sources['label']!=best_label]
        
    print(f'Clustered: {largest_cluster}/{len(literature)}')
        
    return clustered, noise, best_label


def heuristic_silhouette_score(sources_normalized: np.ndarray,
                               dbscan_labels: np.ndarray,
                               cluster_label: int,
                               trials: int = 2500) -> float:
    normalized_clustered_indices = np.argwhere(dbscan_labels==cluster_label).reshape(-1)
    clustered_labels = dbscan_labels[normalized_clustered_indices]
    normalized_clustered = sources_normalized[normalized_clustered_indices]

    normalized_unclustered_indices = np.argwhere(dbscan_labels!=cluster_label).reshape(-1)
    unclustered_labels = dbscan_labels[normalized_unclustered_indices]
    normalized_unclustered = sources_normalized[normalized_unclustered_indices]
    
    means = []
    
    for _ in range(5):
        rand_indices_c = np.random.choice(np.arange(len(normalized_clustered)), size=(trials,))
        rand_indices = np.random.choice(np.arange(len(normalized_unclustered)), size=(trials,))
        means.append(silhouette_score(np.concatenate([normalized_clustered[rand_indices_c], normalized_unclustered[rand_indices]]),
                                      np.concatenate([clustered_labels[rand_indices_c], unclustered_labels[rand_indices]])))
    return np.mean(means)


def cluster_plot(clustered_sources: pd.DataFrame,
                 noise_sources: pd.DataFrame,
                 paper_sources: pd.DataFrame,
                 plot_noise: bool = True):
    plt.figure(figsize=(20, 10))
    plt.scatter(clustered_sources.ra, clustered_sources.dec, 
                color='violet', label='Clustered', zorder=2, s=20.)
    if plot_noise:
        plt.scatter(noise_sources.ra, noise_sources.dec, color='skyblue', label='Unclustered', zorder=1, s=5.)
    plt.scatter(paper_sources.ra, paper_sources.dec, color='black', zorder=1,
                marker='D', label='Reported in papers')
    lgnd = plt.legend(fontsize=14);
    plt.gca().set_xlabel('$\\alpha$ [deg]', fontsize=20);
    plt.gca().set_ylabel('$\delta$ [deg]', fontsize=20);

    for handle in lgnd.legendHandles:
        handle._sizes = [30];
        
        
def plot_arrow(ra: float, dec: float, pmra: float, pmdec: float):
    plt.arrow(ra, dec, pmra, pmdec,
              linewidth=3., color='mediumseagreen',
              head_width=0.1, head_length=0.1)
        
        
def cluster_plot_galactic(clustered_sources: pd.DataFrame,
                          noise_sources: pd.DataFrame,
                          paper_sources: pd.DataFrame,
                          plot_noise: bool = True):
    plt.figure(figsize=(20, 10))
    clustered_coordinates = SkyCoord(ra=clustered_sources.ra*u.deg,
                                     dec=clustered_sources.dec*u.deg,
                                     frame=ICRS).galactic
    
    if plot_noise:
        noise_coordinates = SkyCoord(ra=noise_sources.ra*u.deg,
                                     dec=noise_sources.dec*u.deg,
                                     frame=ICRS).galactic
    paper_coordinates = SkyCoord(ra=paper_sources.ra*u.deg,
                                 dec=paper_sources.dec*u.deg,
                                 frame=ICRS).galactic
    
    plt.scatter(clustered_coordinates.l, clustered_coordinates.b, 
                color='violet', label='Clustered', zorder=2, s=20.)
    if plot_noise:
        plt.scatter(noise_coordinates.l, noise_coordinates.b, color='skyblue', label='Unclustered', zorder=1, s=5.)
    plt.scatter(paper_coordinates.l, paper_coordinates.b, color='black', zorder=1,
                marker='D', label='Reported in papers')
    lgnd = plt.legend(fontsize=14);
    plt.gca().set_xlabel('$l$ [deg]', fontsize=20);
    plt.gca().set_ylabel('$b$ [deg]', fontsize=20);

    for handle in lgnd.legendHandles:
        handle._sizes = [30];
        
        
def plot_arrow_galactic(ra: float, dec: float, parallax: float, pmra: float, pmdec: float):
    galactic_coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg,
                               distance=(1/parallax)*u.kpc,
                               pm_ra_cosdec=pmra*u.mas/u.year,
                               frame=ICRS,
                               pm_dec=pmdec*u.mas/u.year).galactic
    plt.arrow(galactic_coords.l.value,
              galactic_coords.b.value,
              galactic_coords.pm_l_cosb.value,
              galactic_coords.pm_b.value,
              linewidth=3., color='mediumseagreen',
              head_width=0.1, head_length=0.1)
    