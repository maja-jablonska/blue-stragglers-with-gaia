from simbad_download import resolve_name
import pandas as pd
from astropy.coordinates import ICRS, SkyCoord
import astropy.units as u
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.cluster import DBSCAN
import optuna
import click


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

def get_clustered_and_noise(labelled_sources: pd.DataFrame, labels: np.array) -> pd.DataFrame:
    non_noise_labels: np.array = labels[labels!=-1]
    unique_label_count = np.unique(non_noise_labels, return_counts=True)
    
    if len(unique_label_count) == 0:
        return None, None
    
    largest_non_noise: int = unique_label_count[0][np.argmax(unique_label_count[1])]
    
    clustered: pd.DataFrame = labelled_sources[labelled_sources['label']==largest_non_noise]
    noise: pd.DataFrame = labelled_sources[labelled_sources['label']!=largest_non_noise]
        
    print(f'Clustered: {len(clustered)}/{len(labelled_sources)}')
        
    return clustered, noise

def is_in_cluster_function(cluster_sources: pd.DataFrame,
                           source_id_col_name: str = 'source_id'):
    def is_in_cluster(source_id: int) -> bool:
        return len(cluster_sources[cluster_sources[source_id_col_name]==source_id]) > 0
    return is_in_cluster

def check_for_cluster_children(from_lit: pd.DataFrame,
                               cluster_sources: pd.DataFrame) -> bool:
    is_in_cluster = is_in_cluster_function(cluster_sources)
    in_cluster = np.vectorize(is_in_cluster)(from_lit['EDR3 id'].values)
    trues = in_cluster[in_cluster]
    print(f'{len(trues)}/{len(in_cluster)} objects found in the cluster.')
    return in_cluster

def check_for_undiscovered_sources(from_lit: pd.DataFrame,
                                   cluster_sources: pd.DataFrame) -> bool:
    is_in_cluster = is_in_cluster_function(from_lit, 'EDR3 id')
    in_cluster = np.vectorize(is_in_cluster)(cluster_sources['source_id'].values)
    falses = in_cluster[~in_cluster]
    print(f'{len(falses)}/{len(in_cluster)} objects were previously unreported.')
    return len(falses)/len(in_cluster)


def contamination(clustered: pd.DataFrame,
                  all_sources: pd.DataFrame) -> int:
    return (len(clustered[clustered.ra<np.quantile(all_sources.ra, .1)]) +
            len(clustered[clustered.ra>np.quantile(all_sources.ra, .9)]) +
            len(clustered[clustered.dec<np.quantile(all_sources.dec, .1)]) +
            len(clustered[clustered.dec>np.quantile(all_sources.dec, .9)]))


@click.command()
@click.argument('cluster_name', type=str)
def dbscan_cluster(cluster_name: str):
    
    normalized_sources: np.ndarray = np.loadtxt(f'{cluster_name}_normalized.dat')
    sources: pd.DataFrame = pd.read_csv(f'{cluster_name}.csv')
    from_lit: pd.DataFrame = pd.read_csv(f'{cluster_name}_literature.csv')
       
    cp_ra, cp_dec, cp_par, cp_pmra, cp_pmdec, cp_radvel = resolve_name(cluster_name)
    
    is_in_edr3 = is_in_cluster_function(sources)
    from_lit_edr3 = from_lit[np.vectorize(is_in_edr3)(from_lit['EDR3 id'].values)]
    
    def objective(trial):
        eps = trial.suggest_float('eps', 0.01, 0.2)
        min_samples = trial.suggest_int('min_samples', 2, 20)

        dbscan_labels = dbscan(normalized_sources, eps, min_samples)
        if len(np.unique(dbscan_labels)) == 1:
            return -len(sources)

        labelled_sources = label_sources(sources, dbscan_labels)
        clustered, noise = get_clustered_and_noise(labelled_sources, dbscan_labels)
        in_cluster = check_for_cluster_children(from_lit_edr3, clustered)

        return len(from_lit_edr3[in_cluster])-contamination(clustered, sources)
    
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    
    click.secho(f'Best trial: {study.best_params} with value of {study.best_value}', fg='yellow', bold=True)
    
    
if __name__ == '__main__':
    dbscan_cluster()