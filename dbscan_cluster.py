from cluster_utils import (dbscan, label_sources,
                           get_clustered_and_noise,
                           check_for_cluster_children,
                           heuristic_silhouette_score,
                           cluster_plot_galactic)
import optuna
import click
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Tuple


def get_lit_from_edr3(literature: pd.DataFrame,
                      edr3: pd.DataFrame) -> pd.DataFrame:
    lit_in_edr3 = literature[np.in1d(literature['EDR3 id'].values, edr3['source_id'].values)]
    print(f'{len(lit_in_edr3)}/{len(edr3)} sources from literature are also in EDR3')
    return lit_in_edr3


def run_dbscan(normalized_sources: np.ndarray,
               sources: pd.DataFrame,
               literature: pd.DataFrame,
               eps: float,
               min_samples: float,
               calculate_silhouette_score: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dbscan_labels = dbscan(normalized_sources, eps, min_samples)
    labelled_sources = label_sources(sources, dbscan_labels)
    clustered, noise, cluster_label = get_clustered_and_noise(labelled_sources, dbscan_labels, literature)
    found, not_found = check_for_cluster_children(clustered, literature)
        
    if calculate_silhouette_score:
        silh_score = heuristic_silhouette_score(normalized_sources, dbscan_labels, cluster_label, 5000)
        print(f'Heuristic silhouette score: {silh_score}')
    
    print(f'{len(found)}/{len(literature)} sources from literature clustered')
    print(f'{len(clustered)} clustered sources.')
    return clustered, noise, found, not_found


@click.command()
@click.argument('cluster_name', type=str)
@click.option('-cp', is_flag=True, default=False, help='Use the convergent point method')
@click.option('-nt', '--n-trials', type=int, default=10, help='Number of trials in the hyperparameter optimization process')
@click.option('-r', '--reset', is_flag=True, default=False, help='Reset the hyperparameter search (dont load saved search')
def dbscan_cluster(cluster_name: str, cp: bool, n_trials: int, reset: bool):
    
    ROOT_PATH: str = f'data/{cluster_name}'
    SQLITE_PATH: str = f'optuna_db/{cluster_name}'
    
    if not os.path.exists(SQLITE_PATH):
        os.makedirs(SQLITE_PATH)
    
    if cp:
        normalized_sources: np.ndarray = np.loadtxt(f'{ROOT_PATH}/{cluster_name}_normalized_cp.dat')
    else:
        normalized_sources: np.ndarray = np.loadtxt(f'{ROOT_PATH}/{cluster_name}_normalized.dat')
    sources: pd.DataFrame = pd.read_csv(f'{ROOT_PATH}/{cluster_name}.csv')
    from_lit: pd.DataFrame = pd.read_csv(f'{ROOT_PATH}/{cluster_name}_literature.csv')
    
    from_lit_edr3 = get_lit_from_edr3(from_lit, sources)
    
    def objective(trial):
        eps = trial.suggest_float('eps', 0.01, 0.5)
        min_samples = trial.suggest_int('min_samples', 2, 50)
        
        dbscan_labels = dbscan(normalized_sources, eps, min_samples)
        if len(np.unique(dbscan_labels)) == 1:
            return -len(sources)
            
        labelled_sources = label_sources(sources, dbscan_labels)
        clustered, _, cluster_label = get_clustered_and_noise(labelled_sources, dbscan_labels, from_lit_edr3)
        
        print(f'{len(clustered)} clustered.')
        if len(clustered)>2*len(from_lit_edr3):
            return -len(clustered)/len(from_lit_edr3)
        
        found, _ = check_for_cluster_children(clustered, from_lit_edr3)
        
        silh_score = heuristic_silhouette_score(normalized_sources, dbscan_labels, cluster_label, 5000)
        print(silh_score)
        return len(found)/len(from_lit_edr3)+silh_score*len(clustered)/len(from_lit_edr3)
        
    replaced_cluster_name = cluster_name.replace(' ', '_')
    if cp:
        replaced_cluster_name = replaced_cluster_name + '_cp'
    
    if reset:
        optuna.delete_study(study_name=replaced_cluster_name,
                            storage=f'sqlite:///{SQLITE_PATH}/{replaced_cluster_name}.db')
    
    study = optuna.create_study(direction="maximize",
                                study_name=replaced_cluster_name,
                                storage=f'sqlite:///{SQLITE_PATH}/{replaced_cluster_name}.db',
                                load_if_exists=True,
                                sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    click.secho(f'Best trial: {best_params} with value of {study.best_value}', fg='yellow', bold=True)
    
    with open(f'{SQLITE_PATH}/{replaced_cluster_name}_best_params.json', 'w') as f:
        json.dump({'best_value': study.best_value, 'best_params': best_params}, f)
        
    clustered, noise, found, not_found = run_dbscan(normalized_sources, sources, from_lit_edr3,
                                             best_params['eps'], best_params['min_samples'])
    
    clustered.to_csv(f'{ROOT_PATH}/{cluster_name}_clustered.csv')
    found.to_csv(f'{ROOT_PATH}/{cluster_name}_found.csv')
    not_found.to_csv(f'{ROOT_PATH}/{cluster_name}_not_found.csv')
    click.secho(f'Saved clustered sources to {ROOT_PATH}/{cluster_name}_clustered.csv!', fg='green', bold=True)
    
    optuna.visualization.plot_contour(study)
    plt.show()
    cluster_plot_galactic(clustered, noise, from_lit_edr3)
    plt.show()
  
    
if __name__ == '__main__':
    dbscan_cluster()