from cluster_utils import (dbscan, label_sources,
                           get_clustered_and_noise, check_for_cluster_children, heuristic_silhouette_score, cluster_plot_galactic)
import optuna
import click
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import os


@click.command()
@click.argument('cluster_name', type=str)
@click.option('-nt', '--n-trials', type=int, default=10, help='Number of trials in the hyperparameter optimization process')
def dbscan_cluster(cluster_name: str, n_trials: int):
    
    ROOT_PATH: str = f'data/{cluster_name}'
    SQLITE_PATH: str = f'optuna_db/{cluster_name}'
    
    if not os.path.exists(SQLITE_PATH):
        os.makedirs(SQLITE_PATH)
    
    normalized_sources: np.ndarray = np.loadtxt(f'{ROOT_PATH}/{cluster_name}_normalized.dat')
    sources: pd.DataFrame = pd.read_csv(f'{ROOT_PATH}/{cluster_name}.csv')
    from_lit: pd.DataFrame = pd.read_csv(f'{ROOT_PATH}/{cluster_name}_literature.csv')
    
    from_lit_edr3 = from_lit_edr3 = from_lit[np.in1d(from_lit['EDR3 id'].values, sources['source_id'].values)]
    
    def objective(trial):
        eps = trial.suggest_float('eps', 0.01, 0.2)
        min_samples = trial.suggest_int('min_samples', 2, 20)
        
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
        return len(found)/len(from_lit_edr3)+silh_score
        
    replaced_cluster_name = cluster_name.replace(' ', '_')
    
    study = optuna.create_study(direction="maximize",
                                study_name=replaced_cluster_name,
                                storage=f'sqlite:///{SQLITE_PATH}/{replaced_cluster_name}.db',
                                load_if_exists=True,
                                sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    click.secho(f'Best trial: {best_params} with value of {study.best_value}', fg='yellow', bold=True)
    
    with open(f'optuna_db/{replaced_cluster_name}_best_params.json', 'w') as f:
        json.dump({'best_value': study.best_value, 'best_params': best_params}, f)
        
    dbscan_labels = dbscan(normalized_sources, best_params['eps'], best_params['min_samples'])

    labelled_sources = label_sources(sources, dbscan_labels)
    clustered, noise, _ = get_clustered_and_noise(labelled_sources, dbscan_labels, from_lit_edr3)
    
    clustered.to_csv(f'{ROOT_PATH}/{cluster_name}_clustered.csv')
    click.secho(f'Saved clustered sources to {ROOT_PATH}/{cluster_name}_clustered.csv!', fg='green', bold=True)
    
    optuna.visualization.plot_contour(study)
    plt.show()
    cluster_plot_galactic(clustered, noise, from_lit_edr3)
    plt.show()
  
    
if __name__ == '__main__':
    dbscan_cluster()