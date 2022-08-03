from cluster_utils import (dbscan, is_in_cluster_function, label_sources,
                           get_clustered_and_noise, check_for_cluster_children, contamination, cluster_plot)
from simbad_download import resolve_name
import optuna
import click
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt


@click.command()
@click.argument('cluster_name', type=str)
@click.option('-nt', '--n-trials', type=int, default=10, help='Number of trials in the hyperparameter optimization process')
def dbscan_cluster(cluster_name: str, n_trials: int):
    
    normalized_sources: np.ndarray = np.loadtxt(f'data/{cluster_name}/{cluster_name}_normalized.dat')
    sources: pd.DataFrame = pd.read_csv(f'data/{cluster_name}/{cluster_name}.csv')
    from_lit: pd.DataFrame = pd.read_csv(f'data/{cluster_name}/{cluster_name}_literature.csv')
       
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
    
    replaced_cluster_name = cluster_name.replace(' ', '_')
    
    study = optuna.create_study(direction="maximize",
                                study_name=replaced_cluster_name,
                                storage=f'sqlite:///optuna_db/{replaced_cluster_name}.db',
                                load_if_exists=True,
                                sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    click.secho(f'Best trial: {best_params} with value of {study.best_value}', fg='yellow', bold=True)
    
    with open(f'optuna_db/{replaced_cluster_name}_best_params.json', 'w') as f:
        json.dump({'best_value': study.best_value, 'best_params': best_params}, f)
        
    dbscan_labels = dbscan(normalized_sources, best_params['eps'], best_params['min_samples'])

    labelled_sources = label_sources(sources, dbscan_labels)
    clustered, noise = get_clustered_and_noise(labelled_sources, dbscan_labels)
    in_cluster = check_for_cluster_children(from_lit_edr3, clustered)
    
    clustered.to_csv(f'data/{cluster_name}/{cluster_name}_clustered.csv')
    click.secho(f'Saved clustered sources to data/{cluster_name}/{cluster_name}_clustered.csv!', fg='green', bold=True)
    
    optuna.visualization.plot_contour(study)
    plt.show()
    cluster_plot(clustered, noise, from_lit_edr3)
    plt.show()
  
    
if __name__ == '__main__':
    dbscan_cluster()