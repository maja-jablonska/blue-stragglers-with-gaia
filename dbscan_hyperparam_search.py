from sklearn.cluster import DBSCAN
import numpy as np
import click


@click.command()
@click.argument('filepath', type=str)
def download_sources_for_cluster(filepath: str):
    normalized_sources: np.array = np.loadtxt(filepath)
    is_in_edr3 = is_in_cluster_function(sources_to_cluster)
    from_lit_edr3 = from_lit[np.vectorize(is_in_edr3)(from_lit['EDR3 id'].values)]