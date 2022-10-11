import pandas as pd


def mcl():
    import markov_clustering as mc
    import networkx as nx
    import random

    # number of nodes to use
    numnodes = 200

    # generate random positions as a dictionary where the key is the node id and the value
    # is a tuple containing 2D coordinates
    # {0: (-0.3206121045402026, 0.510517966036963), 1: (0.9975755548701939, 0.7765285792125876),
    positions = {i: (random.random() * 2 - 1, random.random() * 2 - 1) for i in range(numnodes)}

    # use networkx to generate the graph
    network = nx.random_geometric_graph(numnodes, 0.3, pos=positions)

    # then get the adjacency matrix (in sparse form)
    matrix = nx.to_scipy_sparse_matrix(network)
    result = mc.run_mcl(matrix)  # run MCL with default parameters
    clusters = mc.get_clusters(result)  # get clusters
    mc.draw_graph(matrix, clusters, pos=positions, node_size=50, with_labels=False, edge_color="silver")


def optics():
    from sklearn.cluster import OPTICS
    import numpy as np
    import matplotlib.pyplot as plt

    input = [[1, 2], [2, 5], [3, 6], [8, 7], [8, 8], [7, 3]]
    X = np.array(input)
    clustering = OPTICS(min_samples=3).fit(X)
    result = list(clustering.labels_)
    colors = np.random.uniform(0.2, 0.7, (len(set(result)), 3))
    # plt.scatter(x, y, marker='.', c=[colors[i]])
    print(result)
    for i, v in enumerate(result):
        plt.scatter(input[i][0], input[i][1], c=[colors[v]])
    plt.show()


def gmm_metrics():
    from sklearn.mixture import GaussianMixture
    import sklearn.metrics as metrics
    import matplotlib.pyplot as plt

    # Create empty list
    S = []

    # Range of clusters to try (2 to 10)
    K = range(2, 11)
    df = pd.DataFrame()
    # Select data for clustering model
    X = df[['Latitude', 'Longitude']]

    for k in K:
        # Set the model and its parameters
        model = GaussianMixture(n_components=k, n_init=20, init_params='kmeans')
        # Fit the model
        labels = model.fit_predict(X)
        # Calculate Silhoutte Score and append to a list
        S.append(metrics.silhouette_score(X, labels, metric='euclidean'))

    # Plot the resulting Silhouette scores on a graph
    plt.figure(figsize=(16, 8), dpi=300)
    plt.plot(K, S, 'bo-', color='black')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('Identify the number of clusters using Silhouette Score')
    plt.show()


if __name__ == '__main__':
    print('344BB577D484DCA602DFDF8616A3D9EB' == '344BB577D484DCA602DFDF8616A3D9EB')
