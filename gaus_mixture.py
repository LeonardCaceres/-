def em_clustering(data, n_clusters = 3, max_iter=100, tol=1e-4):
    np.random.seed(42)
    n_samples, n_features = data.shape

    means = np.random.rand(n_clusters, n_features)
    covariances = np.array([np.eye(n_features) for _ in range(n_clusters)])
    weights = np.ones(n_clusters) / n_clusters
    for _ in range(max_iter):
        # E-step
        likelihoods = np.array([multivariate_normal_pdf(data, mean=means[i],
                                                        cov=covariances[i]) for i in range(n_clusters)]).T
        weighted_likelihoods = likelihoods * weights
        cluster_probs = weighted_likelihoods / weighted_likelihoods.sum(axis=1)[:, np.newaxis]

        # M-step
        new_means = np.dot(cluster_probs.T, data) / cluster_probs.sum(axis=0)[:, np.newaxis]
        new_covariances = np.array([np.dot((data - new_means[i]).T,
                                           np.dot(np.diag(cluster_probs[:, i]),
                                                  (data - new_means[i]))) / cluster_probs[:, i].sum() for i in range(n_clusters)])
        new_weights = cluster_probs.sum(axis=0) / n_samples

        if np.linalg.norm(new_means - means) < tol:
            break

        means = new_means
        covariances = new_covariances
        weights = new_weights

    return means, covariances, weights, np.array([multivariate_normal.pdf(data, mean=means[i], cov=covariances[i]) for i in range(n_clusters)]).T
