def em_algorithm_mody(X, num_components, num_iterations):
    n, d = X.shape
    # Initialize parameters
    np.random.seed(42)
    means = np.random.rand(num_components, d)
    covariances = [np.eye(d) for _ in range(num_components)]
    weights = np.ones(num_components) / num_components
    scales = np.ones(num_components)  # Initialize scales

    for _ in range(num_iterations):
        # E-step
        probs = np.zeros((n, num_components))
        for i in range(num_components):
            if i < num_components - 1:
                probs[:, i] = weights[i] * (
                    multivariate_normal.pdf(X, mean=means[i], cov=covariances[i])
                )
            else:
                probs[:, i] = weights[i] * laplace_pdf(X[:, 0], X[:, 1], means[i][0], means[i][1], scales[i])
        probs_2 = probs / probs.sum(axis=1)[:, np.newaxis]
        probs = probs_2

        # M-step
        for i in range(num_components):
            weights[i] = np.sum(probs[:, i]) / n
            means[i] = np.sum(X * probs[:, i][:, np.newaxis], axis=0) / np.sum(probs[:, i])
            diff = X - means[i]
            covariances[i] = np.dot(diff.T, diff * probs[:, i][:, np.newaxis]) / np.sum(probs[:, i])
            if i == num_components - 1:
                scales[i] = np.sum(np.abs((X - means[i])) * probs[:, i][:, np.newaxis]) / np.sum(probs[:, i])

    return means, covariances, weights, scales, probs
