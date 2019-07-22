import numpy as np

def _retrieve_knn_faiss_gpu_inner_product(query_embeddings, db_embeddings, k, gpu_id=0):
    """
        Retrieve k nearest neighbor based on inner product

        Args:
            query_embeddings:           numpy array of size [NUM_QUERY_IMAGES x EMBED_SIZE]
            db_embeddings:              numpy array of size [NUM_DB_IMAGES x EMBED_SIZE]
            k:                          number of nn results to retrieve excluding query
            gpu_id:                     gpu device id to use for nearest neighbor (if possible for `metric` chosen)

        Returns:
            dists:                      numpy array of size [NUM_QUERY_IMAGES x k], distances of k nearest neighbors
                                        for each query
            retrieved_db_indices:       numpy array of size [NUM_QUERY_IMAGES x k], indices of k nearest neighbors
                                        for each query
    """
    import faiss

    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = gpu_id

    # Evaluate with inner product
    index = faiss.GpuIndexFlatIP(res, db_embeddings.shape[1], flat_config)
    index.add(db_embeddings)
    # retrieved k+1 results in case that query images are also in the db
    dists, retrieved_result_indices = index.search(query_embeddings, k + 1)

    return dists, retrieved_result_indices


def _retrieve_knn_faiss_gpu_euclidean(query_embeddings, db_embeddings, k, gpu_id=0):
    """
        Retrieve k nearest neighbor based on inner product

        Args:
            query_embeddings:           numpy array of size [NUM_QUERY_IMAGES x EMBED_SIZE]
            db_embeddings:              numpy array of size [NUM_DB_IMAGES x EMBED_SIZE]
            k:                          number of nn results to retrieve excluding query
            gpu_id:                     gpu device id to use for nearest neighbor (if possible for `metric` chosen)

        Returns:
            dists:                      numpy array of size [NUM_QUERY_IMAGES x k], distances of k nearest neighbors
                                        for each query
            retrieved_db_indices:       numpy array of size [NUM_QUERY_IMAGES x k], indices of k nearest neighbors
                                        for each query
    """
    import faiss

    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = gpu_id

    # Evaluate with inner product
    index = faiss.GpuIndexFlatL2(res, db_embeddings.shape[1], flat_config)
    index.add(db_embeddings)
    # retrieved k+1 results in case that query images are also in the db
    dists, retrieved_result_indices = index.search(query_embeddings, k + 1)

    return dists, retrieved_result_indices


def evaluate_recall_at_k(dists, results, query_labels, db_labels, k):
    """
        Evaluate Recall@k based on retrieval results

        Args:
            dists:          numpy array of size [NUM_QUERY_IMAGES x k], distances of k nearest neighbors for each query
            results:        numpy array of size [NUM_QUERY_IMAGES x k], indices of k nearest neighbors for each query
            query_labels:   list of labels for each query
            db_labels:      list of labels for each db
            k:              number of nn results to evaluate

        Returns:
            recall_at_k:    Recall@k in percentage
    """

    self_retrieval = False

    if query_labels is db_labels:
        self_retrieval = True

    expected_result_size = k + 1 if self_retrieval else k

    assert results.shape[1] >= expected_result_size, \
        "Not enough retrieved results to evaluate Recall@{}".format(k)

    recall_at_k = np.zeros((k,))

    for i in xrange(len(query_labels)):
        pos = 0 # keep track recall at pos
        j = 0 # looping through results
        while pos < k:
            if self_retrieval and i == results[i, j]:
                # Only skip the document when query and index sets are the exact same
                j += 1
                continue
            if query_labels[i] == db_labels[results[i, j]]:
                recall_at_k[pos:] += 1
                break
            j += 1
            pos += 1

    return recall_at_k/float(len(query_labels))*100.0


def evaluate_float_binary_embedding_faiss(query_embeddings, db_embeddings, query_labels, db_labels,
                                          output, k=1000, gpu_id=0):
    """
        Wrapper function to evaluate Recall@k for float and binary embeddings
        output recall@k strings for Cars, CUBS, Stanford Online Product, and InShop datasets
    """

    # ======================== float embedding evaluation ==========================================================
    # knn retrieval from embeddings (l2 normalized embedding + inner product = cosine similarity)
    dists, retrieved_result_indices = _retrieve_knn_faiss_gpu_inner_product(query_embeddings,
                                                                            db_embeddings,
                                                                            k,
                                                                            gpu_id=gpu_id)

    # evaluate recall@k
    r_at_k_f = evaluate_recall_at_k(dists, retrieved_result_indices, query_labels, db_labels, k)

    output_file = output + '_identity.eval'
    cars_cub_eval_str = "R@1, R@2, R@4, R@8: {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\".format(
        r_at_k_f[0], r_at_k_f[1], r_at_k_f[3], r_at_k_f[7])
    sop_eval_str = "R@1, R@10, R@100, R@1000: {:.2f} & {:.2f} & {:.2f} & {:.2f}  \\\\".format(
        r_at_k_f[0], r_at_k_f[9], r_at_k_f[99], r_at_k_f[999])
    in_shop_eval_str = "R@1, R@10, R@20, R@30, R@40, R@50: {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\".format(
        r_at_k_f[0], r_at_k_f[9], r_at_k_f[19], r_at_k_f[29], r_at_k_f[39], r_at_k_f[49])

    print(cars_cub_eval_str)
    print(sop_eval_str)
    print(in_shop_eval_str)
    with open(output_file, 'w') as of:
        of.write(cars_cub_eval_str + '\n')
        of.write(sop_eval_str + '\n')
        of.write(in_shop_eval_str + '\n')

    # ======================== binary embedding evaluation =========================================================
    binary_query_embeddings = np.require(query_embeddings > 0, dtype='float32')
    binary_db_embeddings = np.require(db_embeddings > 0, dtype='float32')

    # knn retrieval from embeddings (binary embeddings + euclidean = hamming distance)
    dists, retrieved_result_indices = _retrieve_knn_faiss_gpu_euclidean(binary_query_embeddings,
                                                                        binary_db_embeddings,
                                                                        k,
                                                                        gpu_id=gpu_id)
    # evaluate recall@k
    r_at_k_b = evaluate_recall_at_k(dists, retrieved_result_indices, query_labels, db_labels, k)

    output_file = output + '_binary.eval'

    cars_cub_eval_str = "R@1, R@2, R@4, R@8: {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\".format(
        r_at_k_b[0], r_at_k_b[1], r_at_k_b[3], r_at_k_b[7])
    sop_eval_str = "R@1, R@10, R@100, R@1000: {:.2f} & {:.2f} & {:.2f} & {:.2f}  \\\\".format(
        r_at_k_b[0], r_at_k_b[9], r_at_k_b[99], r_at_k_b[999])
    in_shop_eval_str = "R@1, R@10, R@20, R@30, R@40, R@50: {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\".format(
        r_at_k_b[0], r_at_k_b[9], r_at_k_b[19], r_at_k_b[29], r_at_k_b[39], r_at_k_b[49])

    print(cars_cub_eval_str)
    print(sop_eval_str)
    print(in_shop_eval_str)
    with open(output_file, 'w') as of:
        of.write(cars_cub_eval_str + '\n')
        of.write(sop_eval_str + '\n')
        of.write(in_shop_eval_str + '\n')

    return max(r_at_k_f[0], r_at_k_b[0])


