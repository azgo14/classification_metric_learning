import faiss
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score

from argparse import ArgumentParser


def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description="PyTorch metric learning nmi script")
    # Optional arguments for the launch helper
    parser.add_argument("--num_workers", type=int, default=4,
                        help="The number of workers for eval")
    parser.add_argument("--snap", type=str,
                        help="The snapshot to compute nmi")
    parser.add_argument("--output", type=str, default="/data1/output/",
                        help="The output file")
    parser.add_argument("--dataset", type=str, default="StanfordOnlineProducts",
                        help="The dataset for training")
    parser.add_argument('--binarize', action='store_true')

    return parser.parse_args()


def test_nmi(embeddings, labels, output_file):
    unique_labels = np.unique(labels)
    kmeans = KMeans(n_clusters=unique_labels.size, random_state=0, n_jobs=-1).fit(embeddings)

    nmi = normalized_mutual_info_score(kmeans.labels_, labels)

    print("NMI: {}".format(nmi))
    return nmi


def test_nmi_faiss(embeddings, labels):
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0

    unique_labels = np.unique(labels)
    d = embeddings.shape[1]
    kmeans = faiss.Clustering(d, unique_labels.size)
    kmeans.verbose = True
    kmeans.niter = 300
    kmeans.nredo = 10
    kmeans.seed = 0

    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    kmeans.train(embeddings, index)

    dists, pred_labels = index.search(embeddings, 1)

    pred_labels = pred_labels.squeeze()

    nmi = normalized_mutual_info_score(labels, pred_labels)

    print("NMI: {}".format(nmi))
    return nmi


if __name__ == '__main__':
    args = parse_args()
    embedding_file = args.snap.replace('.pth', '_embed.npy')
    all_embeddings = np.load(embedding_file)
    lable_file = args.snap.replace('.pth', '_label.npy')
    all_labels = np.load(lable_file)
    nmi = test_nmi_faiss(all_embeddings, all_labels)
