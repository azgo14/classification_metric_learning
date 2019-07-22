import numpy as np

import torch


def extract_feature(model, loader, gpu_device):
    """
    Extract embeddings from given `model` for given `loader` dataset on `gpu_device`.
    """
    model.eval()

    all_embeddings = []
    all_labels = []
    log_every_n_step = 10

    with torch.no_grad():
        for i, (im, class_label, instance_label, index) in enumerate(loader):
            im = im.to(device=gpu_device)
            embedding = model(im)

            all_embeddings.append(embedding.cpu().numpy())
            all_labels.extend(instance_label.tolist())

            if (i + 1) % log_every_n_step == 0:
                print('Process Iteration {} / {}:'.format(i, len(loader)))

    all_embeddings = np.vstack(all_embeddings)

    print("Generated {} embedding matrix".format(all_embeddings.shape))
    print("Generate {} labels".format(len(all_labels)))

    model.train()
    return all_embeddings, all_labels
