import os
import sys

import time
from itertools import chain

from argparse import ArgumentParser

import torch
from pretrainedmodels.utils import ToRange255
from pretrainedmodels.utils import ToSpaceBGR
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

from data.inshop import InShop
from data.stanford_products import StanfordOnlineProducts
from data.cars196 import Cars196
from data.cub200 import Cub200
from metric_learning.util import SimpleLogger
from metric_learning.sampler import ClassBalancedBatchSampler

import metric_learning.modules.featurizer as featurizer
import metric_learning.modules.losses as losses

from extract_features import extract_feature
from evaluation.retrieval import evaluate_float_binary_embedding_faiss


def parse_args():
    """
    Helper function parsing the command line options
    """
    parser = ArgumentParser(description="PyTorch metric learning training script")
    # Optional arguments for the launch helper
    parser.add_argument("--dataset", type=str, default="StanfordOnlineProducts",
                        help="The dataset for training")
    parser.add_argument("--dataset_root", type=str, default="",
                        help="The root directory to the dataset")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--model_name", type=str, default="resnet50", help="The model name")
    parser.add_argument("--lr", type=float, default=0.01, help="The base lr")
    parser.add_argument("--gamma", type=float, default=0.1, help="Gamma applied to learning rate")
    parser.add_argument("--class_balancing", default=False, action='store_true', help="Use class balancing")
    parser.add_argument("--images_per_class", type=int, default=5, help="Images per class")
    parser.add_argument("--lr_mult", type=float, default=1, help="lr_mult for new params")
    parser.add_argument("--dim", type=int, default=2048, help="The dimension of the embedding")

    parser.add_argument("--test_every_n_epochs", type=int, default=2, help="Tests every N epochs")
    parser.add_argument("--epochs_per_step", type=int, default=4, help="Epochs for learning rate step")
    parser.add_argument("--pretrain_epochs", type=int, default=1, help="Epochs for pretraining")
    parser.add_argument("--num_steps", type=int, default=3, help="Num steps to take")
    parser.add_argument("--output", type=str, default="/data1/output", help="The output folder for training")

    return parser.parse_args()


def adjust_learning_rate(optimizer, epoch, epochs_per_step, gamma=0.1):
    """Sets the learning rate to the initial LR decayed by 10 every epochs"""
    # Skip gamma update on first epoch.
    if epoch != 0 and epoch % epochs_per_step == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= gamma
            print("learning rate adjusted: {}".format(param_group['lr']))


def main():
    args = parse_args()
    torch.cuda.set_device(0)
    gpu_device = torch.device('cuda')

    output_directory = os.path.join(args.output, args.dataset, str(args.dim),
                                    '_'.join([args.model_name, str(args.batch_size)]))
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    out_log = os.path.join(output_directory, "train.log")
    sys.stdout = SimpleLogger(out_log, sys.stdout)

    # Select model
    model_factory = getattr(featurizer, args.model_name)
    model = model_factory(args.dim)

    # Setup train and eval transformations
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(max(model.input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ToSpaceBGR(model.input_space == 'BGR'),
        ToRange255(max(model.input_range) == 255),
        transforms.Normalize(mean=model.mean, std=model.std)
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(max(model.input_size)),
        transforms.ToTensor(),
        ToSpaceBGR(model.input_space == 'BGR'),
        ToRange255(max(model.input_range) == 255),
        transforms.Normalize(mean=model.mean, std=model.std)
    ])

    # Setup dataset
    if args.dataset == 'StanfordOnlineProducts':
        train_dataset = StanfordOnlineProducts('/data1/data/stanford_products/Stanford_Online_Products',
                                               transform=train_transform)
        eval_dataset = StanfordOnlineProducts('/data1/data/stanford_products/Stanford_Online_Products',
                                              train=False,
                                              transform=eval_transform)
    elif args.dataset == 'Cars196':
        train_dataset = Cars196('/data1/data/cars196', transform=train_transform)
        eval_dataset = Cars196('/data1/data/cars196', train=False, transform=eval_transform)
    elif args.dataset == 'Cub200':
        train_dataset = Cub200('/data1/data/cub200/CUB_200_2011', transform=train_transform)
        eval_dataset = Cub200('/data1/data/cub200/CUB_200_2011', train=False, transform=eval_transform)
    elif args.dataset == "InShop":
        train_dataset = InShop('/data1/data/inshop', transform=train_transform)
        query_dataset = InShop('/data1/data/inshop', train=False, query=True, transform=eval_transform)
        index_dataset = InShop('/data1/data/inshop', train=False, query=False, transform=eval_transform)
    else:
        print("Dataset {} is not supported yet... Abort".format(args.dataset))
        return

    # Setup dataset loader
    if args.class_balancing:
        print("Class Balancing")
        sampler = ClassBalancedBatchSampler(train_dataset.instance_labels, args.batch_size, args.images_per_class)
        train_loader = DataLoader(train_dataset,
                                  batch_sampler=sampler, num_workers=4,
                                  pin_memory=True, drop_last=False, collate_fn=default_collate)
    else:
        print("No class balancing")
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  drop_last=False,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=4)

    if args.dataset != "InShop":
        eval_loader = DataLoader(eval_dataset,
                                 batch_size=args.batch_size,
                                 drop_last=False,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=4)
    else:
        query_loader = DataLoader(query_dataset,
                                  batch_size=args.batch_size,
                                  drop_last=False,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=4)
        index_loader = DataLoader(index_dataset,
                                  batch_size=args.batch_size,
                                  drop_last=False,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=4)

    # Setup loss function
    loss_fn = losses.NormSoftmaxLoss(args.dim, train_dataset.num_instance)

    model.to(device=gpu_device)
    loss_fn.to(device=gpu_device)

    # Training mode
    model.train()

    # Start with pretraining where we finetune only new parameters to warm up
    opt = torch.optim.SGD(list(loss_fn.parameters()) + list(set(model.parameters()) -
                                                            set(model.feature.parameters())),
                          lr=args.lr * args.lr_mult, momentum=0.9, weight_decay=1e-4)

    log_every_n_step = 10
    for epoch in range(args.pretrain_epochs):
        for i, (im, _, instance_label, index) in enumerate(train_loader):
            data = time.time()
            opt.zero_grad()

            im = im.to(device=gpu_device, non_blocking=True)
            instance_label = instance_label.to(device=gpu_device, non_blocking=True)

            forward = time.time()
            embedding = model(im)
            loss = loss_fn(embedding, instance_label)

            back = time.time()
            loss.backward()
            opt.step()

            end = time.time()

            if (i + 1) % log_every_n_step == 0:
                print('Epoch {}, LR {}, Iteration {} / {}:\t{}'.format(
                    args.pretrain_epochs - epoch, opt.param_groups[0]['lr'], i, len(train_loader), loss.item()))

                print('Data: {}\tForward: {}\tBackward: {}\tBatch: {}'.format(
                    forward - data, back - forward, end - back, end - forward))

        eval_file = os.path.join(output_directory, 'epoch_{}'.format(args.pretrain_epochs - epoch))
        if args.dataset != "InShop":
            embeddings, labels = extract_feature(model, eval_loader, gpu_device)
            evaluate_float_binary_embedding_faiss(embeddings, embeddings, labels, labels, eval_file, k=1000, gpu_id=0)
        else:
            query_embeddings, query_labels = extract_feature(model, query_loader, gpu_device)
            index_embeddings, index_labels = extract_feature(model, index_loader, gpu_device)
            evaluate_float_binary_embedding_faiss(query_embeddings, index_embeddings, query_labels, index_labels, eval_file,
                                                  k=1000, gpu_id=0)

    # Full end-to-end finetune of all parameters
    opt = torch.optim.SGD(chain(model.parameters(), loss_fn.parameters()), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    for epoch in range(args.epochs_per_step * args.num_steps):
        print('Output Directory: {}'.format(output_directory))
        adjust_learning_rate(opt, epoch, args.epochs_per_step, gamma=args.gamma)

        for i, (im, _, instance_label, index) in enumerate(train_loader):
            data = time.time()

            opt.zero_grad()

            im = im.to(device=gpu_device, non_blocking=True)
            instance_label = instance_label.to(device=gpu_device, non_blocking=True)

            forward = time.time()
            embedding = model(im)
            loss = loss_fn(embedding, instance_label)

            back = time.time()
            loss.backward()
            opt.step()

            end = time.time()

            if (i + 1) % log_every_n_step == 0:
                print('Epoch {}, LR {}, Iteration {} / {}:\t{}'.format(
                    epoch, opt.param_groups[0]['lr'], i, len(train_loader), loss.item()))
                print('Data: {}\tForward: {}\tBackward: {}\tBatch: {}'.format(
                    forward - data, back - forward, end - back, end - data))

        snapshot_path = os.path.join(output_directory, 'epoch_{}.pth'.format(epoch + 1))
        torch.save(model.state_dict(), snapshot_path)

        if (epoch + 1) % args.test_every_n_epochs == 0:
            eval_file = os.path.join(output_directory, 'epoch_{}'.format(epoch + 1))
            if args.dataset != "InShop":
                embeddings, labels = extract_feature(model, eval_loader, gpu_device)
                evaluate_float_binary_embedding_faiss(embeddings, embeddings, labels, labels, eval_file, k=1000, gpu_id=0)
            else:
                query_embeddings, query_labels = extract_feature(model, query_loader, gpu_device)
                index_embeddings, index_labels = extract_feature(model, index_loader, gpu_device)
                evaluate_float_binary_embedding_faiss(query_embeddings, index_embeddings, query_labels, index_labels,
                                                      eval_file,
                                                      k=1000, gpu_id=0)

if __name__ == '__main__':
    main()
