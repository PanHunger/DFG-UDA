from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argsParameters import args

import os
import sys
import math
import time
import warnings
from datetime import datetime
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
import torch.nn.parallel
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler

from data_list import ImageList
import pre_process as prep
from Networks import discriminator, fine_net
import adversarial1 as ad

from Functions import test_target, test_source
from saveTxt import Logger
from saveTensorBoard import createWriter
from saveIO import save_checkpoint

warnings.filterwarnings("ignore")


def entropy(p):
    p = F.softmax(p, dim=1)
    return -torch.mean(torch.sum(p * torch.log(p + 1e-5), 1))


def entropy_margin(p, value, margin=0.2, weight=None):
    p = F.softmax(p, dim=1)
    return -torch.mean(
        hinge(torch.abs(-torch.sum(p * torch.log(p + 1e-5), 1) - value), margin)
    )


def hinge(input, margin=0.2):
    return torch.clamp(input, min=margin)


def entropy_loss_func(input_):
    # if the element in input larger than 0.000001, return True, otherwise False.
    mask = input_.ge(0.000001)
    mask_out = torch.masked_select(input_, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(input_.size(0))


def inv_lr_scheduler(param_lrs, optimizer, iter_num, gamma, power, init_lr=0.001):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (1 + gamma * iter_num) ** (-power)
    i = 0
    for pg in optimizer.param_groups:
        pg["lr"] = lr * param_lrs[i]
        i += 1
    return optimizer


def strategy_progressive(
        iter_num, initial_smooth, final_smooth, max_iter_num, strategy
):
    if strategy == "e":
        lambda_p = 2 / (1 + math.exp(-10 * iter_num / max_iter_num)) - 1
    elif strategy == "l":
        lambda_p = iter_num / max_iter_num
    elif strategy == "s":
        lambda_p = iter_num // (max_iter_num // 10) * 0.1
    elif strategy == "x":
        lambda_p = math.pow(2, (iter_num / max_iter_num)) - 1
    else:
        lambda_p = 2 / (1 + math.exp(-10 * iter_num / max_iter_num)) - 1
    smooth = initial_smooth + (final_smooth - initial_smooth) * lambda_p
    return smooth


if __name__ == "__main__":
    record_name = (
        "PAL_only-Teacher(R50SnapDomain)Student-batch_{}-size_{}_{}-fine{}-{}2{}-{}".format(
            args.batch_size,
            args.resize_size,
            args.crop_size,
            args.fineNet,
            args.source,
            args.target,
            args.tips,
        )
        if args.tips != "a"
        else "a"
    )

    sys.stdout = Logger(record_name, stream=sys.stdout)
    balanced = False

    TIMESTAMP = "{0:%Y-%m-%d %H:%M:%S}".format(datetime.now())
    print("Start time: ", TIMESTAMP)
    """# device assignment"""
    gpu_ids = args.gpu_ids.split(",")
    gpus = [i for i in range(len(gpu_ids))]
    # gpus = [int(g) for g in gpu_ids]

    """# file paths and domains"""
    file_path = {
        "p": "./dataset_list/cub200_drawing.txt",
        "c": "./dataset_list/cub200_2011.txt",
        "bn": "./dataset_list/bird31_nabirds_list.txt",
        "bi": "./dataset_list/bird31_ina_list_2017.txt",
        "bc": "./dataset_list/bird31_cub2011.txt",
        "vt": "./dataset_list/visda2017_train_image_list.txt",
        "vv": "./dataset_list/visda2017_val_image_list.txt",
    }

    dataset_source = file_path[args.source]
    dataset_target = dataset_test = file_path[args.target]

    """# dataset load"""
    batch_size = {
        "train": args.batch_size,
        "val": args.batch_size,
        "test": args.batch_size,
    }
    val_time = 3
    # val_time = 10
    for i in range(val_time):
        batch_size["val" + str(i)] = args.batch_size

    root_path = "../DATASET/FINE_GRAINED"
    dataset_loaders = {}

    dataset_list = ImageList(
        open(dataset_source).readlines(),
        root_path,
        transform=prep.image_train(
            resize_size=args.resize_size, crop_size=args.crop_size
        ),
    )
    source_num = len(dataset_list)
    if balanced:
        label_list = []
        for img in dataset_list.imgs:
            label_list.append(img[1][0])

        freq = Counter(label_list)
        class_weight = {x: 1.0 / freq[x] for x in freq}
        source_weights = [class_weight[x] for x in label_list]
        sampler = WeightedRandomSampler(source_weights, len(label_list))
        print("use balanced loader")
        dataset_loaders["train"] = torch.utils.data.DataLoader(
            dataset_list,
            batch_size=batch_size["train"],
            sampler=sampler,
            drop_last=True,
            num_workers=args.num_worker,
        )
    else:
        dataset_loaders["train"] = torch.utils.data.DataLoader(
            dataset_list,
            batch_size=batch_size["train"],
            shuffle=True,
            num_workers=args.num_worker,
        )

    dataset_list = ImageList(
        open(dataset_target).readlines(),
        root_path,
        transform=prep.image_train(
            resize_size=args.resize_size, crop_size=args.crop_size
        ),
        return_idx=True,
    )
    target_num = len(dataset_list)
    dataset_loaders["val"] = torch.utils.data.DataLoader(
        dataset_list,
        batch_size=batch_size["val"],
        shuffle=True,
        num_workers=args.num_worker,
    )

    dataset_list = ImageList(
        open(dataset_test).readlines(),
        root_path,
        transform=prep.image_test(
            resize_size=args.resize_size, crop_size=args.crop_size
        ),
    )
    dataset_loaders["test"] = torch.utils.data.DataLoader(
        dataset_list,
        batch_size=batch_size["test"],
        shuffle=False,
        num_workers=args.num_worker,
    )

    prep_dict_test = prep.image_test_10crop(
        resize_size=args.resize_size, crop_size=args.crop_size
    )
    for i in range(val_time):
        dataset_list = ImageList(
            open(dataset_test).readlines(),
            root_path,
            transform=prep_dict_test["val" + str(i)],
        )
        dataset_loaders["val" + str(i)] = torch.utils.data.DataLoader(
            dataset_list,
            batch_size=batch_size["val" + str(i)],
            shuffle=False,
            num_workers=args.num_worker,
        )

    """# fine-grained categories and coarse-grained categories"""
    # cate_all = [12, 122, 38, 14, 1, 1, 1]
    cate_all = [200, 122, 38, 14, 1, 1, 1]

    """# (Student) fine-grained feature extractor with bottleneck layer + fine-grained label predictor"""
    my_fine_net = fine_net(
        None, cate_all, name=args.fineNet, pool_type=args.pool_type
    )
    my_fine_net = torch.nn.DataParallel(
        my_fine_net, device_ids=gpus).cuda()
    my_fine_net.train(True)

    """# (Teacher) fine-grained feature extractor with bottleneck layer"""
    # my_fine_tea_net = fine_net(
    #     None, cate_all, name="ConvNeXtBase", pool_type=args.pool_type
    # )
    # my_fine_tea_dict = torch.load(
    #     "pretrainedWeights/PAL_only-batch_32-size_256_224-fineConvNeXtBase-c2p-/model_best.pth.tar")["fine_state_dict"]
    my_fine_tea_net = fine_net(
        None, cate_all, name=args.fineNet, pool_type='MeanLN', is_tea=True,
    )

    # my_fine_tea_dict = torch.load(
    #     "pretrainedWeights/resnet50_c_snapmix_model_best.pth.tar")["state_dict"]
    # trained_dict = {k[7:]: v for k, v in my_fine_tea_dict.items()}
    # my_fine_tea_dict.update(trained_dict)
    # my_fine_tea_net.model_fc.load_state_dict(my_fine_tea_dict)

    my_fine_tea_net = torch.nn.DataParallel(
        my_fine_tea_net, device_ids=gpus).cuda()
    my_fine_tea_net.eval()

    """# domain discriminator"""
    my_discriminator = discriminator(256, cate_all)
    my_discriminator = torch.nn.DataParallel(
        my_discriminator, device_ids=gpus).cuda()
    my_discriminator.train(True)

    """# gradient reversal layer"""
    iter_num_adl = 0
    my_grl = ad.AdversarialLayer.apply

    """# criterion"""
    criterion = {
        "classifier": nn.CrossEntropyLoss(),
        "kl_loss": nn.KLDivLoss(reduction="sum"),
        "adversarial": nn.BCELoss(),
    }

    """# optimizer"""
    optimizer_dict = [
        {
            "params": filter(
                lambda p: p.requires_grad, my_fine_net.module.model_fc.parameters()
            ),
            "lr": 0.1,
        },
        {
            "params": filter(
                lambda p: p.requires_grad, my_fine_net.module.bottleneck_layer.parameters(),
            ),
            "lr": 1,
        },
        {
            "params": filter(
                lambda p: p.requires_grad, my_fine_net.module.classifier_layer.parameters(),
            ),
            "lr": 1,
        },
        {
            "params": filter(
                lambda p: p.requires_grad, my_discriminator.parameters()
            ),
            "lr": 1,
        },
    ]

    optimizer = optim.SGD(optimizer_dict, lr=0.1,
                          momentum=0.9, weight_decay=0.0005)

    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
        
    """# Recoder"""
    # losses
    train_classifier_loss = 0.0
    train_distillation_loss = 0.0
    train_fine_cross_loss = 0.0
    train_transfer_loss = 0.0
    train_entropy_loss_source = 0.0
    train_entropy_loss_target = 0.0
    train_total_loss = 0.0
    # accuracy
    best_acc = 0.0
    # len dataset
    len_source = len(dataset_loaders["train"]) - 1
    len_target = len(dataset_loaders["val"]) - 1
    iter_source = iter(dataset_loaders["train"])
    iter_target = iter(dataset_loaders["val"])
    # TensorBoard
    writer = createWriter(record_name)

    mom_pre = 0.1
    do_pre = 0.1
    count = 0
    split = 1

    # for i, d in enumerate(dataset_loaders["train"]):
    #     print(i)

    """ ## Training Codes """
    for iter_num in tqdm(range(1, args.max_iteration + 1)):
        my_fine_net.train(True)
        optimizer = inv_lr_scheduler(
            param_lr, optimizer, iter_num, init_lr=args.lr, gamma=0.001, power=0.75
        )
        writer.add_scalar(
            "Learning Parameters/lr-FineNet_Backbone",
            optimizer.param_groups[0]["lr"],
            iter_num,
        )
        writer.add_scalar(
            "Learning Parameters/lr-FineNet_BottleNeck",
            optimizer.param_groups[1]["lr"],
            iter_num,
        )
        writer.add_scalar(
            "Learning Parameters/lr-FineNet_Classifier",
            optimizer.param_groups[2]["lr"],
            iter_num,
        )
        writer.add_scalar(
            "Learning Parameters/lr-Discriminator",
            optimizer.param_groups[3]["lr"],
            iter_num,
        )
        optimizer.zero_grad()

        # Load source and target data
        if iter_num % len_source == 0:
            iter_source = iter(dataset_loaders["train"])
        if iter_num % len_target == 0:
            iter_target = iter(dataset_loaders["val"])
        data_source = iter_source.next()
        data_target = iter_target.next()
        inputs_source, labels_source = data_source
        inputs_target, labels_target, index_t = data_target  # do not use labels_target
        # similate input and label
        # inputs_source = torch.randn(36, 3, 224, 224)
        # inputs_target = torch.randn(36, 3, 224, 224)
        # labels_source = torch.tensor([0 for i in range(36)])
        # labels_target = torch.tensor([1 for i in range(36)])

        assert inputs_source.shape == inputs_target.shape
        cwh = inputs_source.shape[1:]

        # ''' simple cross
        if args.cross_method == "simple":
            source_list = [i for i in range(0, batch_size["train"] * 2, 2)]
            target_list = [i for i in range(1, batch_size["train"] * 2, 2)]
        # 2. random cross
        elif args.cross_method == "random":
            batch_list = np.arange(batch_size["train"] * 2)
            np.random.shuffle(batch_list)
            source_list = batch_list[:batch_size["train"]].tolist()
            target_list = batch_list[batch_size["train"]:].tolist()
        else:
            source_list = [i for i in range(0, batch_size["train"], 1)]
            target_list = [i for i in range(batch_size["train"], batch_size["train"] * 2, 1)]

        inputs = torch.zeros(batch_size["train"] * 2, cwh[0], cwh[1], cwh[2])
        inputs[source_list] = inputs_source
        inputs[target_list] = inputs_target

        # inputs = torch.cat((inputs_source, inputs_target), dim=0)
        inputs = inputs.cuda()
        # index_t = index_t.cuda()
        """# DUA: Dynamic Unsupervised Adaptation"""
        if args.momentum_strategy > 0:
            lambda_momentum = strategy_progressive(
                count,
                args.initial_momentum,
                args.final_momentum,
                args.max_iteration // args.period_times,
                args.smooth_stratege,
            )
            lambda_dropout = strategy_progressive(
                count,
                args.initial_dropout,
                args.final_dropout,
                args.max_iteration // args.period_times,
                args.smooth_stratege,
            )
            if args.momentum_strategy == 0:
                mom_new = mom_pre * 0.94
                do_new = do_pre * 0.94
            elif args.momentum_strategy == 1:
                mom_new = lambda_momentum
                do_new = lambda_dropout
            elif args.momentum_strategy == 2:
                if iter_num % (args.max_iteration // args.period_times) == 0:
                    count = 0
                mom_new = lambda_momentum
                do_new = lambda_dropout
            else:
                raise Exception("Wrong momentum strategy!")
            count += 1
            writer.add_scalar("Learning Parameters/Momentum",
                              mom_new, iter_num)
            writer.add_scalar("Learning Parameters/dropout", do_new, iter_num)

            for m in my_fine_net.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                    m.train()
                    # m.momentum = mom_new + 0.005
                    m.momentum = mom_new
            for m in my_discriminator.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                    m.train()
                    # m.momentum = mom_new + 0.005
                    m.momentum = mom_new
            """ Progressive Dropout
            for m in my_fine_net.modules():
                if isinstance(m, nn.Dropout):
                    m.train()
                    m.p = do_new
            for m in my_discriminator.modules():
                if isinstance(m, nn.Dropout):
                    m.train()
                    m.p = do_new
            # """
            # 0
            # mom_pre = mom_new
        else:
            writer.add_scalar("Learning Parameters/Momentum", 0.1, iter_num)
            writer.add_scalar("Learning Parameters/dropout", 0.5, iter_num)
        """# Fine Path"""
        fine_labels_source_cpu = labels_source.view(-1, 1)
        labels_source = labels_source.cuda()
        # Smooth fine-coarse weight
        lambda_progressive = strategy_progressive(
            iter_num,
            args.initial_smooth,
            args.final_smooth,
            args.max_iteration,
            args.smooth_stratege,
        )
        # (Teacher) Get bottleneck feature
        with torch.no_grad():
            feature_btnk_tea, _ = my_fine_tea_net(inputs)
            feature_btnk_tea = feature_btnk_tea.detach()
        # (Student) Get bottleneck feature and fine-grained logits of source and target domain
        features_btnk, logits_fine = my_fine_net(inputs)
        teacher_loss = F.mse_loss(features_btnk, feature_btnk_tea)
        # get source fine-grained logits

        # logits_fine_source = logits_fine.narrow(0, 0, batch_size["train"])
        logits_fine_source = logits_fine[source_list]

        fine_labels_onehot = torch.zeros(logits_fine_source.size()).scatter_(
            1, fine_labels_source_cpu, 1
        )
        fine_labels_onehot = fine_labels_onehot.cuda()
        labels_onehot_smooth = lambda_progressive * fine_labels_onehot
        fine_classifier_loss = criterion["kl_loss"](
            nn.LogSoftmax(dim=1)(logits_fine_source), labels_onehot_smooth
        )
        fine_classifier_loss = fine_classifier_loss / batch_size["train"]
        classifier_loss = fine_classifier_loss  # PAL only
        """# Domain Path"""
        domain_labels = torch.zeros(batch_size['train'] * 2, 1).float()
        domain_labels[source_list] = 1.0
        domain_labels = domain_labels.cuda()
        iter_num_adl += 1
        domain_predicted = my_discriminator(
            my_grl(features_btnk, iter_num_adl), nn.Softmax(
                dim=1)(logits_fine).detach()
        )
        transfer_loss = nn.BCELoss()(domain_predicted, domain_labels)
        """# Entropy Path"""
        entropy_loss_source = entropy_loss_func(
            nn.Softmax(dim=1)(
                logits_fine[source_list]
            )
        )

        entropy_loss_target = entropy_loss_func(
            nn.Softmax(dim=1)(
                logits_fine[target_list]
            )
        )

        total_loss = classifier_loss + \
                     teacher_loss * args.teacher_weight + \
                     entropy_loss_source * args.entropy_source + \
                     entropy_loss_target * args.entropy_target + \
                     transfer_loss
        # total_loss = classifier_loss + \
        #              entropy_loss_source * args.entropy_source + \
        #              entropy_loss_target * args.entropy_target + \
        #              transfer_loss

        total_loss.backward()
        optimizer.step()

        # Tensorboard recoder
        writer.add_scalar(
            "Progressive Loss Weight/Fine Progressive Weight", lambda_progressive, iter_num,
        )
        writer.add_scalar(
            "Loss/Distillation Loss", teacher_loss.item(), iter_num
        )
        writer.add_scalar(
            "Loss/Classifier Added Loss", classifier_loss.item(), iter_num
        )
        writer.add_scalar(
            "Loss/Classifier Fine Loss", fine_classifier_loss.item(), iter_num
        )
        writer.add_scalar(
            "Loss/Entropy Source Loss", entropy_loss_source.item(), iter_num
        )
        writer.add_scalar(
            "Loss/Entropy Target Loss", entropy_loss_target.item(), iter_num
        )
        writer.add_scalar(
            "Loss/GAN Loss", transfer_loss.item(), iter_num
        )
        writer.add_scalar(
            "Loss/Total Loss", total_loss.item(), iter_num
        )

        train_classifier_loss += classifier_loss.item()
        train_distillation_loss += teacher_loss.item()
        train_fine_cross_loss += fine_classifier_loss.item()
        train_transfer_loss += transfer_loss.item()
        train_entropy_loss_source += entropy_loss_source.item()
        train_entropy_loss_target += entropy_loss_target.item()
        train_total_loss += total_loss.item()

        # test
        # interval = 1
        # interval = 500
        interval = int(args.max_iteration / 25)
        if iter_num % interval == 0:
            my_fine_net.eval()
            # test_source_acc = test_source(dataset_loaders, my_fine_net)
            test_acc = test_target(
                dataset_loaders, my_fine_net, multi=False, val_time=val_time
            )
            # print('test source acc:%.4f' % test_source_acc)
            print("test_acc:%.4f" % test_acc)

            # Record best score
            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)
            if is_best:
                save_checkpoint(
                    {
                        "iter_num": iter_num,
                        "fine_state_dict": my_fine_net.module.state_dict(),
                        "discriminator_state_dict": my_discriminator.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_score": best_acc,
                    },
                    is_best=is_best,
                    outdir=os.path.join("ExperimentRecords", record_name),
                )

            writer.add_scalar(
                "Accuracy/Target Accuracy", test_acc, iter_num
            )
            writer.add_scalar(
                "Accuracy/Target Best Accuracy", best_acc, iter_num
            )

            print(
                "Iter {:05d}, "
                "Avg Fine Cross Entropy Loss: {:.4f}; "
                "Avg Distillation Loss: {:.4f}; "
                "Avg Transfer Loss: {:.4f}; "
                "Avg Entropy Loss Source: {:.4f}; "
                "Avg Entropy Loss Target: {:.4f}; "
                "Avg Training Loss: {:.4f}; "
                "Best Accuracy: {:.4f}".format(
                    iter_num,
                    train_fine_cross_loss / float(interval),
                    train_distillation_loss / float(interval),
                    train_transfer_loss / float(interval),
                    train_entropy_loss_source / float(interval),
                    train_entropy_loss_target / float(interval),
                    train_total_loss / float(interval),
                    best_acc,
                )
            )
            it_t = 500
            writer.add_scalar(
                "Avg Loss {} interval/Classifier Fine Loss".format(it_t),
                train_fine_cross_loss / float(it_t),
                iter_num,
            )
            writer.add_scalar(
                "Avg Loss {} interval/Classifier Added Loss".format(it_t),
                train_classifier_loss / float(it_t),
                iter_num,
            )
            writer.add_scalar(
                "Avg Loss {} interval/Classifier Added Loss".format(it_t),
                train_distillation_loss / float(it_t),
                iter_num,
            )
            writer.add_scalar(
                "Avg Loss {} interval/Entropy Source Loss".format(it_t),
                train_entropy_loss_source / float(it_t),
                iter_num,
            )
            writer.add_scalar(
                "Avg Loss {} interval/Entropy Target Loss".format(it_t),
                train_entropy_loss_target / float(it_t),
                iter_num,
            )
            writer.add_scalar(
                "Avg Loss {} interval/GAN Loss".format(it_t),
                train_transfer_loss / float(it_t),
                iter_num,
            )
            writer.add_scalar(
                "Avg Loss {} interval/Total Loss".format(it_t),
                train_total_loss / float(it_t),
                iter_num,
            )

            train_classifier_loss = 0.0
            train_distillation_loss = 0.0
            train_fine_cross_loss = 0.0
            train_transfer_loss = 0.0
            train_entropy_loss_source = 0.0
            train_entropy_loss_target = 0.0
            train_total_loss = 0.0
            # time.sleep(240)

    TIMESTAMP = "{0:%Y-%m-%d %H:%M:%S}".format(datetime.now())
    print("End time: ", TIMESTAMP)
