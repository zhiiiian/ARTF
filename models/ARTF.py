import logging
import time

import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from models.base import BaseLearner
from utils.artf_net import ARTFNet
from utils.toolkit import count_parameters, tensor2numpy
import pickle
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

EPSILON = 1e-8
T = 2


class ARTF(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._batch_size = args["batch_size"]
        self._num_workers = args["workers"]
        self._lr = args["lr"]
        self._epochs = args["epochs"]
        self._momentum = args["momentum"]
        self._weight_decay = args["weight_decay"]
        self._lr_steps = args["lr_steps"]
        self._modality = args["modality"]
        self.increment=args["increment"]
        self._freeze = args["freeze"]
        self._clip_gradient = args["clip_gradient"]

        self._network = ARTFNet(args["num_segments"], args["modality"], args["arch"],
                                convnet_type=args["convnet_type"], dropout=args["dropout"])
        self._lams = {'RGB': [], 'Accespec': [], 'Gyrospec': []}
        self.embedding_generator = EmbeddingGenerator(args["modality"], 768)
        self.discriminator = Discriminator(128, args["increment"])

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self.data_manager = data_manager

        self._network._gen_train_fc(data_manager.get_task_size(self._cur_task) * 2)
        self.para_count()
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        if self._cur_task > 0:
            for i in range(self._cur_task):
                for p in self._network.fusion_networks[i].parameters():
                    p.requires_grad = False

                for p in self._network.fc_list[i].parameters():
                    p.requires_grad = False

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
          
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers, drop_last=True
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)

    def train(self):
        self._network.train()
        self.embedding_generator.train()
        self.discriminator.train()
        if self._cur_task > 0:
            for i in range(self._cur_task):
                self._network.fusion_networks[i].eval()
                self._network.fc_list[i].eval()

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        self.embedding_generator.to(self._device)
        self.discriminator.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        param_groups = [
            {'params': filter(lambda p: p.requires_grad, self._network.parameters())},
            {'params': filter(lambda p: p.requires_grad, self.embedding_generator.parameters())},
            {'params': filter(lambda p: p.requires_grad, self.discriminator.parameters())},
        ]
        if self._cur_task == 0:
            optimizer = torch.optim.Adam(param_groups,
                                         self._lr,
                                         weight_decay=self._weight_decay)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, self._lr_steps, gamma=0.1)
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = torch.optim.Adam(param_groups,
                                         self._lr,
                                         weight_decay=self._weight_decay)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, self._lr_steps, gamma=0.1)
            t1 = time.time()
            self._update_representation(train_loader, test_loader, optimizer, scheduler)
            t2 = time.time()
            logging.info("Time taken for training: {}".format(t2 - t1))
        self._network.save_parameter()

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self._epochs))
        self.results_train = {'RGB': [],  'label': [], 'fusion_features': []}
        for _, epoch in enumerate(prog_bar):
            self.train()
            if self._freeze:
                self._network.feature_extract_network.freeze()

            losses_clf, losses_dis, losses = 0., 0., 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                for m in self._modality:
                    inputs[m] = inputs[m].to(self._device)
                targets = targets.to(self._device)

                features = self._network.feature_extract_network(inputs)
                fake_inputs, fake_targets,b = self._class_aug(features, targets)

                self.results_train['RGB'].append(fake_inputs['RGB'].detach().cpu().numpy())
                self.results_train['label'].append(fake_targets.detach().cpu().numpy())

                fusion_features = self._network.fusion_network(fake_inputs)["features"]

                fake_logits = self._network.fc(fusion_features)['logits']
                loss_clf = F.cross_entropy(fake_logits, fake_targets)
                self.results_train['fusion_features'].append(fusion_features.detach().cpu().numpy())
                dis_logits, dis_targets = self.discriminator(fusion_features, fake_targets, 16)
                loss_dis = F.cross_entropy(dis_logits, dis_targets)
                
                loss =loss_clf + 0.5 *  loss_dis
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_clf += loss_clf.item()
                losses_dis += loss_dis.item()

                _, preds = torch.max(fake_logits, dim=1)
                correct += preds.eq(fake_targets.expand_as(preds)).cpu().sum()
                total += len(fake_targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            self.training_iterations += 1
            if epoch % 5 == 0:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_dis {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self._epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_dis / len(train_loader),
                    train_acc,

                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_dis {:.3f},  Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self._epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_dis / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self._epochs))
        for _, epoch in enumerate(prog_bar):
            self.train()
            if self._freeze:
                self._network.feature_extract_network.freeze()
            losses_clf, losses_dis, losses = 0., 0., 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                for m in self._modality:
                    inputs[m] = inputs[m].to(self._device)
                targets = targets.to(self._device)
                targets = targets - self._known_classes
                features = self._network.feature_extract_network(inputs)
                fake_inputs, fake_targets,b = self._class_aug(features, targets)
                self.results_train['RGB'].append(fake_inputs['RGB'].detach().cpu().numpy())
                self.results_train['label'].append(fake_targets.detach().cpu().numpy())

                fusion_features = self._network.fusion_network(fake_inputs)["features"]
                fake_logits = self._network.fc(fusion_features)['logits']
                loss_clf = F.cross_entropy(fake_logits, fake_targets)

                dis_logits, dis_targets = self.discriminator(fusion_features, fake_targets, 16)
                loss_dis = F.cross_entropy(dis_logits, dis_targets)

                self.results_train['fusion_features'].append(fusion_features.detach().cpu().numpy())
                loss = loss_clf + 0.5 * loss_dis
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_clf += loss_clf.item()
                losses_dis += loss_dis.item()

                _, preds = torch.max(fake_logits, dim=1)
                correct += preds.eq(fake_targets.expand_as(preds)).cpu().sum()
                total += len(fake_targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            self.training_iterations += 1

            if epoch % 5 == 0:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_dis {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self._epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_dis / len(train_loader),
                    train_acc,

                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_dis {:.3f},Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self._epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_dis / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

    def _eval_cnn(self, loader):
        self._network.fusion_networks.to(self._device)
        self._network.fc_list.to(self._device)
        self._network.eval()
        y_pred, y_true = [], []
        results = []
        for _, (_, inputs, targets) in enumerate(loader):
            for m in self._modality:
                inputs[m] = inputs[m].to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs, mode='test')
                logits = outputs["logits"]
            predicts = torch.topk(
                logits, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

            results.append({'features': {m: outputs['features'][m].cpu().numpy() for m in self._modality},
                            'fusion_features': outputs['fusion_features'].cpu().numpy(),
                            'logits': logits.cpu().numpy()})

        return np.concatenate(y_pred), np.concatenate(y_true), results  # [N, topk]

    def eval_task(self, scores_dir):
        t1 = time.time()
        y_pred, y_true, results = self._eval_cnn(self.test_loader)
        t2 = time.time()
        logging.info("Time taken for evaluation: {}".format((t2 - t1)/1316))
        self.save_scores(results, y_true, y_pred, '{}/{}.pkl'.format(scores_dir, self._cur_task))
        cnn_accy = self._evaluate(y_pred, y_true)

        filename = '{}/task{}.pkl'.format(scores_dir, self._cur_task)

        with open(filename, 'wb') as f:
            pickle.dump(self.results_train, f)

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        return cnn_accy, nme_accy

    def _map_targets(self, select_targets):
        mixup_targets = select_targets + 4
        return mixup_targets

    def _class_aug(self, inputs, targets, alpha=0.2 ,mix_time=2):
        mixup_targets = []
        gen_inputs = {}
        mix_inputs = {}
        index_select = []
        index_perm = []

        for _ in range(mix_time):
            numbers = list(range(0, inputs[self._modality[0]].shape[0]))
            ori = torch.LongTensor(numbers)
            index = torch.randperm(inputs[self._modality[0]].shape[0])
            perm_targets = targets[index]
            mask = perm_targets != targets

            select_targets = targets[mask]
            perm_targets = perm_targets[mask]

            mixup_targets.append(self._map_targets(select_targets))
            if sum(mask) != 0:
                index_select.append(ori[mask])
                index_perm.append(index[mask])

        index_select = torch.cat(index_select, dim=0)
        index_perm = torch.cat(index_perm, dim=0)

        if len(mixup_targets) != 0:
            gen_targets = torch.cat(mixup_targets, dim=0)

        for m in self._modality:
            lams = np.random.beta(alpha, alpha, len(index_perm))
            lams = np.where(lams < 0.5, 0.75, lams)
            lams = torch.from_numpy(lams).to(self._device)[:, None].float()
            mixup_input = torch.cat(
                [torch.unsqueeze(lams[n] * (inputs[m][index_select[n]]) + (1 - lams[n]) * (inputs[m][index_perm[n]]), 0) for n in range(len(lams))], 0)
            #
            gen_input = self.embedding_generator(
                m, mixup_input
            )

            gen_inputs[m] = gen_input
            mix_inputs[m] = mixup_input
            inputs[m] = torch.cat([inputs[m], gen_inputs[m]], dim=0)
        targets = torch.cat([targets, gen_targets], dim=0)
        return inputs, targets,len(index_select)

    def para_count(self):
        gen_params = 0
        gen_params_train = 0
        for m in self._modality:
            gen_params += count_parameters(self.embedding_generator.model[m])
            gen_params_train += count_parameters(self.embedding_generator.model[m], True)
        dis_params = count_parameters(self.discriminator)
        dis_params_train = count_parameters(self.discriminator, True)

        logging.info("All params: {}".format(count_parameters(self._network) + gen_params + dis_params))
        logging.info(
            "Trainable params: {}".format(count_parameters(self._network, True) + gen_params_train + dis_params_train)
        )

class EmbeddingGenerator(nn.Module):
    def __init__(self, modality, embed_dim):
        super().__init__()
        self._modality = modality
        self.model = {}

        for m in self._modality:
            self.model[m] = nn.Sequential(nn.Linear(embed_dim, 1536),
                                          nn.LeakyReLU(),
                                          nn.Linear(1536, embed_dim),
                                          nn.ReLU()).cuda(0)
            
    def forward(self, m, h):
        outputs = self.model[m](h)
        return outputs + h

class Discriminator(nn.Module):
    def __init__(self, fusion_dim, increment):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(nn.Dropout(p=0.5),
                                   nn.Linear(128, increment))
    

    def forward(self, h, tar, b):
        outputs = self.layer(h)
        dis_targets = torch.cat([tar[:b], tar[b:] - 4])
        return outputs, dis_targets