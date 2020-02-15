import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from efficientnet_pytorch import EfficientNet

import apex
from apex import amp

from tqdm import tqdm
import pandas as pd

class ClassificationTrainer():
    def __init__(self, train, valid, args):
        self.device = torch.device("cuda:0")
        self.classes = args.classes
        self.model, self.optimizer = self.__init_model(args.classes, args.lr)
        self.criterion = nn.BCEWithLogits() if args.multilabel else nn.CrossEntropyLoss()
        self.best_pred = 0.0

        self.scheduler = ReduceLROnPlateau(self.optimizer,
                                           factor=0.7, patience=1,
                                           verbose=True, threshold=0.03,
                                           min_lr=1e-5)

        self.cur_epoch = 0
        self.num_epoch = args.epochs

        self.train = train
        self.valid = valid

        self.mdl_path = args.model_path
        self.csv_path = args.csv_path

        self.train_loss = []
        self.train_accu = []
        self.valid_loss = []
        self.valid_accu = []

        self.confusion = np.zeros((total_epoch, 2, classes, classes))

        if save is not None and savetype == 'full':
            self.model.load_state_dict(save)
        elif save is not None and savetype == 'transfer':
            self.load_transfer_dict(save)

    def load_transfer_dict(self, pretrained_dict, fc_bias, fc_weight):
        own_state = self.model.state_dict()
        for name, param in pretrained_dict.items():
            if name not in own_state or \
            name == '_fc.bias' or \
            name == '_fc.weight':
                 continue
            own_state[name].copy_(param)

    def __init_model(self, classes, lr, wd):
        optimizer = apex.optimizers.FusedSGD(model.parameters(),
                                             lr=lr, momentum=0.9,
                                             weight_decay=wd, nesterov=True)

        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
        return model, optimizer

    def train(self):
        print('[Epoch: %d/%d]' % (self.epoch, self.total_epoch))

        total_loss = 0.0
        total_acc = 0.0
        self.model.train()

        tbar = tqdm(self.train_loader)
        for i, (inputs, targets) in enumerate(tbar):
            inputs = inputs.to(self.device, dtype=torch.float)
            targets = targets.to(self.device, dtype=torch.long)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            acc = self.acc(outputs, targets)

            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.cpu().data.numpy()
            total_acc += acc

            self.confusion[self.epoch, 0, preds, :] +=
            
            if i == 0 and self.epoch == 0:
                self.train_loss += [total_loss]
                self.train_acc += [total_acc]

            if i % 8 == 0 and bar:
                cur_loss = total_loss / (i + 1)
                cur_acc  = total_acc / (i + 1)
                tbar.set_description('Train [loss: %.3f - acc: %.3f]' \
                                      % (cur_loss, cur_acc))

        self.train_loss += [total_loss / len(tbar)]
        self.train_acc += [total_acc / len(tbar)]

    def valid(self, bar=True):
        total_loss = 0.0
        total_acc = 0.0
        self.model.eval()

        if bar: tbar = tqdm(self.val_loader)
        else: tbar = self.val_loader

        for i, (inputs, targets) in enumerate(tbar):
            inputs = inputs.to(self.device, dtype=torch.float)
            targets = targets.to(self.device, dtype=torch.long)

            with torch.no_grad():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                targets = targets.cpu()
                preds = outputs.argmax(axis=1).cpu().data
                acc = self.acc(preds, targets)
            
                total_loss += loss.cpu().data.numpy()
                total_acc += acc.data.numpy()

            if i % 8 == 0 and bar:
                cur_loss = total_loss / (i + 1)
                cur_acc  = total_acc / (i + 1)
                tbar.set_description('Train [loss: %.3f - acc: %.3f]' \
                                      % (cur_loss, cur_acc))

        cur_ac = total_acc / len(tbar)
        self.scheduler.step(cur_ac)
        self.valid_loss += [total_loss / len(tbar)]
        self.valid_acc += [cur_ac]

        if cur_ac > self.best_pred:

    def run_train(self):
        for _ in range(self.total_epoch):
            self.train()
            self.valid()
            self.generate_csv()

    def generate_csv(self):
        history_df = pd.DataFrame({'epoch': list(range(self.epoch + 1)),
                                   'train_loss': self.train_loss,
                                   'train_acc': self.train_acc,
                                   'valid_loss': self.valid_loss,
                                   'valid_acc': self.valid_acc})

        history_df.to_csv(self.history_loc, index=False)


