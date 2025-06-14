import torch
import numpy as np
import scipy.io
from torch.utils.data import DataLoader, TensorDataset
import time
import datetime
import random
import torch.nn as nn
from torch.autograd import grad
from torch import Tensor
import os
from scipy.signal import butter, lfilter
from utils.util import init_seed, compute_gradient_penalty, butter_bandpass_filter
from models.generator import FeatureExtractor
from models.generator import Classifier
from models.discriminator import DomainDiscriminator



class WGAN_MI_Classifier():
    def __init__(self, target_idx, data_root='./Datasets/'):
        self.batch_size = 32
        self.n_epochs = 200
        self.num_of_classes = 4 # in Dataset 2a
        self.lr = 0.0005
        self.b1 = 0.5
        self.b2 = 0.999
        self.feature_dim = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_idx = None
        self.data_root = data_root
        self.in_channels = 22

        # Initialization Models
        self.F = FeatureExtractor(n_channels=self.in_channels, n_bands=6, depth_mul=2, window_size=1).to(self.device)
        self.C = Classifier(in_dim=self.feature_dim, n_cls=self.num_of_classes).to(self.device)
        self.D = DomainDiscriminator(in_dim=64).to(self.device)

        self.optimizer_F_and_C = torch.optim.Adam(list(self.F.parameters())+list(self.C.parameters()), lr=self.lr, betas=(self.b1, self.b2))
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.cls_loss = nn.CrossEntropyLoss()
        self.target_idx = target_idx
        self.log_write = open("./results/log_subject{}.txt".format(target_idx), "w")


    def get_source_data(self):

        src_X, src_y = [], []

        for s in range(1, 10):
            if s == self.target_idx: continue # Avoiding to be trained by target domain
            mat = scipy.io.loadmat(os.path.join(self.data_root, 'A0{}T'.format(s)))
            data = mat["data"]
            label = mat["label"].reshape(-1).astype(np.int64) - 1
            data = data[:, None, :, :]
            src_X.append(data)
            src_y.append(label)

        src_X = np.concatenate(src_X, axis=0)
        src_y = np.concatenate(src_y, axis=0)
        sourceData = (src_X - src_X.mean()) / src_X.std()
        sourceLabel = src_y

        mat_t = scipy.io.loadmat(os.path.join(self.data_root, 'A0{}T'.format(self.target_idx)))
        data_t = mat_t["data"]
        label_t = mat_t["label"].reshape(-1).astype(np.int64) - 1
        if data_t.shape[0] == 1000 and data_t.shape[2] == 288:
            data_t = np.transpose(data_t, (2, 1, 0))
        data_t = data_t[:, None, :, :]
        targetData = (data_t - data_t.mean()) / data_t.std()
        targetLabel = label_t

        mat_e = scipy.io.loadmat(os.path.join(self.data_root, 'A0{}E'.format(self.target_idx)))
        data_e = mat_e["data"]
        label_e = mat_e["label"].reshape(-1).astype(np.int64) - 1
        if data_e.shape[0] == 1000 and data_e.shape[2] == 288:
            data_e = np.transpose(data_e, (2, 1, 0))
        data_e = data_e[:, None, :, :]
        testData = (data_e - data_e.mean()) / data_e.std()
        testLabel = label_e

        return sourceData, sourceLabel, targetData, targetLabel, testData, testLabel

    def run(self, mu=0.2, lambda_gp=1.0, n_critic=3):
        bestAcc = 0
        averAcc = 0
        num = 0
        Y_true = 0
        Y_pred = 0

        # Setting DataLoader (Source Domain[T], Target Domain[T], Target Domain[E])
        sourceData, sourceLabel, targetData, targetLabel, testData, testLabel = self.get_source_data()

        dataset = torch.utils.data.TensorDataset(torch.from_numpy(sourceData), torch.from_numpy(sourceLabel))
        source_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        dataset = torch.utils.data.TensorDataset(torch.from_numpy(targetData), torch.from_numpy(targetLabel))
        target_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        dataset = torch.utils.data.TensorDataset(torch.from_numpy(testData), torch.from_numpy(testLabel))
        target_eval_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)


        # TRAINING PART
        for e in range(self.n_epochs):
            src_iter = iter(source_loader)
            tgt_iter = iter(target_loader)

            for _ in range(len(source_loader)):

                # Step 1. Maximization Step
                # Only Tune the Domain Discriminator
                self.F.eval() # Freeze
                self.C.eval() # Freeze
                self.D.train() # Training Only

                for _ in range(n_critic):
                    try: src_x, src_y = next(src_iter)
                    except StopIteration:
                        src_iter = iter(source_loader)
                        src_x, src_y = next(src_iter)
                    try: tgt_x, _ = next(tgt_iter)
                    except StopIteration:
                        tgt_iter = iter(target_loader)
                        tgt_x, _ = next(tgt_iter)

                    src_x = src_x.float().to(self.device)
                    tgt_x = tgt_x.float().to(self.device)

                    src_feat = self.F(src_x)
                    tgt_feat = self.F(tgt_x)
                    D_src = self.D(src_feat)
                    D_tgt = self.D(tgt_feat)

                    wasserstein = D_src.mean() - D_tgt.mean() # Calculating the wasserstein distance
                    gradient_penalty = compute_gradient_penalty(self.D, src_feat, tgt_feat, self.device)
                    d_loss = -(wasserstein - lambda_gp * gradient_penalty) # To maximize loss

                    self.optimizer_D.zero_grad()
                    d_loss.backward()
                    self.optimizer_D.step()

                # Step 2. Minimization Step
                # Only Tune the Classifier and Feature Extractor
                self.F.train() # Training
                self.C.train() # Training
                self.D.eval() # Freezing

                # Forward models in step 2.
                src_feat = self.F(src_x)
                tgt_feat = self.F(tgt_x)
                src_pred = self.C(src_feat)
                D_src = self.D(src_feat)
                D_tgt = self.D(tgt_feat)

                # Loss
                cls_loss = self.cls_loss(src_pred, src_y.to(self.device))
                wasserstein = D_src.mean() - D_tgt.mean() # Equation 5
                fc_loss = cls_loss + mu * wasserstein # Calculating the wasserstein distance

                self.optimizer_F_and_C.zero_grad()
                fc_loss.backward()
                self.optimizer_F_and_C.step()

            # EVALUATION PART
            self.F.eval()
            self.C.eval()
            self.D.eval()

            correct, total = 0, 0
            with torch.no_grad():
                for test_data, test_label in target_eval_loader:
                    test_data = test_data.float().to(self.device)
                    test_label = test_label.to(self.device)
                    feat = self.F(test_data)
                    out = self.C(feat)
                    pred = torch.max(out, 1)[1]
                    correct += (pred == test_label).sum().item()
                    total += test_label.size(0)

            acc = correct / total
            print('Epoch: {}, CLS Loss: {}, Discriminator Loss: {}, ACC: {}'.format(e+1, round(cls_loss.item(), 5), round(d_loss.item(), 5), round(acc, 4)))

            self.log_write.write(str(e) + " " + str(acc) + "\n")
            num = num + 1
            averAcc = averAcc + acc
            if acc > bestAcc:
                bestAcc = acc
                Y_true = test_label.cpu()
                Y_pred = pred.cpu()

        averAcc = averAcc / num
        print('The average accuracy is:', averAcc)
        print('The best accuracy is:', bestAcc)
        self.log_write.write('The average accuracy is: ' + str(averAcc) + "\n")
        self.log_write.write('The best accuracy is: ' + str(bestAcc) + "\n")
        return bestAcc, averAcc, Y_true, Y_pred



def main():

    result_write = open("./results/sub_result.txt", "w")
    seed = np.random.randint(2025)
    init_seed(seed)
    best = 0
    average = 0

    for i in range(1, 10):  # 1~9
        print('Seed is ' + str(seed))
        print('Subject %d' % (i))

        model = WGAN_MI_Classifier(target_idx=i)

        bestAcc, averageAcc = model.run()
        result_write.write('Subject ' + str(i) + ' : ' + 'Seed is: ' + str(seed) + "\n")
        result_write.write('Subject ' + str(i) + ' : ' + 'The best accuracy is: ' + str(bestAcc) + "\n")
        result_write.write('Subject ' + str(i) + ' : ' + 'The average accuracy is: ' + str(averageAcc) + "\n")

        best = best + bestAcc
        average = average + averAcc

    best = best / 9.0
    average = average / 9.0
    result_write.write('The average Best accuracy is: {}\n'.format(str(best)))
    result_write.write('The average Average accuracy is: {}\n'.format(str(average)))
    result_write.close()

if __name__ == "__main__":
    main()
