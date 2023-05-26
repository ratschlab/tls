import logging
import os
import pickle

import gin
import numpy as np
import torch
import torch.nn.functional as F
from ignite.contrib.metrics import AveragePrecision, ROC_AUC, PrecisionRecallCurve, RocCurve
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from tls.models.utils import save_model, load_model_state


class MultiOutput_Binary_Metric:
    def __init__(self, metric, num_outputs=4, output_dim=0):
        self.num_outputs = num_outputs
        self.output_dim = output_dim
        self.metric = metric()

    def reset(self):
        self.metric.reset()

    def update(self, output):
        y_pred, y = map(lambda x: x.reshape(-1, self.num_outputs), output)
        output = (y_pred[:, self.output_dim], y[:, self.output_dim])
        self.metric.update(output)

    def compute(self):
        return self.metric.compute()


def l1_reg(embedding_module):
    n_params = sum(len(p.reshape(-1, )) for p in embedding_module.parameters())
    return sum(torch.abs(p).sum() for p in embedding_module.parameters()) / n_params


def focal_binary_loss(output, label, gamma_focal, reduction='mean'):
    log_prob = torch.nn.functional.log_softmax(output, dim=-1)  # or dim = 2? might have last dim for 1-p and p.
    prob = torch.exp(log_prob)

    # Trick to get focal loss to work with non-binary labels
    scaling_term = ((label[..., 1] - prob[..., 1]) ** 2) ** (gamma_focal / 2)

    # Manually compute nll_loss as pytorch only accepts one-hot targets:
    loss = -(scaling_term.unsqueeze(1) * label * log_prob).sum(dim=-1)

    if reduction == 'none':
        return loss
    if reduction == 'mean':
        return torch.mean(loss)
    if reduction == 'sum':
        return torch.sum(loss)


@gin.configurable('SurvivalWrapper')
class SurvivalWrapper(object):
    def __init__(self, encoder=gin.REQUIRED, optimizer_fn=gin.REQUIRED, reg='l1', reg_weight=1e-3, lr_decay=1.0,
                 pred_horizon=gin.REQUIRED, objective_type='landmarking', upweight_p_one=1.0, alpha_ddrsa=0.5):

        self.upweight_p_one = upweight_p_one
        self.batch_correction = False
        self.smooth_labels = False
        self.pred_horizon = pred_horizon
        if torch.cuda.is_available():
            logging.info('Model will be trained using GPU Hardware')
            device = torch.device('cuda')
            self.pin_memory = True
            self.n_worker = 8
        else:
            logging.info('Model will be trained using CPU Hardware. This should be considerably slower')
            self.pin_memory = False
            self.n_worker = 16
            device = torch.device('cpu')

        self.device = device
        self.encoder = encoder
        self.encoder.to(device)
        self.optimizer = optimizer_fn(self.encoder.parameters())
        self.objective_type = objective_type
        if lr_decay < 1.0:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=lr_decay)
        else:
            self.scheduler = None
        self.scaler = None

        # Regularization set-up
        if reg is None:
            self.reg_fn = None
        elif reg == 'l1':
            self.reg_fn = l1_reg
        self.reg_weight = reg_weight
        self.alpha = alpha_ddrsa

    def set_logdir(self, logdir):
        self.logdir = logdir

    def set_scaler(self, scaler):
        self.scaler = scaler

    def get_eep_pred_from_hazard(self, hazard_pred):
        log_hs = torch.log(torch.sigmoid(hazard_pred))
        eep_curve = 1 - torch.exp(torch.cumsum(log_hs - hazard_pred, dim=-1))
        return eep_curve

    def set_metrics(self):
        pred_horizon = self.pred_horizon

        def surv_output_transform(output):
            with torch.no_grad():
                y_pred, y = output
                labeled = torch.any((y != -1), dim=-1)
                # We check that labeled are equal to the early event labels in the loader
                eep_preds = self.get_eep_pred_from_hazard(y_pred)
                labels = torch.any((y[:, :, :pred_horizon] == 1), dim=-1).to(torch.float32)
                return torch.masked_select(eep_preds[..., self.pred_horizon], labeled), torch.masked_select(labels,
                                                                                                            labeled)

        self.task_type = 'binary_classification'
        self.output_transform = surv_output_transform
        self.metrics = {'PR': AveragePrecision(), 'AUC': ROC_AUC(),
                        'PR_Curve': PrecisionRecallCurve(), 'ROC_Curve': RocCurve()}

    def get_weight(self, labels, smooth_labels, loss_weight):
        with torch.no_grad():
            if loss_weight is not None:
                if self.agg_type == 'class':
                    samples_weight = (1 - labels) * loss_weight[0] + labels * loss_weight[1]
                elif self.agg_type == 'patient':
                    pos_samples = torch.any(labels == 1, dim=1, keepdim=True).float()
                    samples_weight = torch.ones_like(labels)
                    samples_weight *= ((1 - pos_samples) * loss_weight[0] + pos_samples * loss_weight[1])
                elif self.agg_type == 'event':
                    pos_samples = (smooth_labels[..., 0] > 0).float() * (smooth_labels[..., 1] < 1).float()
                    samples_weight = torch.ones_like(labels)
                    samples_weight *= ((1 - pos_samples) * loss_weight[0] + pos_samples * loss_weight[1])
            else:
                samples_weight = torch.ones_like(labels)
        return samples_weight

    def loss_fn(self, output, label, loss_weight, mask):
        """Compute loss based on the model hazard output and ground-truth label."""

        # This is a MLE with landmarking
        if self.objective_type == 'landmarking':
            eep_mask = torch.any((mask != 0), dim=-1)
            loss_sample = (F.binary_cross_entropy_with_logits(output, label, reduction='none') * mask).sum(dim=-1)
            loss = torch.masked_select(loss_sample, eep_mask).mean()
            return loss

        if self.objective_type == 'ddrsa':
            eep_mask = torch.any((mask != 0), dim=-1)
            uncensored_mask = torch.any((label == 1), dim=-1)
            censored_mask = torch.all((label != 1), dim=-1)

            # contains -log(S(T_i)) = log(P(T_e>T_i)) if c_i=1 and -log(f(T_i)) = log(P(T_e = T_i))
            loss_sample = (F.binary_cross_entropy_with_logits(output, label, reduction='none') * mask).sum(dim=-1)

            l_z = 0
            l_u = 0
            if (eep_mask * uncensored_mask).sum() != 0:
                l_z += (1 - self.alpha) * torch.masked_select(loss_sample, eep_mask * uncensored_mask).mean()
                event_rate_output = self.get_eep_pred_from_hazard(output)
                loss_event_rate = (F.binary_cross_entropy_with_logits(event_rate_output, label,
                                                                      reduction='none') * label * mask).sum(dim=-1)
                # We select only uncensored and average.
                l_u += self.alpha * torch.masked_select(loss_event_rate, eep_mask * uncensored_mask).mean()
            l_c = 0
            if (eep_mask * censored_mask).sum() != 0:
                l_c += self.alpha * torch.masked_select(loss_sample, eep_mask * censored_mask).mean()


            # This return -log(P(T_e <= T_i)) for event that are uncensored 0 otherwise

            return l_u + l_c + l_z

        # This is TCSR without gradient from other loss term than your own
        elif self.objective_type == 'tcsr-no-grad':
            with torch.no_grad():
                labels_p1 = label[..., :1]  # the immediate next step label is the true label
                labels_tc = torch.cat([torch.sigmoid(output[:, 1:, :-1]), label[:, -1:, 1:]],
                                      dim=1)  # last line has at max 1 non-masked pred so we don't care about the padding
                labels_tc = torch.cat([labels_p1, labels_tc], dim=-1)
                survival_probs = 1 - self.get_eep_pred_from_hazard(output)  # eep  is failure so S =  1 - eep
                loss_weight_p1 = mask[..., :1] * self.upweight_p_one  # 1 for all consired samples => w_m1=1
                loss_weight_p2 = 1 - labels_p1  # w_m2 = S(0|x_k)  = 0 if else 1 thus S(0|x_k) == 1- label_p1.
                loss_weigth_tc = torch.cat([survival_probs[:, 2:, :-2], mask[:, -2:, 2:]], dim=1)
                loss_weigth_tc = torch.cat([loss_weight_p1, loss_weight_p2, loss_weigth_tc], dim=-1)
            eep_mask = torch.any((mask != 0), dim=-1)  # We only train on the samples where we would train eep
            loss_sample = (F.binary_cross_entropy_with_logits(output, labels_tc,
                                                              reduction='none') * loss_weigth_tc * mask).sum(dim=-1)
            loss = torch.masked_select(loss_sample, eep_mask).mean()
            return loss

        # This is TCSR with TC on both weights (Surv) and predictions (Hazard)
        elif self.objective_type == 'tcsr':
            labels_p1 = label[..., :1]  # the immediate next step label is the true label
            labels_tc = torch.cat([torch.sigmoid(output[:, 1:, :-1]), label[:, -1:, 1:]],
                                  dim=1)  # last line has at max 1 non-masked pred so we don't care about the padding
            labels_tc = torch.cat([labels_p1, labels_tc], dim=-1)
            survival_probs = 1 - self.get_eep_pred_from_hazard(output)  # eep  is failure so S =  1 - eep
            loss_weight_p1 = mask[..., :1] * self.upweight_p_one  # 1 for all consired samples => w_m1=1
            loss_weight_p2 = 1 - labels_p1  # w_m2 = S(0|x_k)  = 0 if else 1 thus S(0|x_k) == 1- label_p1.
            loss_weigth_tc = torch.cat([survival_probs[:, 2:, :-2], mask[:, -2:, 2:]], dim=1)
            loss_weigth_tc = torch.cat([loss_weight_p1, loss_weight_p2, loss_weigth_tc], dim=-1)
            eep_mask = torch.any((mask != 0), dim=-1)  # We only train on the samples where we would train eep
            loss_sample = (F.binary_cross_entropy_with_logits(output, labels_tc,
                                                              reduction='none') * loss_weigth_tc * mask).sum(dim=-1)
            loss = torch.masked_select(loss_sample, eep_mask).mean()
            return loss

    def step_fn(self, element, loss_weight=None):

        data, labels, mask, event = element[0], element[1].to(self.device), element[2].to(self.device), element[3].to(
            self.device)
        data = data.float().to(self.device)

        # We remove excess padding given the current batch
        true_length = torch.where(torch.any(data != 0.0, axis=0))[0][-1] + 1  # We need to use data instead of mask
        data = data[:, :true_length]
        labels = labels[:, :true_length]
        mask = mask[:, :true_length]

        out = self.encoder(data)
        samples_weight = mask.to(torch.float32)
        loss = self.loss_fn(out, labels, samples_weight, mask)
        # print(loss)
        if self.reg_fn is not None:
            loss += self.reg_weight * self.reg_fn(self.encoder.embedding_layer)
        return loss, out, labels

    def _do_training(self, train_loader, weight, metrics):
        # Training epoch
        train_loss = []
        self.encoder.train()
        tot_elem = 0
        for t, elem in tqdm(enumerate(train_loader)):
            for k in range(self.repeat_step):
                loss, preds, target = self.step_fn(elem, weight)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                train_loss.append(loss)
                tot_elem += 1
                for _, metric in metrics.items():
                    metric.update(self.output_transform((preds, target)))
        train_metric_results = {}
        for name, metric in metrics.items():
            train_metric_results[name] = metric.compute()
            metric.reset()
        train_loss = float(sum(train_loss) / (tot_elem))
        return train_loss, train_metric_results

    @gin.configurable(module='SurvivalWrapper')
    def train(self, train_dataset, val_dataset, weight,
              epochs=gin.REQUIRED, batch_size=gin.REQUIRED, patience=gin.REQUIRED,
              min_delta=gin.REQUIRED, save_weights=True, agg_type='class', repeat_step=1):

        logging.info('Setting up metrics and DataLoader')
        self.set_metrics()
        metrics = self.metrics
        self.batch_size = batch_size
        self.agg_type = agg_type
        self.repeat_step = repeat_step

        torch.autograd.set_detect_anomaly(True)  # Check for any nans in gradients
        if not train_dataset.h5_loader.on_RAM:
            self.n_worker = 1
            logging.info('Data is not loaded to RAM, thus number of worker has been set to 1')

        self.smooth = train_dataset.smooth_labels
        if self.smooth:
            self.smooth_labels = torch.from_numpy(train_dataset.h5_loader.smooth_labels).to(self.device).to(
                torch.float32)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=self.n_worker,
                                  pin_memory=self.pin_memory, prefetch_factor=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=self.n_worker,
                                pin_memory=self.pin_memory, prefetch_factor=2)

        with torch.no_grad():
            neg, pos = train_dataset.get_balance('balanced_event')
            self.encoder.logit.bias.copy_(self.encoder.logit.bias - np.log(pos / neg))
            print(np.log(pos / neg))

        best_loss = float('inf')
        epoch_no_improvement = 0
        train_writer = SummaryWriter(os.path.join(self.logdir, 'tensorboard', 'train'))
        val_writer = SummaryWriter(os.path.join(self.logdir, 'tensorboard', 'val'))

        logging.info('=== Starting training ===')
        for epoch in range(epochs):
            # Train step
            train_loss, train_metric_results = self._do_training(train_loader, weight, metrics)

            # Validation step
            val_loss, val_metric_results = self.evaluate(val_loader, metrics, weight)

            if self.scheduler is not None:
                self.scheduler.step()

            # Early stopping
            if val_loss <= best_loss - min_delta:
                best_metrics = val_metric_results
                epoch_no_improvement = 0
                if save_weights:
                    self.save_weights(epoch, os.path.join(self.logdir, 'model.torch'))
                best_loss = val_loss
                logging.info('Validation loss improved to {:.4f} '.format(val_loss))
            else:
                epoch_no_improvement += 1
                logging.info('No improvement on loss for {} epochs'.format(epoch_no_improvement))
            if epoch_no_improvement >= patience:
                logging.info('No improvement on loss for more than {} epochs. We stop training'.format(patience))
                break

            # Logging
            train_string = 'Train Epoch:{}'
            train_values = [epoch + 1]
            for name, value in train_metric_results.items():
                if 'Curve' not in name.split('_')[-1]:
                    train_string += ', ' + name + ':{:.4f}'
                    train_values.append(value)
                    train_writer.add_scalar(name, value, epoch)
            train_writer.add_scalar('Loss', train_loss, epoch)
            if self.scheduler is not None:
                train_writer.add_scalar('Learning rate', self.scheduler.get_last_lr()[0], epoch)

            val_string = 'Val Epoch:{}'
            val_values = [epoch + 1]
            for name, value in val_metric_results.items():
                if 'Curve' not in name.split('_')[-1]:
                    val_string += ', ' + name + ':{:.4f}'
                    val_values.append(value)
                    val_writer.add_scalar(name, value, epoch)
            val_writer.add_scalar('Loss', val_loss, epoch)

            logging.info(train_string.format(*train_values))
            logging.info(val_string.format(*val_values))

        with open(os.path.join(self.logdir, 'val_metrics.pkl'), 'wb') as f:
            best_metrics['loss'] = best_loss
            pickle.dump(best_metrics, f)

        logging.info('=== Finished training ===')
        self.load_weights(os.path.join(self.logdir, 'model.torch'))  # We load back the best iteration

    def test(self, dataset, weight, test_filename='test_metrics.pkl'):
        self.set_metrics()
        test_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_worker,
                                 pin_memory=self.pin_memory)
        if isinstance(weight, list):
            weight = torch.FloatTensor(weight).to(self.device)
        test_loss, test_metrics = self.evaluate(test_loader, self.metrics, weight)

        with open(os.path.join(self.logdir, test_filename), 'wb') as f:
            test_metrics['loss'] = test_loss
            pickle.dump(test_metrics, f)
        for key, value in test_metrics.items():
            if isinstance(value, float):
                logging.info('Test {} :  {}'.format(key, value))

    def evaluate(self, eval_loader, metrics, weight):
        self.encoder.eval()
        eval_loss = []
        with torch.no_grad():
            tot_elem = 0
            for v, elem in enumerate(eval_loader):
                loss, preds, target = self.step_fn(elem, weight)
                eval_loss.append(loss)
                tot_elem += 1
                for name, metric in metrics.items():
                    metric.update(self.output_transform((preds, target)))

            eval_metric_results = {}
            for name, metric in metrics.items():
                eval_metric_results[name] = metric.compute()
                metric.reset()
        eval_loss = float(sum(eval_loss) / (tot_elem))
        return eval_loss, eval_metric_results

    def save_weights(self, epoch, save_path):
        save_model(self.encoder, self.optimizer, epoch, save_path)

    def load_weights(self, load_path):
        load_model_state(load_path, self.encoder, optimizer=self.optimizer)


@gin.configurable('DLWrapper')
class DLWrapper(object):
    def __init__(self, encoder=gin.REQUIRED, optimizer_fn=gin.REQUIRED, reg='l1', reg_weight=1e-3, lr_decay=1.0,
                 gamma_focal=None):

        self.batch_correction = False
        self.smooth_labels = False
        self.gamma_focal = gamma_focal
        if torch.cuda.is_available():
            logging.info('Model will be trained using GPU Hardware')
            device = torch.device('cuda')
            self.pin_memory = True
            self.n_worker = 4
        else:
            logging.info('Model will be trained using CPU Hardware. This should be considerably slower')
            self.pin_memory = False
            self.n_worker = 16
            device = torch.device('cpu')

        self.device = device
        self.encoder = encoder
        self.encoder.to(device)
        self.optimizer = optimizer_fn(self.encoder.parameters())
        if lr_decay < 1.0:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=lr_decay)
        else:
            self.scheduler = None
        self.scaler = None

        # Regularization set-up
        if reg is None:
            self.reg_fn = None
        elif reg == 'l1':
            self.reg_fn = l1_reg
        self.reg_weight = reg_weight

    def set_logdir(self, logdir):
        self.logdir = logdir

    def set_scaler(self, scaler):
        self.scaler = scaler

    def set_metrics(self):
        def softmax_binary_output_transform(output):
            with torch.no_grad():
                y_pred, y = output
                y_pred = torch.softmax(y_pred, dim=1)
                return y_pred[:, -1], y

        def softmax_multi_output_binary_transform(output):
            with torch.no_grad():
                y_pred, y = output
                y_pred = torch.sigmoid(y_pred)
                return y_pred, y

        # output transform is not applied for contrib metrics so we do our own.
        if self.encoder.logit.out_features == 2:
            self.task_type = 'binary_classification'
            self.output_transform = softmax_binary_output_transform
            self.metrics = {'PR': AveragePrecision(), 'AUC': ROC_AUC(),
                            'PR_Curve': PrecisionRecallCurve(), 'ROC_Curve': RocCurve()}

        else:  # MULTIHORIZON
            self.task_type = 'multioutput_binary_classification'
            self.output_transform = softmax_multi_output_binary_transform
            metrics = {'PR': AveragePrecision, 'AUC': ROC_AUC,
                       'PR_Curve': PrecisionRecallCurve, 'ROC_Curve': RocCurve}
            self.metrics = {}
            num_outputs = self.encoder.logit.out_features
            for met in metrics.keys():
                for out in range(num_outputs):
                    self.metrics[met + f"{out}"] = MultiOutput_Binary_Metric(metrics[met],
                                                                             num_outputs=num_outputs,
                                                                             output_dim=out)

    def loss_fn(self, output, label, loss_weight):
        """Compute loss based on flattened model output and ground-truth label."""

        if self.task_type in ['multiclass_classification', 'binary_classification']:
            if self.gamma_focal is None:
                return torch.mean(torch.nn.functional.cross_entropy(output, label, reduction='none') * loss_weight)
            else:
                return torch.mean(
                    focal_binary_loss(output, label, gamma_focal=self.gamma_focal, reduction='none') * loss_weight)
        else:  # multi-horizon
            return torch.mean(
                torch.nn.functional.binary_cross_entropy_with_logits(output, label, reduction='none') * loss_weight)

    def accumulate_logits(self, logits):
        """Sets all but first negative logits to zero and cumulative sums them."""
        # Create array of positive logits.
        zeros = torch.zeros_like(logits)
        positive_logits = torch.where(logits > 0, logits, zeros)
        # Create bool mask that only excludes first lookahead window.
        mask = torch.ones_like(logits)
        bool_mask = torch.greater(torch.cumsum(mask, axis=-1), 1)
        # Replace original logits with positive values over bool mask.
        transformed_logits = torch.where(bool_mask, positive_logits, logits)
        logits = torch.cumsum(transformed_logits, axis=-1)
        return logits

    def get_weight(self, labels, smooth_labels, loss_weight):
        with torch.no_grad():
            if loss_weight is not None:
                if self.agg_type == 'class':
                    samples_weight = (1 - labels) * loss_weight[0] + labels * loss_weight[1]
                elif self.agg_type == 'patient':
                    pos_samples = torch.any(labels == 1, dim=1, keepdim=True).float()
                    samples_weight = torch.ones_like(labels)
                    samples_weight *= ((1 - pos_samples) * loss_weight[0] + pos_samples * loss_weight[1])
                elif self.agg_type == 'event':
                    pos_samples = (smooth_labels[..., 0] > 0).float() * (smooth_labels[..., 1] < 1).float()
                    samples_weight = torch.ones_like(labels)
                    samples_weight *= ((1 - pos_samples) * loss_weight[0] + pos_samples * loss_weight[1])
            else:
                samples_weight = torch.ones_like(labels)
        return samples_weight

    def step_fn(self, element, loss_weight=None):

        data, labels, mask = element[0], element[1].to(self.device), element[2].to(self.device)
        data = data.float().to(self.device)

        # We remove excess padding given the current batch
        true_length = torch.where(torch.any(data != 0.0, axis=0))[0][-1] + 1  # We need to use data instead of mask
        data = data[:, :true_length]
        labels = labels[:, :true_length]
        mask = mask[:, :true_length]

        out = self.encoder(data)
        if self.task_type != 'multioutput_binary_classification':

            if self.smooth_labels and len(labels.shape) == 3:  # we have smoothed labels and in train/val
                true_labels = labels[..., 1]  # We use smooth label for loss
                samples_weight = self.get_weight(labels[..., 0], true_labels, loss_weight)
            else:  # test or non-smoothed labels
                true_labels = labels
                samples_weight = self.get_weight(true_labels, true_labels, loss_weight)

            out_flat = torch.masked_select(out, mask.unsqueeze(-1)).reshape(-1, out.shape[-1])
            label_flat = torch.masked_select(true_labels, mask)
            samples_weight_flat = torch.masked_select(samples_weight, mask)

            label_flat = torch.stack([1 - label_flat, label_flat], dim=-1).float()

        else:  # Multi-horizon
            if self.smooth_labels and len(labels.shape) == 4:  # we have smoothed labels and in train/val
                true_labels = labels[..., 1]  # We use smooth label for loss
                samples_weight = self.get_weight(labels[..., 0], true_labels, loss_weight)
            else:
                true_labels = labels
                samples_weight = self.get_weight(true_labels, true_labels, loss_weight)
            out = self.accumulate_logits(out)
            out_flat = torch.masked_select(out, mask.unsqueeze(-1))
            label_flat = torch.masked_select(true_labels, mask.unsqueeze(-1))  # .reshape(-1, labels.shape[-1])
            samples_weight_flat = torch.masked_select(samples_weight,
                                                      mask.unsqueeze(-1))  # .reshape(-1, labels.shape[-1])

        loss = self.loss_fn(out_flat, label_flat, samples_weight_flat)

        if self.reg_fn is not None:
            loss += self.reg_weight * self.reg_fn(self.encoder.embedding_layer)

        if self.smooth_labels and len(labels.shape) == 4:  # We use hard labels for metrics
            label_flat = torch.masked_select(labels[..., 0], mask.unsqueeze(-1))
        elif self.task_type == 'multioutput_binary_classification':
            label_flat = torch.masked_select(labels, mask.unsqueeze(-1))
        elif self.smooth_labels and len(labels.shape) == 3:  # We use hard labels for metrics
            label_flat = torch.masked_select(labels[..., 0], mask)
        else:
            label_flat = torch.masked_select(labels, mask)

        return loss, out_flat, label_flat

    def _do_training(self, train_loader, weight, metrics):
        # Training epoch
        train_loss = []
        self.encoder.train()
        tot_elem = 0
        for t, elem in tqdm(enumerate(train_loader)):
            loss, preds, target = self.step_fn(elem, weight)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            train_loss.append(loss)
            tot_elem += 1
            for _, metric in metrics.items():
                metric.update(self.output_transform((preds, target)))

        train_metric_results = {}
        for name, metric in metrics.items():
            train_metric_results[name] = metric.compute()
            metric.reset()
        train_loss = float(sum(train_loss) / (tot_elem))
        return train_loss, train_metric_results

    @gin.configurable(module='DLWrapper')
    def train(self, train_dataset, val_dataset, weight,
              epochs=gin.REQUIRED, batch_size=gin.REQUIRED, patience=gin.REQUIRED,
              min_delta=gin.REQUIRED, save_weights=True, agg_type='class'):

        logging.info('Setting up metrics and DataLoader')
        self.set_metrics()
        metrics = self.metrics
        self.batch_size = batch_size
        self.agg_type = agg_type

        torch.autograd.set_detect_anomaly(True)  # Check for any nans in gradients
        if not train_dataset.h5_loader.on_RAM:
            self.n_worker = 1
            logging.info('Data is not loaded to RAM, thus number of worker has been set to 1')

        self.smooth_labels = train_dataset.smooth_labels

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=self.n_worker,
                                  pin_memory=self.pin_memory, prefetch_factor=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=self.n_worker,
                                pin_memory=self.pin_memory, prefetch_factor=2)

        if isinstance(weight, list):  # We give exact class-weight
            weight = torch.FloatTensor(weight).to(self.device)
        elif isinstance(weight, float):  # We give positive class prevalance
            weight = torch.FloatTensor([0.5 / (1 - weight), 0.5 / (weight)]).to(self.device)
        elif weight == 'balanced':
            weight = torch.FloatTensor(train_dataset.get_balance(weight)).to(self.device)
        elif weight == 'balanced_patient':
            weight = torch.FloatTensor(train_dataset.get_balance(weight)).to(self.device)

        best_loss = float('inf')
        epoch_no_improvement = 0
        train_writer = SummaryWriter(os.path.join(self.logdir, 'tensorboard', 'train'))
        val_writer = SummaryWriter(os.path.join(self.logdir, 'tensorboard', 'val'))

        logging.info('=== Starting training ===')
        for epoch in range(epochs):
            # Train step
            train_loss, train_metric_results = self._do_training(train_loader, weight, metrics)

            # Validation step
            val_loss, val_metric_results = self.evaluate(val_loader, metrics, weight)

            if self.scheduler is not None:
                self.scheduler.step()

            # Early stopping
            if val_loss <= best_loss - min_delta:
                best_metrics = val_metric_results
                epoch_no_improvement = 0
                if save_weights:
                    self.save_weights(epoch, os.path.join(self.logdir, 'model.torch'))
                best_loss = val_loss
                logging.info('Validation loss improved to {:.4f} '.format(val_loss))
            else:
                epoch_no_improvement += 1
                logging.info('No improvement on loss for {} epochs'.format(epoch_no_improvement))
            if epoch_no_improvement >= patience:
                logging.info('No improvement on loss for more than {} epochs. We stop training'.format(patience))
                break

            # Logging
            train_string = 'Train Epoch:{}'
            train_values = [epoch + 1]
            for name, value in train_metric_results.items():
                if 'Curve' not in name.split('_')[-1]:
                    train_string += ', ' + name + ':{:.4f}'
                    train_values.append(value)
                    train_writer.add_scalar(name, value, epoch)
            train_writer.add_scalar('Loss', train_loss, epoch)
            if self.scheduler is not None:
                train_writer.add_scalar('Learning rate', self.scheduler.get_last_lr()[0], epoch)

            val_string = 'Val Epoch:{}'
            val_values = [epoch + 1]
            for name, value in val_metric_results.items():
                if 'Curve' not in name.split('_')[-1]:
                    val_string += ', ' + name + ':{:.4f}'
                    val_values.append(value)
                    val_writer.add_scalar(name, value, epoch)
            val_writer.add_scalar('Loss', val_loss, epoch)

            logging.info(train_string.format(*train_values))
            logging.info(val_string.format(*val_values))

        with open(os.path.join(self.logdir, 'val_metrics.pkl'), 'wb') as f:
            best_metrics['loss'] = best_loss
            pickle.dump(best_metrics, f)

        logging.info('=== Finished training ===')
        self.load_weights(os.path.join(self.logdir, 'model.torch'))  # We load back the best iteration

    def test(self, dataset, weight, test_filename='test_metrics.pkl'):
        self.set_metrics()
        test_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_worker,
                                 pin_memory=self.pin_memory)
        if isinstance(weight, list):
            weight = torch.FloatTensor(weight).to(self.device)
        test_loss, test_metrics = self.evaluate(test_loader, self.metrics, weight)

        with open(os.path.join(self.logdir, test_filename), 'wb') as f:
            test_metrics['loss'] = test_loss
            pickle.dump(test_metrics, f)
        for key, value in test_metrics.items():
            if isinstance(value, float):
                logging.info('Test {} :  {}'.format(key, value))

    def evaluate(self, eval_loader, metrics, weight):
        self.encoder.eval()
        eval_loss = []

        with torch.no_grad():
            tot_elem = 0
            for v, elem in enumerate(eval_loader):
                loss, preds, target = self.step_fn(elem, weight)
                eval_loss.append(loss)
                tot_elem += 1
                for name, metric in metrics.items():
                    metric.update(self.output_transform((preds, target)))

            eval_metric_results = {}
            for name, metric in metrics.items():
                eval_metric_results[name] = metric.compute()
                metric.reset()
        eval_loss = float(sum(eval_loss) / (tot_elem))
        return eval_loss, eval_metric_results

    def save_weights(self, epoch, save_path):
        save_model(self.encoder, self.optimizer, epoch, save_path)

    def load_weights(self, load_path):
        load_model_state(load_path, self.encoder, optimizer=self.optimizer)
