import numpy as np
import torch
from base import BaseTrainer
from utils.template_util import inf_loop, MetricTracker
import torch.nn.functional as F
from tqdm import tqdm

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader) if len_epoch is None else len_epoch
        self.data_loader = data_loader if len_epoch is None else inf_loop(data_loader)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None and len(self.valid_data_loader.sampler) != 0
        
        self.logger.info("training sample size: {}".format(len(self.data_loader.sampler) * 2))
        # self.logger.info("validation split: {}".format(self.config['data_loader']['args']['validation_split']))
        if self.do_validation:
            self.logger.info("validation sample size: {}".format(len(self.valid_data_loader.sampler) * 2))
            
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _get_inputs(self, vertices, eVals, eVecs, Ls, Ms, gradXs, gradYs):
        return (vertices, eVecs, eVals, Ls, Ms, gradXs, gradYs)

    def update_metrics(self, metric_name, value):
        for met in self.metric_ftns:
            if met.__name__ == metric_name:
                self.train_metrics.update(metric_name, met(value))
                break

    def set_writer_step(self, step, mode='train'):
        self.writer.set_step(step, mode)

    def _get_data_loader_with_progress(self, data_loader, desc):
        return tqdm(data_loader, desc=desc)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        for batch_idx, (file_names, vertices, eVals, eVecs, Ls, Ms, descriptors, gradXs, gradYs) in enumerate(self._get_data_loader_with_progress(self.data_loader, "Training batches")):
            self.optimizer.zero_grad()

            # Prepare inputs and perform forward pass
            inputs = self._get_inputs(vertices, eVals, eVecs, Ls, Ms, gradXs, gradYs)
            consistent_bases = self.model(inputs)
            loss, loss_details = self.criterion(eVals, consistent_bases, Ls, Ms, descriptors)

            # Extract additional losses
            off_penalty_loss = loss_details.get("off_penalty_loss", 0)
            pos_contrastive_loss = loss_details.get("pos_contrastive_loss", 0)
            ortho_loss = loss_details.get("ortho_loss", 0)

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            # Update metrics and log training progress
            self.set_writer_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())

            # Update additional metrics
            self.update_metrics("off_penalty_loss", off_penalty_loss)
            self.update_metrics("pos_contrastive_loss", pos_contrastive_loss)
            self.update_metrics("ortho_loss", ortho_loss)

            if batch_idx == self.len_epoch:
                break

        # Collect training log
        log = self.train_metrics.result()

        # Perform validation if applicable
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update({'val_loss': val_log['loss']})

        # Update learning rate scheduler
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()

        loss_list = []
        with torch.no_grad():
            for batch_idx, (file_names, vertices, eVals, eVecs, Ls, Ms, descriptors, gradXs, gradYs) in enumerate(self._get_data_loader_with_progress(self.valid_data_loader, "Validation batches")):
                # Prepare inputs and perform forward pass
                inputs = self._get_inputs(vertices, eVals, eVecs, Ls, Ms, gradXs, gradYs)
                consistent_bases = self.model(inputs)

                # Compute loss
                loss, _ = self.criterion(eVals, consistent_bases, Ls, Ms, descriptors)
                loss_list.append(loss.item())

        # Calculate average validation loss
        loss_avg = torch.mean(torch.tensor(loss_list)) if loss_list else 0
        self.set_writer_step(epoch, 'valid')
        self.valid_metrics.update('loss', loss_avg)

        return self.valid_metrics.result()