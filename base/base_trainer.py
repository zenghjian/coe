import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter
import time

class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')
        # self.consistent_error_threshold = cfg_trainer.get('consistent_error_threshold', None)

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance                
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        best_model = None  # Initialize to None to track whether the best model has been updated
        best_metric = float('inf') if self.mnt_mode == 'min' else -float('inf')  # Best metric initialization based on minimization or maximization
        best_epoch = 0
        early_stop_triggered = False  # Track if early stop was triggered
        time_counter = 0
        
        for epoch in range(self.start_epoch, self.epochs + 1):
            start_time = time.perf_counter()
            result = self._train_epoch(epoch)
            end_time = time.perf_counter()
            time_counter += end_time - start_time

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:30s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] < best_metric) or \
                            (self.mnt_mode == 'max' and log[self.mnt_metric] > best_metric)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    best_metric = log[self.mnt_metric]
                    not_improved_count = 0
                    best_model = self.model.state_dict()
                    best_epoch = epoch
                    self.logger.info(f"Found best model at epoch {best_epoch} with val_loss {best_metric}.")
                else:
                    not_improved_count += 1
                    self.logger.info(f"Validation performance didn't improve for {not_improved_count} epochs.")

                if not_improved_count > self.early_stop:
                    early_stop_triggered = True  # Mark that early stopping was triggered
                    self.logger.info("Validation performance didn't improve for {} epochs. "
                                    "Training stops.".format(self.early_stop))
                    self.logger.info(f"Best epoch: {best_epoch}")
                    torch.save(best_model, str(self.checkpoint_dir / 'model_best.pth'))
                    break

            if epoch % self.save_period == 0:
                filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
                torch.save(self.model.state_dict(), filename)

        # Check if the best model has been saved, if not, save it
        if not early_stop_triggered and best_model is not None:
            self.logger.info(f"Early stopping wasn't triggered or not configured. Saving best model from epoch {best_epoch} with metric {best_metric}.")
            torch.save(best_model, str(self.checkpoint_dir / 'model_best.pth'))
        elif best_model is None:
            self.logger.info(f"No improvement was observed, saving last model from epoch {epoch}.")
            torch.save(self.model.state_dict(), str(self.checkpoint_dir / 'model_best.pth'))

        self.logger.info(f"Average time per epoch: {time_counter / epoch} seconds")
        self.logger.info(f"Total time: {time_counter} seconds")
