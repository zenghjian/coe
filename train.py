import argparse
import torch
import functools
import dataset.dataset as module_dataset
import data_loader.data_loaders as module_data_loader
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device, set_seed

def main(config):
    # Set random seed for reproducibility
    set_seed(config)
    logger = config.get_logger('train')

    # Setup dataset instances
    dataset = config.init_obj('dataset', module_dataset, split='train')
    valid_dataset = config.init_obj('dataset', module_dataset, split='valid')

    # Setup data loaders
    data_loader = config.init_obj('data_loader', module_data_loader, dataset=dataset, training=True)
    valid_data_loader = config.init_obj('data_loader', module_data_loader, dataset=valid_dataset, training=True)

    # Build model architecture
    model = config.init_obj('arch', module_arch, n_eig=data_loader.n_eig - 1, input_type=data_loader.input_type)
    logger.info(model)

    # Load pre-trained model if available
    load_model_if_available(config, model, logger)

    # Prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # Initialize loss, metrics, and optimizer
    loss_obj = config.init_obj('loss', module_loss)
    criterion = functools.partial(loss_obj.loss)
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    optimizer = config.init_obj('optimizer', torch.optim, filter(lambda p: p.requires_grad, model.parameters()))

    # Learning rate scheduler (optional)
    lr_scheduler = None  # Uncomment the next line if you want to use a scheduler
    # lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    # Initialize trainer and start training
    trainer = Trainer(model, criterion, metrics, optimizer, config, device, data_loader, valid_data_loader, lr_scheduler)
    trainer.train()


def load_model_if_available(config, model, logger):
    """
    Loads the model from a checkpoint if available.
    
    :param config: Configuration object
    :param model: Model to be loaded
    :param logger: Logger for printing information
    """
    if config.model_path is not None:
        logger.info('Loading checkpoint: {} ...'.format(str(config.model_path)))
        checkpoint = torch.load(str(config.model_path))
        if config['n_gpu'] == 1:
            # If loading from multi-GPU checkpoint to single GPU
            if 'module.' in list(checkpoint.keys())[0]:
                checkpoint = {k[len("module."):]: v for k, v in checkpoint.items()}
            model.load_state_dict(checkpoint)
        else:
            # If loading from single-GPU checkpoint to multi-GPU
            if 'module.' not in list(checkpoint.keys())[0]:
                checkpoint = {'module.' + k: v for k, v in checkpoint.items()}
            temp = torch.nn.DataParallel(model)
            temp.load_state_dict(checkpoint)
            model = temp.module


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)

    # Parse command line arguments
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='config_test.json', type=str, help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str, help='do not use it! (default: None)')
    args.add_argument('-d', '--dataset', default=None, choices=["faust, scape"], help='dataset to evaluate (default: None)')
    args.add_argument('-n', '--number', default=80, type=str, help='number of stored consistent bases (default: 80)')
    args.add_argument('-m', '--model_path', default=None, type=str, help='path to the model_best.pth file (default: None)')
    args.add_argument('--run_id', default=None, type=str, help='run_id for logger (default: None)')
    args.add_argument('--device', default=None, type=str, help='indices of GPUs to enable (default: all)')    

    config = ConfigParser.from_args(args)
    main(config)
