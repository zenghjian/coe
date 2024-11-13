import argparse
import torch
import dataset.dataset as module_dataset
import data_loader.data_loaders as module_data_loader
import model.model as module_arch
from utils import set_seed
from parse_config import TestConfigParser

def main(config):
    # Set seed for reproducibility
    set_seed(config)
    logger = config.get_logger('test')

    # Setup dataset and data loader
    dataset = config.init_obj('dataset', module_dataset, split='test')
    data_loader = config.init_obj('data_loader', module_data_loader, dataset=dataset, training=False)

    # Build model architecture
    model = config.init_obj('arch', module_arch, n_eig=data_loader.n_eig - 1, input_type=data_loader.input_type)
    logger.info(model)

    # Load the model from checkpoint
    load_model_if_available(config, model, logger)

    # Prepare device for computation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Perform evaluation
    evaluate_model(config, model, data_loader, device, logger)


def load_model_if_available(config, model, logger):
    """
    Load the model from a checkpoint if available.
    
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


def evaluate_model(config, model, data_loader, device, logger):
    """
    Evaluate the model on the test dataset.
    
    :param config: Configuration object
    :param model: Model to evaluate
    :param data_loader: Data loader for the test dataset
    :param device: Device to perform evaluation on
    :param logger: Logger for printing information
    """
    logger.info("Storing evaluated consistent bases")

    embedding_collection = []
    file_identifiers = []

    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            file_path, vertices, eVals, eVecs, Ls, Ms, descriptors, gradXs, gradYs = data
            identifier = file_path[0].split('/')[-1].split('.')[0]
            file_identifiers.append(identifier)

            inputs = (vertices, eVecs, eVals, Ls, Ms, gradXs, gradYs)
            doubled_inputs = tuple(torch.cat([x, x], dim=1) for x in inputs)

            embedding = model(doubled_inputs)[:, 0, :, :].squeeze()

            if config.number is not None:
                number = int(config.number)
                if embedding.shape[1] < number:
                    raise ValueError(f"Requested {number} dimensions, but only {embedding.shape[1]} available.")
                embedding = embedding[:, :number]

            embedding_collection.append(embedding.detach().cpu().numpy())

    # Save embeddings
    config.save_eigenbases(embedding_collection, file_identifiers)
    logger.info("Evaluation complete. Consistent bases saved.")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-c', '--config', default='config_test.json', type=str, help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str, help='do not use it! (default: None)')
    args.add_argument('-d', '--dataset', default=None, choices=["faust","scape"], help='dataset to evaluate (default: None)')
    args.add_argument('-n', '--number', default=79, type=str, help='number of stored consistent bases (default: 79)')
    args.add_argument('-m', '--model_path', default=None, type=str, help='path to the model_best.pth file (default: None)')
    args.add_argument('--run_id', default=None, type=str, help='run_id for logger (default: None)')
    args.add_argument('--device', default=None, type=str, help='indices of GPUs to enable (default: all)')    

    config = TestConfigParser.from_args(args)
    main(config)