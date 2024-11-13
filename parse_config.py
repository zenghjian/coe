import os
import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
from logger import setup_logging
from utils import read_json, write_json
import shutil
import torch
class ConfigParser:
    def __init__(self, config, resume=None, modification=None, run_id=None, model_path=None, dataset=None, number=None):
        """
        class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving
        and logging module.
        :param config: Dict containing configurations, hyperparameters for training. contents of `config.json` file for example.
        :param resume: String, path to the checkpoint being loaded.
        :param modification: Dict keychain:value, specifying position values to be replaced from config dict.
        :param run_id: Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default
        """
        # load config file and apply modification
        self._config = _update_config(config, modification)
        self.resume = resume
        self.model_path = model_path
        self.dataset = dataset
        self.number = number
        # set save_dir where trained model and log will be saved.
        save_dir = Path(self.config['trainer']['save_dir'])

        exper_name = self.config['name']
        if run_id is None: # use timestamp as default run-id
            run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        
        # name experiment with timestamp
        name = f'{run_id}_{exper_name}'
        self._save_dir = save_dir / name 
        self._log_dir = self._save_dir / 'log'

        # make directory for saving checkpoints and log.
        exist_ok = run_id == ''
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)

        # save updated config file to the checkpoint dir
        write_json(self.config, self.log_dir / 'config.json')

        # configure logging module
        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

        # track run_id
        self.run_id = run_id
        if self.model_path is not None:
            self.model_path = Path(self.model_path)
            self.checkpoint_dir = self.model_path

    @classmethod
    def from_args(cls, args, options=''):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple):
            args = args.parse_args()

        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        if args.run_id is not None:
            run_id = args.run_id
        else:
            run_id = None
        if args.model_path is not None:
            model_path = args.model_path
        else:
            model_path = None
        if args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = resume.parent / 'config.json'
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            resume = None
            cfg_fname = Path(args.config)
        
        if args.dataset is not None:
            dataset = args.dataset
        else:
            dataset = None
        
        if args.number is not None:
            number = args.number
        else:
            number = None

        config = read_json(cfg_fname)
        if args.config and resume:
            # update new config for fine-tuning
            config.update(read_json(args.config))

        # parse custom cli options into dictionary
        modification = {opt.target : getattr(args, _get_opt_name(opt.flags)) for opt in options}
        return cls(config, resume, modification, run_id, model_path, dataset, number)

    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def get(self, name, default=None):
        return self.config.get(name, default)

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger
    
    def save_eigenbases(self, eigenbases, identifiers, partial=False):
        """
        Save computed eigenbases to the directory with identifiers.
        
        :param eigenbases: List of computed eigenbases tensors to be saved.
        :param identifiers: List of identifiers for naming the saved files.
        """
        save_eigenbases_dir = self.save_dir / "consistent_bases"
        print("ready to save consistent bases...")

        print("saving consistent bases...")
        os.makedirs(save_eigenbases_dir, exist_ok=True)
        
        for eigenbasis, identifier in zip(eigenbases, identifiers):
            if partial:
                # for partial we save the identifier=file_name
                torch.save(eigenbasis, save_eigenbases_dir / f'{identifier}.pt')
            else:
                torch.save(eigenbasis, save_eigenbases_dir / f'consistent_bases_{identifier}.pt')
            
        print("save done!")



    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir
    
class TestConfigParser(ConfigParser):
    def __init__(self, config,  resume=None, modification=None, run_id=None, model_path=None, dataset=None, number=None):
        """
        Initialize the TestConfigParser with a specific model path instead of run_id.
        :param config: Dict containing configurations for training.
        :param model_path: String, the path where the model and logs will be saved.
        :param modification: Dict keychain:value, specifying position values to be replaced from config dict.
        """
        # Load config file and apply modification
        self._config = _update_config(config, modification)
        self.resume = resume
        self.model_path = model_path
        self.dataset = dataset
        self.number = number
        if not model_path:
            raise ValueError("model_path must be provided")
        model_path = Path(model_path)

        # Check if model_path is a directory or a file (assuming checkpoint)
        if model_path.is_file():
            # Extract the checkpoint name without extension for folder naming
            checkpoint_name = model_path.stem
            formatted_checkpoint_name = f'test_{checkpoint_name}'            
        else:
            raise ValueError("model_path should point to a checkpoint file")

        model_dir = model_path.parent
        log_path_str = str(model_dir).replace('models', 'log', 1)  # Ensure only the first occurrence is replaced
        log_path = Path(log_path_str)

        test_run_id = datetime.now().strftime('%m%d_%H%M%S')
        self._save_dir = model_path.parent / formatted_checkpoint_name / test_run_id
        self._log_dir = log_path / formatted_checkpoint_name / test_run_id


        # Make directory for saving checkpoints and log
        self._make_dir(self._save_dir)
        self._make_dir(self._log_dir)

        # Configure logging module
        # setup_logging(self._log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

        self.model_path = model_path
        self.run_id = model_dir.name
        self.test_run_id = test_run_id
    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

    def _make_dir(self, path):
        if os.path.exists(path):
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

# helper functions to update config dict with custom cli options
def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config

def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')

def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value

def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
