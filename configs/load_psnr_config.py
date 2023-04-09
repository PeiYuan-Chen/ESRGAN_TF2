import yaml
import sys

sys.path.append('../')


class Config:
    __instance = None

    def __init__(self):
        if Config.__instance is not None:
            raise Exception("Config class is a singleton!")
        else:
            with open("configs/config_psnr.yaml", "r") as f:
                self.config_data = yaml.safe_load(f)

            # Data settings
            data = self.config_data['data']
            self.Use_TFRecord = data['Use_TFRecord']
            self.TFRecord_file = data['TFRecord_file']
            self.train_lr_dir = data['train_lr_dir']
            self.train_hr_dir = data['train_hr_dir']
            self.val_lr_dir = data['val_lr_dir']
            self.val_hr_dir = data['val_hr_dir']
            self.eval_lr_dir = data['eval_lr_dir']
            self.eval_hr_dir = data['eval_hr_dir']
            self.test_lr_dir = data['test_lr_dir']
            self.test_hr_dir = data['test_hr_dir']
            self.cache_dir = data['cache_dir']
            self.lr_size = data['lr_size']
            self.hr_size = data['hr_size']
            self.upscale_factor = data['upscale_factor']
            self.channels = data['channels']
            self.batch_size = data['batch_size']

            # Training settings
            training = self.config_data['training']
            self.iterations = training['iterations']
            self.save_every = training['save_every']
            self.init_learning_rate = training['init_learning_rate']
            self.lr_decay_rate = training['lr_decay_rate']
            self.lr_decay_iter_list = training['lr_decay_iter_list']

            # Model checkpoints
            checkpoint = self.config_data['checkpoint']
            self.latest_checkpoint_dir = checkpoint['latest_checkpoint_dir']
            self.best_weights_file = checkpoint['best_weights_file']
            self.history_file = checkpoint['history_file']

            # Logs
            logs = self.config_data['logs']
            self.eval_log_file = logs['eval_log_file']

            Config.__instance = self

    @staticmethod
    def getInstance():
        if Config.__instance is None:
            Config()
        return Config.__instance


cfg = Config.getInstance()
