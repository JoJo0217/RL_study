import os
import logging
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        handlers = [logging.StreamHandler(os.sys.stdout)]
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
            handlers.append(logging.FileHandler(
                os.path.join(self.log_dir, 'log.txt')))
            self.writer = SummaryWriter(log_dir=self.log_dir)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(message)s',
            handlers=handlers
        )

    def log(self, global_step, is_tensor_board=True, ** kwargs):
        msg = f"global_step: {global_step},"
        for k, v in kwargs.items():
            msg += f"{k}: {v}, "
            if is_tensor_board:
                self.writer.add_scalar(k, v, global_step)
                self.writer.flush()
        logging.info(msg)
