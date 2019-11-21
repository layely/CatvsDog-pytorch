import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.utils import make_grid

CHECKPOINT_PATH = "checkpoint.tar"
TENSORBOARD_LOG_DIR = "tensorboard_logs"


def accuracy(outs, truth):
    preds = outs >= 0.5
    targets = truth >= 0.5
    # Make both predictions and ground truths have the same shape
    preds = torch.reshape(preds, targets.shape)
    return preds.eq(targets).sum().item() / targets.numel()

def save_checkpoint(model, optimizer, epoch, loss):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss
    }
    torch.save(checkpoint, CHECKPOINT_PATH)

def load_checkpoint(model):
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer_state_dict = checkpoint['optimizer_state_dict']
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, optimizer_state_dict, loss

class TBManager():
    """
        A wrapper around Tensorboard.
    """
    def __init__(self):
        self.writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR, comment='', purge_step=None)

    def add_scalar(self, name, scalar, epoch):
        self.writer.add_scalar(name, scalar, epoch)

    def add_images(self, name, model, arg_images):
        images = arg_images * 255
        grid = make_grid(images)
        self.writer.add_image(name, grid, 0)
        # if model:
            # self.writer.add_graph(model, images)

    def close(self):
        self.writer.close()