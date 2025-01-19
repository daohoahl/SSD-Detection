import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from clearml import Task, Logger  # Import ClearML
from model import SSD300, MultiBoxLoss
from datasets import PascalVOCDataset
from utils import *
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Data parameters
data_folder = './'  # folder with data files
keep_difficult = True  # use objects considered difficult to detect?

# Model parameters
n_classes = len(label_map)  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning parameters
checkpoint = None  # path to model checkpoint, None if none
batch_size = 32  # batch size 
workers = 8  # number of workers for loading data in the DataLoader
print_freq = 200  # print training status every __ batches
lr = 0.001  # learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding
epochs = 3  # number of epochs to train for
cudnn.benchmark = True

# Initialize ClearML Task
task = Task.init(project_name="SSD Training", task_name="SSD300 Training with ClearML")

# Log hyperparameters
task.connect({
    "batch_size": batch_size,
    "learning_rate": lr,
    "momentum": momentum,
    "weight_decay": weight_decay,
    "workers": workers,
    "print_freq": print_freq,
    "grad_clip": grad_clip,
    "epochs": epochs
})

def main():
    """
    Training.
    """
    global start_epoch, label_map, epoch, checkpoint

    # Initialize model or load checkpoint
    if checkpoint is None:
        start_epoch = 0
        model = SSD300(n_classes=n_classes)
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    train_dataset = PascalVOCDataset(data_folder, split='train', keep_difficult=keep_difficult)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)
    
    for epoch in range(start_epoch, epochs):
        train(train_loader=train_loader, model=model, criterion=criterion, optimizer=optimizer, epoch=epoch, task=task)

        save_checkpoint(epoch, model, optimizer)

    # Final checkpoint saving in ClearML
    task.upload_artifact(name='final_model', artifact_object=model)
    task.upload_artifact(name='final_optimizer', artifact_object=optimizer)

def train(train_loader, model, criterion, optimizer, epoch, task):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    :param task: ClearML task object
    """
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    start = time.time()

    # Get ClearML logger
    logger = task.get_logger()

    for i, (images, boxes, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - start)
        images = images.to(device)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        predicted_locs, predicted_scores = model(images)

        loss = criterion(predicted_locs, predicted_scores, boxes, labels)

        optimizer.zero_grad()
        loss.backward()

        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
            # Log metrics to ClearML
            iteration = epoch * len(train_loader) + i
            task.get_logger().report_scalar("Loss", "Training", iteration=iteration, value=loss.item())
            task.get_logger().report_scalar("Time", "Batch Time", iteration=iteration, value=batch_time.val)
            task.get_logger().report_scalar("Time", "Data Time", iteration=iteration, value=data_time.val)

    del predicted_locs, predicted_scores, images, boxes, labels
if __name__ == '__main__':
    main()
