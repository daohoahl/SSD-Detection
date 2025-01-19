import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from clearml import Task  # Import ClearML
from model import SSD300, MultiBoxLoss
from datasets import PascalVOCDataset
from utils import *

# Data parameters
data_folder = './'  # folder with data files
keep_difficult = True  # use objects considered difficult to detect?

# Model parameters
n_classes = len(label_map)  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning parameters
checkpoint = None  # path to model checkpoint, None if none
batch_size = 8  # batch size
iterations = 120000  # number of iterations to train
workers = 2  # number of workers for loading data in the DataLoader
print_freq = 200  # print training status every __ batches
lr = 1e-3  # learning rate
decay_lr_at = [80000, 100000]  # decay learning rate after these many iterations
decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding

cudnn.benchmark = True

# Initialize ClearML Task
task = Task.init(project_name="SSD Training", task_name="SSD300 Training with ClearML")

configuration_data = {'num_classes': 91, 'image_size': 512, 'model_type': 'ssd', 'ssd_backbone': 'resnet50',
                      'mask_predictor_hidden_layer': 256}
configuration_data = task.connect_configuration(configuration_data)
# Log hyperparameters
task.connect({
    "batch_size": batch_size,
    "learning_rate": lr,
    "momentum": momentum,
    "weight_decay": weight_decay,
    "iterations": iterations,
    "decay_lr_at": decay_lr_at,
    "workers": workers,
    "print_freq": print_freq,
    "decay_lr_to": decay_lr_to,
    "grad_clip": grad_clip
})


def main():
    """
    Training.
    """
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at

    # Initialize model or load checkpoint
    if checkpoint is None:
        start_epoch = 0
        model = SSD300(n_classes=n_classes)
        # Initialize the optimizer
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # Custom dataloaders
    train_dataset = PascalVOCDataset(data_folder,
                                     split='train',
                                     keep_difficult=keep_difficult)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)

    # Calculate total number of epochs
    epochs = 10
    decay_lr_at_epochs = [it // (len(train_dataset) // batch_size) for it in decay_lr_at]

    # Training loop
    for epoch in range(start_epoch, epochs):
        if epoch in decay_lr_at_epochs:
            adjust_learning_rate(optimizer, decay_lr_to)

        train(train_loader=train_loader, model=model, criterion=criterion, optimizer=optimizer, epoch=epoch)

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer)

        # Upload model to ClearML
        task.upload_artifact(name=f"ssd_checkpoint_epoch_{epoch}", artifact_object=f"checkpoint_ssd_epoch_{epoch}.pth")


def train(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.
    """
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    start = time.time()

    for i, (images, boxes, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(device)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop
        predicted_locs, predicted_scores = model(images)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)

        # Backward prop
        optimizer.zero_grad()
        loss.backward()

        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        optimizer.step()

        # Update metrics
        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Log metrics to ClearML
        iteration = epoch * len(train_loader) + i
        task.get_logger().report_scalar("Loss", "Training", iteration=iteration, value=loss.item())
        task.get_logger().report_scalar("Time", "Batch Time", iteration=iteration, value=batch_time.val)
        task.get_logger().report_scalar("Time", "Data Time", iteration=iteration, value=data_time.val)


        if i % print_freq == 0:
            print(f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t"
                  f"Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                  f"Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t"
                  f"Loss {losses.val:.4f} ({losses.avg:.4f})")

    del predicted_locs, predicted_scores, images, boxes, labels


if __name__ == '__main__':
    main()
