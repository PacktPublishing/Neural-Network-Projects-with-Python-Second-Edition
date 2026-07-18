import argparse
import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import ViTForImageClassification
import numpy as np
import albumentations as A
import cv2

import data

BATCH_SIZE = 256
MAX_EPOCHES = 200
LEARNING_RATE = 1e-4
LOADER_WORKERS = 4
IMAGE_SIZE = 224


@torch.no_grad()
def validate(model: nn.Module, dataset: data.Dataset,
             device: torch.device) -> float:
    loader = DataLoader(dataset, BATCH_SIZE, shuffle=True,
                        num_workers=LOADER_WORKERS)
    loss = nn.CrossEntropyLoss()
    losses = []
    for batch_x, batch_y in tqdm.tqdm(loader, desc="Validation"):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        out_t = model(batch_x).logits
        loss_t = loss(out_t, batch_y)
        losses.append(loss_t.detach().item())
    return np.mean(losses)


if __name__ == '__main__':
    cv2.setNumThreads(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="Random seed",
                        default=data.TRAIN_TEST_DEFAULT_SEED)
    parser.add_argument("-n", "--name", required=True,
                        help="Name of the run")
    parser.add_argument("-g", "--gpu", default="cpu",
                        help="GPU device, default=cpu")
    args = parser.parse_args()
    device = torch.device(args.gpu)
    torch.manual_seed(args.seed)

    augment = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussianBlur(p=0.2),
        A.Rotate(angle_range=(-30, 30), p=0.5),
    ])

    # data loading
    images_path = data.get_images_path()
    images_classes = data.get_classes(images_path)
    samples = data.get_sorted_samples(images_path)
    train_samples, val_samples, test_samples = data.split_train_val_test(
        samples, args.seed, shuffle=True)
    rescale_transform = A.Compose([
        A.LongestMaxSize(IMAGE_SIZE),
        A.PadIfNeeded(IMAGE_SIZE, IMAGE_SIZE, fill=data.FILL_COLOR),
    ])
    train_dataset = data.ImagesDataset(images_classes, train_samples,
                                       rescale=rescale_transform, augment=augment)
    val_dataset = data.ImagesDataset(images_classes, val_samples,
                                     rescale=rescale_transform)
    test_dataset = data.ImagesDataset(images_classes, test_samples,
                                      rescale=rescale_transform)
    print(f"Train={len(train_dataset)}, val={len(val_dataset)}, test={len(val_dataset)}")

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True,
                              num_workers=LOADER_WORKERS)

    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    for param in model.parameters():
        param.requires_grad = False
    output_layer = nn.Linear(model.classifier.in_features, len(images_classes))
    model.classifier = output_layer
    model = model.to(device)

    # model
    optimizer = optim.Adam(output_layer.parameters(), lr=LEARNING_RATE)
    loss = nn.CrossEntropyLoss()
    print(model)
    best_val_loss = None

    try:
        with SummaryWriter(log_dir=f"runs/{args.name}") as writer:
            for epoch in range(1, MAX_EPOCHES+1):
                losses = []
                for batch_x, batch_y in tqdm.tqdm(train_loader, desc="Training"):
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    optimizer.zero_grad()

                    out_t = model(batch_x).logits
                    loss_t = loss(out_t, batch_y)
                    loss_t.backward()
                    optimizer.step()
                    losses.append(loss_t.detach().item())
                train_loss = np.mean(losses)
                val_loss = validate(model, val_dataset, device)
                print(f"Epoch {epoch}: train_loss={train_loss:.5f}, "
                      f"val_loss={val_loss:.5f}")
                writer.add_scalar("loss", train_loss, epoch)
                writer.add_scalar("loss-val", val_loss, epoch)
                if best_val_loss is None or best_val_loss > val_loss:
                    print(f"Model improved, saving")
                    torch.save(output_layer.state_dict(), args.name + "_best.model")
                    best_val_loss = val_loss
    except KeyboardInterrupt:
        print("Interrupting, checking on test dataset...")
    finally:
        test_loss = validate(model, test_dataset, device)
        print(f"Test loss={test_loss:.5f}")
