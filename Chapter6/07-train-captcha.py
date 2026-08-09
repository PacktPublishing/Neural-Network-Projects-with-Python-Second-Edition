import argparse
import tqdm
import torch
import typing as tt
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2

import data_captcha
import models_captcha

BATCH_SIZE = 128
MAX_EPOCHES = 1000
LEARNING_RATE = 1e-4
LOADER_WORKERS = 4


def get_ocr_predictions(
        ocr_model: models_captcha.OCRNetwork,
        input_t: torch.Tensor
) -> tt.List[str]:
    res = ['' for _ in range(input_t.shape[0])]
    for y_idx in range(input_t.shape[-1] // data_captcha.CELL_WIDTH):
        y_pos = y_idx * data_captcha.CELL_WIDTH
        img_t = input_t[:, :, :, y_pos:(y_pos+data_captcha.CELL_WIDTH)]
        char_probs_t = ocr_model(img_t)
        char_indices_t = torch.argmax(char_probs_t, dim=-1)
        res = [
            s + data_captcha.CAPTCHA_CHARS[idx]
            for s, idx in zip(res, char_indices_t.cpu().tolist())
        ]
    return res


def get_ocr_matches(preds: tt.List[str], labels: tt.List[str]) -> tt.Tuple[int, int, int]:
    res_total, res_matched = 0, 0
    res_full_match = 0
    for p, l in zip(preds, labels):
        if p == l:
            res_full_match += 1
        for c1, c2 in zip(p, l):
            res_total += 1
            if c1 == c2:
                res_matched += 1
    return res_total, res_matched, res_full_match


@torch.no_grad()
def validate(model: models_captcha.CaptchaUNet,
             ocr_model: models_captcha.OCRNetwork,
             dataset: data_captcha.Dataset,
             device: torch.device) -> tt.Tuple[float, float, float]:
    loader = DataLoader(dataset, BATCH_SIZE, shuffle=True,
                        num_workers=LOADER_WORKERS)
    loss = nn.MSELoss()
    losses = []
    chars_total, chars_matched = 0, 0
    captchas_matched = 0
    for batch_x, (batch_y, labels) in tqdm.tqdm(loader, desc="Validation"):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        out_t = model(batch_x)
        loss_t = loss(out_t, batch_y)
        losses.append(loss_t.detach().item())

        ocr_res = get_ocr_predictions(ocr_model, out_t)
        ocr_total, ocr_matched, full_matched = get_ocr_matches(ocr_res, labels)
        chars_total += ocr_total
        chars_matched += ocr_matched
        captchas_matched += full_matched
    return np.mean(losses), chars_matched / chars_total, captchas_matched / len(dataset)


if __name__ == '__main__':
    cv2.setNumThreads(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="Random seed",
                        default=data_captcha.TRAIN_TEST_DEFAULT_SEED)
    parser.add_argument("-n", "--name", required=True,
                        help="Name of the run")
    parser.add_argument("-g", "--gpu", default="cpu",
                        help="GPU device, default=cpu")
    parser.add_argument("--ocr", required=True,
                        help="Path to trained OCR network")
    args = parser.parse_args()
    device = torch.device(args.gpu)
    torch.manual_seed(args.seed)

    # data loading
    images_path = data_captcha.get_images_path()
    samples = data_captcha.get_samples(images_path)
    train_samples, val_samples, test_samples = data_captcha.split_train_val_test(
        samples, args.seed, shuffle=True)
    train_dataset = data_captcha.CaptchaDataset(train_samples)
    val_dataset = data_captcha.CaptchaDataset(val_samples)
    test_dataset = data_captcha.CaptchaDataset(test_samples)
    print(f"Train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True,
                              num_workers=LOADER_WORKERS)

    # model
    model = models_captcha.CaptchaUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss = nn.MSELoss()
    print(model)
    best_val_loss = None

    # load OCR network
    ocr_model = models_captcha.OCRNetwork(len(data_captcha.CAPTCHA_CHARS)).to(device)
    ocr_model.eval()
    weights = torch.load(args.ocr, map_location=device, weights_only=True)
    ocr_model.load_state_dict(weights)

    try:
        with SummaryWriter(log_dir=f"runs/{args.name}") as writer:
            for epoch in range(1, MAX_EPOCHES+1):
                losses = []
                for batch_x, (batch_y, _) in tqdm.tqdm(train_loader, desc="Training"):
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    optimizer.zero_grad()

                    out_t = model(batch_x)
                    loss_t = loss(out_t, batch_y)
                    loss_t.backward()
                    optimizer.step()
                    losses.append(loss_t.detach().item())
                train_loss = np.mean(losses)
                val_loss, ocr_correct, captchas_correct = validate(
                    model, ocr_model, val_dataset, device)
                print(f"Epoch {epoch}: train_loss={train_loss:.5f}, "
                      f"val_loss={val_loss:.5f}, ocr_correct={ocr_correct:.5f}, "
                      f"captchas_correct={captchas_correct:.5f}")
                writer.add_scalar("loss", train_loss, epoch)
                writer.add_scalar("loss-val", val_loss, epoch)
                writer.add_scalar("ocr-correct", ocr_correct, epoch)
                writer.add_scalar("captchas-correct", captchas_correct, epoch)
                if best_val_loss is None or best_val_loss > val_loss:
                    print(f"Model improved, saving")
                    torch.save(model.state_dict(), args.name + "_best.model")
                    best_val_loss = val_loss
    except KeyboardInterrupt:
        print("Interrupting...")
