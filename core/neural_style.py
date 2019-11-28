import argparse
import os
import sys
import time
import re

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx

from ..module.utils import Utils
from ..module.transformer_net import TransformerNet
from ..module.vgg import Vgg16


class HyperParameter:
    def __init__(self, command, cuda, param_dict):
        self.cuda = cuda
        self.command = command
        
        if self.cuda and not torch.cuda.is_available():
            print("ERROR: cuda is not available, try running on CPU")
            sys.exit(1)

        if command == "train":
            self.set_train_parameter(param_dict)
        else:
            self.set_eval_parameter(param_dict)
    
    def set_train_parameter(self, param_dict):
        # Required
        self.dataset = param_dict["dataset"]
        self.style_image = param_dict["style_image"]
        self.save_model_dir = param_dict["save_model_dir"]

        # Option
        self.checkpoint_model_dir = os.path.join(self.save_model_dir, "checkpoint")
        self.transfer_learning = param_dict.get("transfer_learning", 0)
        self.epochs = param_dict.get("epochs", 2)
        self.batch_size = param_dict.get("batch_size", 4)
        self.image_size = param_dict.get("image_size", 256)
        self.style_size = param_dict.get("style_size") # None
        self.seed = param_dict.get("seed", 42)
        self.content_weight = param_dict.get("content_weight", 1e5)
        self.style_weight = param_dict.get("style_weight", 1e10)
        self.lr = param_dict.get("lr", 1e-3)
        self.log_interval = param_dict.get("log_interval", 500)
        self.checkpoint_interval = param_dict.get("checkpoint_interval", 2000)
    
    def set_eval_parameter(self, param_dict):
        # Required
        self.content_image = param_dict["content_image"]
        self.output_image = param_dict["output_image"]
        self.model = param_dict["model"]

        # Option
        self.content_scale = param_dict.get("content_scale") # None
        self.export_onnx = param_dict.get("export_onnx")



def check_paths(param):
    try:
        if not os.path.exists(param.save_model_dir):
            os.makedirs(param.save_model_dir)
        if param.checkpoint_model_dir is not None and not (os.path.exists(param.checkpoint_model_dir)):
            os.makedirs(param.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)

def get_value_from_checkpoint(checkpoint_dir):
    latest_checkpoint = os.listdir(checkpoint_dir)[-1]
    checkpoint_val = re.findall(r'\d+', latest_checkpoint)
    checkpoint_val = map(int, checkpoint_val)

    return tuple(checkpoint_val)


def train(param):
    device = torch.device("cuda" if param.cuda else "cpu")

    np.random.seed(param.seed)
    torch.manual_seed(param.seed)

    transform = transforms.Compose([
        transforms.Resize(param.image_size),
        transforms.CenterCrop(param.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder(param.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=param.batch_size)

    transformer = TransformerNet().to(device)
    optimizer = Adam(transformer.parameters(), param.lr)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False).to(device)
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style = Utils.load_image(param.style_image, size=param.style_size)
    style = style_transform(style)
    style = style.repeat(param.batch_size, 1, 1, 1).to(device)

    features_style = vgg(Utils.normalize_batch(style))
    gram_style = [Utils.gram_matrix(y) for y in features_style]

    transfer_learning = bool(param.transfer_learning)

    if transfer_learning: 
        model_dir = args.save_model_dir
        checkpoint_dir = os.path.join(model_dir, "checkpoint")
        check_epoch, check_batch_id = get_value_from_checkpoint(checkpoint_dir)

        ckpt_model_path = os.path.join(args.checkpoint_model_dir, f"ckpt_epoch_{check_epoch}_batch_id_{check_batch_id}.pth")
        checkpoint = torch.load(ckpt_model_path, map_location=device)
        transformer.load_state_dict(checkpoint)
        transformer.to(device)

        transfer_learning_epoch = check_epoch + 1
    else:
        check_epoch, check_batch_id = 0, -1
        transfer_learning_epoch = 0

    for e in range(transfer_learning_epoch, param.epochs):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

            x = x.to(device)
            y = transformer(x)

            y = Utils.normalize_batch(y)
            x = Utils.normalize_batch(x)

            features_y = vgg(y)
            features_x = vgg(x)

            content_loss = param.content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = Utils.gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= param.style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            if (batch_id + 1) % param.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(mesg)

            if param.checkpoint_model_dir is not None and (batch_id + 1) % param.checkpoint_interval == 0:
                transformer.eval().cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
                ckpt_model_path = os.path.join(param.checkpoint_model_dir, ckpt_model_filename)
                torch.save(transformer.state_dict(), ckpt_model_path)
                transformer.to(device).train()

    # save model
    transformer.eval().cpu()
    save_model_filename = "epoch_" + str(param.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
        param.content_weight) + "_" + str(param.style_weight) + ".model"
    save_model_path = os.path.join(param.save_model_dir, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


def stylize(param):
    device = torch.device("cuda" if param.cuda else "cpu")

    content_image = Utils.load_image(param.content_image, scale=param.content_scale)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    if param.model.endswith(".onnx"):
        output = stylize_onnx_caffe2(content_image, param)
    else:
        with torch.no_grad():
            style_model = TransformerNet()
            state_dict = torch.load(param.model)
            # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
            for k in list(state_dict.keys()):
                if re.search(r'in\d+\.running_(mean|var)$', k):
                    del state_dict[k]
            style_model.load_state_dict(state_dict)
            style_model.to(device)
            if param.export_onnx:
                assert param.export_onnx.endswith(".onnx"), "Export model file should end with .onnx"
                output = torch.onnx._export(style_model, content_image, param.export_onnx).cpu()
            else:
                output = style_model(content_image).cpu()
    Utils.save_image(param.output_image, output[0])


def stylize_onnx_caffe2(content_image, param):
    """
    Read ONNX model and run it using Caffe2
    """

    assert not param.export_onnx

    import onnx
    import onnx_caffe2.backend

    model = onnx.load(param.model)

    prepared_backend = onnx_caffe2.backend.prepare(model, device='CUDA' if param.cuda else 'CPU')
    inp = {model.graph.input[0].name: content_image.numpy()}
    c2_out = prepared_backend.run(inp)[0]

    return torch.from_numpy(c2_out)
