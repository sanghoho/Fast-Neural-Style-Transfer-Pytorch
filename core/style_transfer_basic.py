import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import copy, os, pkg_resources

class StyleTransfer:
    def __init__(self, style_img, content_img, imgsize, output_path):
        super(StyleTransfer, self).__init__()
        self.loader = transforms.Compose([
            transforms.Scale(imgsize),       # 한 축을 128로 조절하고
            transforms.CenterCrop(imgsize),  # square를 한 후,
            transforms.ToTensor(),       # Tensor로 바꾸고 (0~1로 자동으로 normalize)                           
        ])        
        self.unloader = transforms.ToPILImage()        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.style_img = self.image_loader(style_img)
        self.content_img = self.image_loader(content_img)
        self.output_path = output_path
        torch.cuda.empty_cache()

        
    def run(self, num_steps=300):
        print(f"{self.style_img.size()} : {self.content_img.size()}")
        print(f"{self.style_img.size()} : {self.content_img.size()}")
        print(f"{self.style_img.size()} : {self.content_img.size()}")

        assert self.style_img.size() == self.content_img.size(), \
            "we need to import style and content images of the same size"
        root = __name__.split(".")[0]
        vgg_model = pkg_resources.resource_filename(root, "models/vgg/vgg19-dcbb9e9d.pth")

        ## Model Import
        vgg19 = models.vgg19()
        vgg19.load_state_dict(torch.load(vgg_model))
        cnn = vgg19.features.to(self.device).eval()

        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)    

        # desired depth layers to compute style/content losses :
        self.content_layers_default = ['conv_4']
        self.style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

        input_img = self.content_img.clone()
        output = self.run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std, self.content_img, self.style_img, input_img, num_steps)

        self.save_image(output)

        torch.cuda.empty_cache()
        # print("hello")
        return output

    def save_image(self, tensor):
        image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)      # remove the fake batch dimension
        torchvision.utils.save_image(image, self.output_path)

    def image_loader(self, image_name):
        image = Image.open(image_name)
        # fake batch dimension required to fit network's input dimensions
        image = self.loader(image.convert('RGB')).unsqueeze(0)
        return image.to(self.device, torch.float)


    ##### LOSS Function

    # Content Loss
    class ContentLoss(nn.Module):

        def __init__(self, target,):
            super(StyleTransfer.ContentLoss, self).__init__()
            # we 'detach' the target content from the tree used
            # to dynamically compute the gradient: this is a stated value,
            # not a variable. Otherwise the forward method of the criterion
            # will throw an error.
            self.target = target.detach()

        def forward(self, input):
            self.loss = F.mse_loss(input, self.target)
            return input

    # Style Loss 
    class StyleLoss(nn.Module):

        def __init__(self, target_feature):
            super(StyleTransfer.StyleLoss, self).__init__()
            self.target = self.gram_matrix(target_feature).detach()

        def forward(self, input):
            G = self.gram_matrix(input)
            self.loss = F.mse_loss(G, self.target)
            return input

        def gram_matrix(self, input):
            a, b, c, d = input.size()  # a=batch size(=1)
            features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

            G = torch.mm(features, features.t())  # compute the gram product

            # we 'normalize' the values of the gram matrix
            # by dividing by the number of element in each feature maps.
            return G.div(a * b * c * d)   

    #### Model Impoert
    class Normalization(nn.Module):
        def __init__(self, mean, std):
            super(StyleTransfer.Normalization, self).__init__()
            # .view the mean and std to make them [C x 1 x 1] so that they can
            # directly work with image Tensor of shape [B x C x H x W].
            # B is batch size. C is number of channels. H is height and W is width.
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

        def forward(self, img):
            # normalize img
            return (img - self.mean) / self.std    



    def get_style_model_and_losses(self, cnn, normalization_mean, normalization_std,
                                style_img, content_img,
                                content_layers=['conv_4'],
                                style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
        cnn = copy.deepcopy(cnn)

        # normalization module
        normalization = self.Normalization(normalization_mean, normalization_std).to(self.device)

        # just in order to have an iterable access to or list of content/syle
        # losses
        content_losses = []
        style_losses = []

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(normalization)

        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(content_img).detach()
                content_loss = self.ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).detach()
                style_loss = self.StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], self.ContentLoss) or isinstance(model[i], self.StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses


    ### Gradient Descent
    def get_input_optimizer(self, input_img):
        # this line to show that input is a parameter that requires a gradient
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        return optimizer

    def run_style_transfer(self, cnn, normalization_mean, normalization_std,
                        content_img, style_img, input_img, num_steps=300,
                        style_weight=1000000, content_weight=1):
        """Run the style transfer."""
        print('Building the style transfer model..')
        model, style_losses, content_losses = self.get_style_model_and_losses(cnn,
            normalization_mean, normalization_std, style_img, content_img)
        optimizer = self.get_input_optimizer(input_img)

        print('Optimizing..')
        run = [0]
        while run[0] <= num_steps:

            def closure():
                # correct the values of updated input image
                input_img.data.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()

                return style_score + content_score

            optimizer.step(closure)
        # a last correction...
        input_img.data.clamp_(0, 1)

        return input_img



def get_image_from_url(img_url):
    response = requests.get(img_url)
    image = BytesIO(response.content)
    return image

def image_to_bytes(image):
    byteIO = BytesIO()
    image.save(byteIO, format='PNG')
    byteArr = byteIO.getvalue()
    return byteArr
