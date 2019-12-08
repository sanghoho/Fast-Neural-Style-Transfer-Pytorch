from ..core import neural_style as ns
from ..core import style_transfer_basic as st

import glob
import torch
import torchvision.transforms as transforms

def get_image(content_img, style_img, custom_style=None, output_path="output.jpg"):

    if custom_style != None:
        style_img = st.get_image_from_url(custom_style)
        content_img = st.get_image_from_url(content_img)

        transfer = st.StyleTransfer(style_img, content_img, 256, output_path)
        output = transfer.run().cpu().squeeze(0)
        # im = transforms.ToPILImage()(output).convert("RGB")
        return transforms.ToPILImage()(output).convert("RGB") 

    else:

        content_img = st.get_image_from_url(content_img)
        param_dict = {
            "content_image": content_img,
            "output_image": output_path,
            "model": glob.glob(style_img + "/*.*")[0]
        }

        cuda = [ 1 if torch.cuda.is_available() else 0][0]

        # cuda = 0 # True

        param = ns.HyperParameter("eval", cuda, param_dict)
        return ns.stylize(param)

