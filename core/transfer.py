from ..core import neural_style as ns
from ..core import style_transfer_basic as st

import torchvision.transforms as transforms
import glob

def get_image(content_img, style_img, custom_style=None):

    if custom_style != None:
        style_img = st.get_image_from_url(custom_style)
        content_img = st.get_image_from_url(content_img)

        transfer = st.StyleTransfer(style_img, content_img, 256)
        output = transfer.run().cpu().squeeze(0)
        # im = transforms.ToPILImage()(output).convert("RGB")
        return transforms.ToPILImage()(output).convert("RGB") 

    else:

        content_img = st.get_image_from_url(content_img)
        param_dict = {
            "content_image": content_img,
            "output_image": "output.jpg",
            "model": glob.glob(style_img + "/*.*")[0]
        }

        cuda = 0 # True

        param = ns.HyperParameter("eval", cuda, param_dict)
        return ns.stylize(param)

