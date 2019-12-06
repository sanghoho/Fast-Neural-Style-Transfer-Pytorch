import torchvision.transforms as transforms
from core import style_transfer_basic as st

style_url = "http://www.bloter.net/wp-content/uploads/2016/08/13239928_1604199256575494_4289308691415234194_n-765x510.jpg"
content_url = "http://mblogthumb1.phinf.naver.net/MjAxODAzMzBfNDYg/MDAxNTIyMzg0MjU0MzYz.3bcKRqTsu-4mfxA7ZPk6qxd9QcXDVk1Gs4ro7CrM2dwg.3IFoOxPwwIfl1KYk0diAvb0jpp7WEbKbUk1ZjMKueGYg.JPEG.rigel0380/29572519_1020316934781941_6757553864936979144_n.jpg?type=w800"

style_image = st.get_image_from_url(style_url)
content_image = st.get_image_from_url(content_url)

transfer = st.StyleTransfer(style_image, content_image, 256)
output = transfer.run().cpu().squeeze(0)
im = transforms.ToPILImage()(output).convert("RGB")
im.save('python.jpg')