import math
from torchvision import datasets, transforms
import numpy as np
# import classify
from PIL import Image
import os
from torch.autograd import Variable

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]


class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


denorm_image_tensor = Denormalize(norm_mean, norm_std)


def save_denorm_image(image_tensor, filename):
    dn = denorm_image_tensor(image_tensor)
    # dn = image_tensor
    dn = (255 * dn.squeeze().detach().cpu().numpy()).astype(np.uint8)
    dn = np.transpose(dn, (1, 2, 0))
    img = Image.fromarray(dn)
    img.save(filename)


def image_transform(image, input_size):
    loader = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    return image


def image_loader(image_name, input_size, cuda=True):
    """
    Load image and apply normalization as required for feeding to network
    """
    #    loader = transforms.Compose([
    #	transforms.Resize(input_size),
    #	transforms.CenterCrop(input_size),
    #	transforms.ToTensor(),
    #	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #	])
    #    image = loader(image).float()
    #    image = Variable(image, requires_grad=True)
    # image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    image = Image.open(image_name)
    image = image_transform(image, input_size)
    if cuda: image = image.cuda()
    return image  # assumes that you're using GPU


def display_image_loader(image_name, input_size):
    """
    Load image and crop - to make figures, no other normalization
    """
    loader = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
    ])
    image = Image.open(image_name)
    image = loader(image)
    return image


def melanoma_examples():
    path = '/data/isic/dataset-classify/train/melanoma/'
    files = '0000434.jpeg,0011267.jpeg,0014076.jpeg,0026360.jpeg,0026420.jpeg,0026754.jpeg,0026236.jpeg,0026158.jpeg,0028029.jpeg,0031941.jpeg'.split(
        ',')
    files = ['ISIC_' + x for x in files]
    for x in files:
        if not os.path.exists(path + x):
            print('NOT FOUND:' + x)
            assert False
    return path, files


def make_image_grid():
    """
    Tiles image contents of a directory 
    """
    path = '/data/isic/analysed/topdisc/'
    img_files = []
    w = None;
    h = None

    for filename in os.listdir(path):
        if filename == 'combined.jpg': continue
        if not filename.endswith('jpg'): continue
        img_files.append(filename)
        if w is None:
            img = Image.open(path + filename)
            w = img.size[0];
            h = img.size[1]
            assert w == h

    n = math.ceil(math.sqrt(len(img_files)))

    background = Image.new('RGB', (n * w, n * h), (255, 255, 255))

    x = 0
    y = 0
    for i, filename in enumerate(img_files):
        print(filename)
        img = Image.open(path + filename)
        background.paste(img, (x, y))
        x += w
        if x + w > w * n:
            x = 0
            y += h

    background.save(path + 'combined.jpg')
