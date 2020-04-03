from __future__ import division
from __future__ import print_function
import copy
import math
import os
import pickle
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from PIL import ImageFile
from torch.autograd import Variable
from torchvision import datasets, models, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]


def get_norm():
    return norm_mean, norm_std


DATA = '/data/isic/analysed/'
INVALID_CACHE = False
# Detect if we have a GPU available
is_cuda = torch.cuda.is_available()
assert is_cuda
device = torch.device("cuda:1")


def np_to_tensor(X, type=torch.FloatTensor, requires_grad=False):
    v = Variable(torch.from_numpy(X), requires_grad=requires_grad).type(type)
    if is_cuda: v = v.cuda()
    return v


def cache_run(f, cache_file=None, cache=True, args=None):
    """
    Wrap another function cache the return value
    Probably better to use np.save for large np arrays
    """
    argstring = None
    # if args is not None:
    #    argstring = '.'.join([str(k)+'.'+str(v) for k,v in args.items()])
    if cache_file is None:
        cache_file = f.__name__
        if argstring is not None: cache_file += '.' + argstring
        cache_file += '.pkl'

    v = None
    cache_file = DATA + cache_file
    if (not INVALID_CACHE) and cache and os.path.exists(cache_file):
        # print('loading cache...',cache_file)
        v = pickle.load(open(cache_file, 'rb'))
    else:
        print('cache_run calculating...')
        if args is not None:
            v = f(args)
        else:
            v = f()

        pickle.dump(v, open(cache_file, 'wb'))

    assert len(v) > 0
    return v


# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

BASE = './isic/'
data_dir = BASE + 'dataset-classify'
# model_dir = '/data/isic/analysed/classify/' # to train with magnus model
model_dir = BASE + 'analysed/classify/'  # to train with CA retrained jitter
# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
num_classes = 2
batch_size = 8
num_epochs = 30  # was 15
model_name = 'inception'  # CA
# model_name = 'vgg'
# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False


def create_image_folder():
    import os, json
    from shutil import copyfile
    classes = ['melanoma', 'nevus']
    files = {c: [] for c in classes}

    in_desc = BASE + 'descriptions/'
    in_img = BASE + 'images/'
    out_dir = BASE + 'dataset-classify/'
    train_frac = 0.8

    for filename in os.listdir(in_desc):
        if filename[0] == '.': continue
        data = json.load(open(in_desc + filename))
        if 'diagnosis' in data['meta']['clinical']:
            diag = data['meta']['clinical']['diagnosis']
        else:
            diag = None
        src = in_img + filename + '.jpeg'
        if diag in classes and os.path.exists(src):
            files[diag].append(filename)

    for c in classes:
        random.shuffle(files[c])
        n_train = int(train_frac * float(len(files[c])))

        for phase in ['train', 'val']:
            if not os.path.exists(out_dir + phase): os.mkdir(out_dir + phase)
            if not os.path.exists(out_dir + phase + '/' + c): os.mkdir(out_dir + phase + '/' + c)

            if phase == 'train':
                v = files[c][:n_train]
            elif phase == 'val':
                v = files[c][n_train:]

            for filename in v:
                print(filename, c, phase)
                dest = out_dir + phase + '/' + c + '/' + filename + '.jpeg'
                if not os.path.exists(dest):
                    copyfile(in_img + filename + '.jpeg', dest)


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True, load=False):
    # Initialize these variables which will be set in this if statement. Each of th
    #   variables is model specific.
    model_ft = None
    input_size = 0
    assert not feature_extract
    if load: use_pretrained = False

    if model_name == "resnet":
        """ Resnet18
        """
        assert False
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        assert False
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        # model_ft = models.vgg11_bn(pretrained=use_pretrained)
        model_ft = models.vgg19(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        assert False
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        assert False
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    if load:
        filename = model_dir + model_name + '.pth'
        assert os.path.exists(filename)
        print('loading model', filename)
        model_ft.load_state_dict(torch.load(filename))

    return model_ft, input_size


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    i = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

                filename = model_dir + model_name + '.pth'
                print('saving', filename)
                torch.save(model.state_dict(), filename)

            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def create_dataloaders(input_size, batch_size=batch_size, only_val_transforms=False):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),  # magnus added
            transforms.RandomRotation(180),  # magnus added
            transforms.ColorJitter(hue=0.4),  # callum added for the second round of training with hue changes
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            transforms.Normalize(norm_mean, norm_std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            transforms.Normalize(norm_mean, norm_std)
        ]),
    }
    if only_val_transforms: data_transforms['train'] = data_transforms['val']

    # Create training and validation datasets
    # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in
        ['train', 'val']}
    return dataloaders_dict


def run_training(load=True, save=True):
    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True, load=load)

    # Data augmentation and normalization for training # Just normalization for valid   ation

    # Create training and validation datasets
    # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloaders
    # dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=   batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
    dataloaders_dict = create_dataloaders(input_size)

    # to fine label-> index: image_datasets['train'].class_to_idx
    # 0=melanoma, 1= nevus

    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    # print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                # print("\t",name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad:
                pass
                # print("\t",name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs,
                                 is_inception=(model_name == "inception"))


# ===============================================================================

# def image_loader(image_name,input_size):
#    """
#    Load image and apply normalization as required for feeding to network
#    """
#    loader = transforms.Compose([
#	transforms.Resize(input_size),
#	transforms.CenterCrop(input_size),
#	transforms.ToTensor(),
#	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#	])
#    image = Image.open(image_name)
#    image = loader(image).float()
#    image = Variable(image, requires_grad=True)
##image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
#    return image.cuda()  #assumes that you're using GPU
#
#
# def display_image_loader(image_name,input_size):
#    """
#    Load image and crop - to make figures, no other normalization
#    """
#    loader = transforms.Compose([
#	transforms.Resize(input_size),
#	transforms.CenterCrop(input_size),
#	])
#    image = Image.open(image_name)
#    image = loader(image)
#    return image

def create_masks(img_size, mask_size, step):
    img_size = int(img_size)
    n_masks = int(math.ceil(img_size / step) ** 2)
    masks = np.ones((n_masks, 3, img_size, img_size))
    mask_dims = []

    i = 0
    for y1 in range(0, img_size, step):
        for x1 in range(0, img_size, step):
            x2 = min(x1 + mask_size, img_size)
            y2 = min(y1 + mask_size, img_size)
            # print(x1,y1,x2,y2)
            masks[i, :, x1:x2, y1:y2] = 0
            mask_dims.append((x1, x2, y1, y2))
            i += 1

    # a = Image.fromarray(masks[15,1,:,:]*255)
    # a.show()
    return masks, mask_dims


def analyse_model():
    """
    Run the trained inception model on all of the validation images and save results
    """
    import common
    out = open('/data/isic/analysed/inception_results.csv', 'w')
    model_ft, img_size = initialize_model('inception', num_classes=2, feature_extract=False, use_pretrained=False,
                                          load=True)
    model_ft.cuda()
    model_ft.eval()
    paths = [('/data/isic/dataset-classify/val/nevus/', 'nevus'),
             ('/data/isic/dataset-classify/val/melanoma/', 'melanoma')]

    correct = 0.0
    incorrect = 0.0

    for path in paths:
        for filename in os.listdir(path[0]):
            img = common.image_loader(path[0] + filename, img_size)
            img = img.repeat(1, 1, 1, 1)
            X = torch.softmax(model_ft(img), 1)
            X = X.detach().cpu().numpy()[0][0]
            if X >= 0.5 and path[1] == 'melanoma':
                correct += 1
            elif X < 0.5 and path[1] == 'nevus':
                correct += 1
            else:
                incorrect += 1
            v = [filename, path[1], X, correct / (correct + incorrect)]
            v = [str(a) for a in v]
            print(v)
            out.write(','.join(v) + '\n')


def analyse_image_mask(model_ft, img_size, path, filename, modes=['heatmap', 'best']):
    """
    Occlude areas of hte image using tiled masks and assess which are most important for classification
    Save the region that causes the most differentiation
    """
    max_batch = 50
    # filename =  '/data/isic/dataset-classify/val/melanoma/ISIC_0009934.jpeg'
    # filename =  '/data/isic/dataset-classify/val/melanoma/ISIC_0000284.jpeg'
    # filename =  '/data/isic/dataset-classify/val/melanoma/ISIC_0029740.jpeg'
    out_dir = '/data/isic/analysed/heatmap/'
    out_dir2 = '/data/isic/analysed/topdisc/'
    if not os.path.exists(out_dir): os.mkdir(out_dir)
    if not os.path.exists(out_dir2): os.mkdir(out_dir2)

    # model_ft, img_size = initialize_model('inception', num_classes=2, feature_extract=False, use_pretrained=False,load=True)
    # model_ft.cuda()
    # model_ft.eval()

    def f1():
        mask_size = int(img_size / 5)
        step = int(mask_size / 4)
        masks, mask_dims = create_masks(img_size, mask_size, step)
        masks = np_to_tensor(masks)
        n_masks = masks.shape[0]
        results = np.zeros(n_masks)

        for start in range(0, n_masks, max_batch):
            batch_size = min(max_batch, n_masks - start)
            current_masks = masks[start:start + batch_size, :, :, :]
            img = image_loader(path + filename, img_size)
            img = img.repeat(batch_size, 1, 1, 1)
            img = img * current_masks
            X = torch.softmax(model_ft(img), 1)
            X = X.detach().cpu().numpy()
            results[start:start + batch_size] = X[:, 1]
            # print(X.shape,current_masks.shape)

        return results, masks.cpu().numpy(), mask_dims

    results, masks, mask_dims = cache_run(f1, cache=False)
    img_size = masks.shape[2]
    #
    heat_map = np.zeros((img_size, img_size))
    total = np.zeros((img_size, img_size))

    for i, x in enumerate(results):
        inverse_mask = np.logical_not(masks[i, 0, :, :].copy())
        heat_map += x * inverse_mask
        total += inverse_mask

    input_img = display_image_loader(path + filename, img_size)
    input_array = np.asarray(input_img).copy()
    out_file = filename.split('.')[0]

    # save the top discrim region
    if 'best' in modes:
        best = np.argmax(results)
        x1, x2, y1, y2 = mask_dims[best]
        best_img = Image.fromarray(input_array[x1:x2, y1:y2])
        best_img.save(out_dir2 + out_file + '.jpg')

    # save the heatmap
    if 'heatmap' in modes:
        heat_map = heat_map / total
        # input_img = display_image_loader(path+filename,img_size)
        input_img.save(out_dir + out_file + '_img.jpg')
        import pylab as pl
        # input_img = np.asarray(input_img).copy()
        pl.pcolor(heat_map, cmap='jet')
        pl.axis('off')
        pl.gca().set_aspect('equal', adjustable='box')
        pl.savefig(out_dir + out_file + '_heatmap.jpg')


def all_analyse_image_mask():
    path = '/data/isic/dataset-classify/val/melanoma/'
    model_ft, img_size = initialize_model('inception', num_classes=2, feature_extract=False, use_pretrained=False,
                                          load=True)
    model_ft.cuda()
    model_ft.eval()

    for filename in os.listdir(path):
        if not filename.endswith('jpeg'): continue
        print(path, filename)
        analyse_image_mask(model_ft, img_size, path, filename, modes=['best'])


if __name__ == "__main__":
    run_training()
