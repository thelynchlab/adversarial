"""
Various attack methods against inception v3 using differential evolution
Based on: https://github.com/sarathknv/adversarial-examples-pytorch
"""
import torch
from torchvision import datasets, transforms
import numpy as np
from scipy.optimize import differential_evolution
import classify
import pickle
from PIL import Image
import PIL

is_cuda = torch.cuda.is_available()
assert is_cuda
device = torch.device("cuda:0")

# Global variables - do not change during runtime
iters = 600
popsize = 20
input_size = (299, 299)
class_names = ['melanoma', 'nevus']
max_stage = 20


class DifevVars:
    """
    Variables that are changed during the process of differential evolution
    and accessed by the callback functions
    """
    perturb_fn = None
    stage = None
    image = None
    trans_image = None
    adv_image = None
    trans_adv_image = None
    model = None
    prob_orig = None
    pred_orig = None
    prob_adv = None
    pred_adv = None


difev_vars = DifevVars()


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


loader1 = transforms.Compose([
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
])

loader2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_image(filename):
    # Load the image that we modify
    image = Image.open(filename)
    image = loader1(image)
    trans_image = loader2(image)
    trans_image = trans_image.repeat(1, 1, 1, 1).cuda()
    return image, trans_image


def optimize(x):
    global difev_vars
    adv_image = difev_vars.perturb_fn(x)
    trans_adv_image = loader2(adv_image).repeat(1, 1, 1, 1)
    out = difev_vars.model(trans_adv_image)
    prob = softmax(out.data.numpy()[0])

    return prob[difev_vars.pred_orig]


def callback(x, convergence):
    global difev_vars
    difev_vars.adv_image = difev_vars.perturb_fn(x)
    difev_vars.trans_adv_image = loader2(difev_vars.adv_image).repeat(1, 1, 1, 1)
    # inp = Variable(torch.from_numpy(preprocess(adv_img)).float().unsqueeze(0))
    out = difev_vars.model(difev_vars.trans_adv_image)
    difev_vars.prob_adv = softmax(out.data.numpy()[0])
    difev_vars.pred_adv = np.argmax(difev_vars.prob_adv)
    p = difev_vars.prob_adv[difev_vars.pred_adv]
    difev_vars.stage += 1
    # if pred_adv != pred_orig and prob_adv >= 0.9:
    if difev_vars.pred_adv != difev_vars.pred_orig and p > 0.9:
        return True
    if difev_vars.stage > max_stage:
        return True
    else:
        print('Prob [%s]: %f, %d iterations' % (
            class_names[difev_vars.pred_orig], difev_vars.prob_adv[difev_vars.pred_orig], difev_vars.stage))


class PixelAttack:
    """
    Use differential evolution to modify a small number of pixels (self.d pixels)
    """

    def __init__(self):
        self.d = 3
        self.bounds = [(0, input_size[0]), (0, input_size[1]), (0, 255), (0, 255), (0, 255)] * self.d
        self.name = 'pixel'

    @staticmethod
    def perturb(x):
        global difev_vars
        adv_image = np.array(difev_vars.image.copy())

        # calculate pixel locations and values
        pixs = np.array(np.split(x, len(x) / 5)).astype(int)
        loc = (pixs[:, 0], pixs[:, 1])
        val = pixs[:, 2:]
        adv_image[loc] = val
        adv_image = Image.fromarray(adv_image)
        return adv_image


class ColorAttack:
    """
    Change the color balance and try to defeat the classifier
    """

    def __init__(self):
        # v = (0.47,0.53)
        v = (0.9, 1.1)
        self.bounds = [v, v, v]
        self.name = 'color'

    @staticmethod
    def perturb(x):
        global difev_vars
        adv_image = np.array(difev_vars.image.copy())
        # calculate pixel locations and values
        adv_image = adv_image * x
        adv_image[adv_image > 255] = 255
        adv_image = Image.fromarray(adv_image.astype('uint8'))
        return adv_image


class RotationTranslationAttack:
    """
    Translate / Rotate the image to defeat the classifier
    """

    def __init__(self):
        self.bounds = [(0, 360), (0, 50), (0, 50)]  # rotation, x translation, y translation
        self.name = 'rotation'

    @staticmethod
    def perturb(x):
        global difev_vars
        adv_image = difev_vars.image.copy()
        adv_image = adv_image.transform(adv_image.size, Image.AFFINE, (1, 0, x[1], 0, 1, x[2]))
        adv_image = adv_image.rotate(x[0])

        return adv_image


def run_attack(attack, img_path, filename, target, fig_path, save=True):
    global difev_vars
    assert difev_vars.model is not None
    assert target in class_names
    difev_vars.stage = 0
    difev_vars.perturb_fn = attack.perturb

    # load image to perturb
    difev_vars.image, difev_vars.trans_image = load_image(img_path + filename)
    X = difev_vars.model(difev_vars.trans_image)
    difev_vars.prob_orig = softmax(X.data.numpy()[0])
    difev_vars.pred_orig = np.argmax(difev_vars.prob_orig)
    print('Prediction before attack: %s' % (class_names[difev_vars.pred_orig]))
    print('Probability: %f' % (difev_vars.prob_orig[difev_vars.pred_orig]))

    if class_names[difev_vars.pred_orig] == target:
        print('Matches target before attack')
        return 'incorrect class'

    # Run the differential evolution attack
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        result = differential_evolution(optimize, attack.bounds, maxiter=iters, popsize=popsize, tol=1e-5,
                                        callback=callback, workers=5)
        # result = differential_evolution(optimize, attack.bounds, maxiter=iters, popsize=popsize, tol=1e-5,
        # callback=callback)
    adv_image = difev_vars.perturb_fn(result.x)
    trans_adv_image = loader2(adv_image).repeat(1, 1, 1, 1)
    out = difev_vars.model(trans_adv_image)
    prob = softmax(out.data.numpy()[0])

    a = class_names[difev_vars.pred_orig]
    b = class_names[difev_vars.pred_adv]

    if a != b:
        print('Successful attack')
        print('Prob [%s]: %f --> Prob[%s]: %f' % (class_names[difev_vars.pred_orig],
                                                  difev_vars.prob_orig[difev_vars.pred_orig],
                                                  class_names[difev_vars.pred_adv],
                                                  difev_vars.prob_adv[difev_vars.pred_adv]))
        base_name = filename.split('.')[0]
        name_image = fig_path + base_name + '_orig_%.3f' % (difev_vars.prob_orig[difev_vars.pred_orig]) + '.jpg'
        name_adv = fig_path + base_name + '_adv_%.3f' % (difev_vars.prob_adv[difev_vars.pred_adv]) + '.jpg'
        adv_image.save(name_adv, 'jpeg')
        difev_vars.image.save(name_image, 'jpeg')
        if attack.name == 'pixel':
            name_diff = fig_path + base_name + '_diff' + '.jpg'
            diff = PIL.ImageChops.difference(adv_image, difev_vars.image)
            diff.save(name_diff)

        # difev_vars.image.show()
        # adv_image.show()
        return 'success'

    else:
        print('Attack failed')
        return 'failed'


def attack_all(attack, img_path, results_path, fig_path):
    """
    Run attacks on all images in the validation set
    """
    import os
    from shutil import copyfile

    if attack == 'pixel':
        attack = PixelAttack()
    elif attack == 'color':
        attack = ColorAttack()
    elif attack == 'rotation':
        attack = RotationTranslationAttack()
    attack.d = 3
    target = 'nevus'
    # load model to attack
    difev_vars.model, _ = classify.initialize_model('inception', num_classes=2, feature_extract=False,
                                                    use_pretrained=False, load=True)
    difev_vars.model.cuda()
    difev_vars.model.eval()
    results = {}
    if os.path.exists(results_path + os.sep + 'results.pkl'):
        results = pickle.load(open(results_path + 'results.pkl', 'rb'))

    for filename in os.listdir(img_path):
        print(img_path + filename)
        assert (os.path.exists(img_path + filename))
        if filename + os.sep + attack.name in results:
            print('skipping')
            continue
        outcome = run_attack(attack, img_path, filename, target, fig_path=fig_path, save=False)
        # p_best = difev_vars.prob_adv[class_names.index(target)]
        results[filename + os.sep + attack.name] = {'outcome': outcome,
                                                    'orig': difev_vars.prob_orig[difev_vars.pred_orig]}
        # 'adv': p_best}
        if os.path.exists(results_path + 'results.pkl'):
            copyfile(results_path + 'results.pkl', results_path + 'results.old')
        pickle.dump(results, open(results_path + 'results.pkl', 'wb'))


def edit_results():
    assert False
    results_path = '/data/figs/lesions-adversarial/difev/'
    results = pickle.load(open(results_path + 'results.pkl', 'rb'))
    new = {}
    for k, v in results.items():
        if k.find('color') == -1: new[k] = v

    pickle.dump(new, open(results_path + 'results.pkl', 'wb'))


def plot_results():
    import math
    results_path = '/data/figs/lesions-adversarial/difev/'
    # results_path = '/data/figs/lesions-adversarial/fgsm/'
    results = pickle.load(open(results_path + 'results.temp.pkl', 'rb'))

    # select = 'color'
    # select = 'rotation'
    # select = 'fgsm'
    select = 'pixel'

    heatmap = np.zeros((10, 10))

    def bin(a):
        return min(math.floor(a * heatmap.shape[0]), heatmap.shape[0] - 1)

    for k, v in results.items():
        if k.find(select) != -1:
            if v['outcome'] in ['success', 'failed']:
                x = v['orig']
                y = v['adv']
                heatmap[bin(x), bin(y)] += 1.0

    import pylab as pl
    # input_img = np.asarray(input_img).copy()
    heatmap = np.log10(heatmap)
    heatmap[heatmap == -np.inf] = 0
    print(heatmap)
    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list('my', [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    pl.pcolor(heatmap, cmap=cmap)
    pl.colorbar()
    # pl.axis('off')
    pl.gca().set_aspect('equal', adjustable='box')
    labels = '0.0,0.2,0.4,0.6,0.8,1.0'.split(',')
    pl.gca().set_xticklabels(labels, minor=False)
    pl.gca().set_yticklabels(labels, minor=False)
    pl.savefig('/data/figs/lesions-adversarial/difev/' + select + '.eps')
    pl.show()


if __name__ == "__main__":
    attack = 'pixel'
    attack_all(attack, img_path='./melanoma/', results_path='./difev/',
               fig_path='./difev/' + attack + '/')
    attack = 'color'
    attack_all(attack, img_path='./melanoma/', results_path='./difev/',
               fig_path='./difev/' + attack + '_colour_jitter/')
    attack = 'rotation'
    attack_all(attack, img_path='./melanoma/', results_path='./difev/',
               fig_path='./difev/' + attack + '/')
