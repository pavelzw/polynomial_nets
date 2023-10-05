import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3, Inception_V3_Weights

import numpy as np
from scipy.stats import entropy


def inception_score(imgs, cuda=True, batch_size=1, resize=True, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print(
                "WARNING: You have a CUDA device, so you should probably set cuda=True"
            )
        dtype = torch.FloatTensor

    # Set up dataloader
    # dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False).type(dtype)
    inception_model.eval()

    # up = nn.functional.interpolate(size=(299, 299), mode='bilinear', align_corners=True).type(dtype)
    def get_pred(x):
        if resize:
            # x = up(x)
            x = nn.functional.interpolate(
                x, size=(299, 299), mode="bilinear", align_corners=True
            ).type(dtype)
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    # for i, batch in enumerate(dataloader, 0):
    # for i, batch in enumerate(imgs):
    for i in range(N // batch_size):
        batch = imgs[batch_size * i : batch_size * (i + 1), :, :, :]
        batch = torch.from_numpy(batch)
        batch = batch.type(dtype)
        batchv = Variable(batch)
        # batchv = batchv.unsqueeze(0)
        batch_size_i = batch.size()[0]
        preds[i * batch_size : i * batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits) : (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)
