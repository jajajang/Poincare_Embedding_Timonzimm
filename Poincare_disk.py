import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm_notebook
from torch.autograd import Variable
from nltk.corpus import wordnet as wn

SEED = 42
EPS = 1e-5
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

target = wn.synset('mammal.n.01')
words = wn.words()

nouns = list(wn.all_synsets('n'))

print(len(nouns), 'nouns')

hypernyms = []
for noun in nouns:
    paths = noun.hypernym_paths()
    for path in paths:
        for i in range(0, len(path) - 1):
            hypernyms.append((noun, path[i]))
            
            
hypernyms = np.array(list(set(hypernyms)))
uniq_hypernyms = np.array(list(set([e for tup in hypernyms for e in tup])))

word2idx = {val: i for i, val in enumerate(uniq_hypernyms)}
random.shuffle(hypernyms)

print(len(hypernyms), 'hypernyms')
lhds, rhds = hypernyms[:, 0], hypernyms[:, 1]
targets = set(lhd for i, lhd in enumerate(lhds) if rhds[i] == target)

def proj(params):
    norm = params.norm(p=2, dim=1).unsqueeze(1)
    norm[norm < 1] = 1 + EPS
    params = params.div(norm) - EPS
    return params


def arcosh(x):
    return torch.log(x + torch.sqrt(x ** 2 - 1))


def distance(u, v):
    uu = u.norm(dim=1) ** 2
    vv = v.norm(dim=1) ** 2
    uv = u.mm(v.t())
    alpha = 1 - uu
    alpha = alpha.clamp(min=EPS)
    beta = 1 - vv
    beta = beta.clamp(min=EPS)

    gamma = 1 + 2 * (uu - 2 * uv + vv) / (alpha * beta)
    gamma = gamma.clamp(min=1 + EPS)

    return arcosh(gamma)


def plot(filename):
    embeddings = (EMBEDDINGS.cpu()).numpy()

    if len(targets) + 1 > 30:
        targets = random.sample(targets, 30-1)
    targets.append(target)
    embeddings = (EMBEDDINGS.cpu()).numpy()

    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.cla()  # clear things for fresh plot

    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((-1.1, 1.1))

    circle = plt.Circle((0, 0), 1., color='black', fill=False)
    ax.add_artist(circle)

    for n in targets:
        x, y = embeddings[word2idx[n]]
        ax.plot(x, y, 'o', color='y')
        ax.text(x + 0.01, y + 0.01, n, color='b')
    plt.savefig(filename)


EPOCHS = 20
DIM = 2
START_LR = 0.1
FINAL_LR = 0.0001
NEG = 10

EMBEDDINGS = torch.Tensor(len(uniq_hypernyms), DIM).cuda()
nn.init.uniform(EMBEDDINGS, a=-0.001, b=0.001)

NEG_SAMPLES = torch.from_numpy(np.random.randint(0, len(uniq_hypernyms), size=(EPOCHS, len(hypernyms), NEG))).cuda()

for epoch in range(EPOCHS):
    filename='epoch_'+str(epoch)+'.png'
    plot(filename)
    bar2 = hypernyms
    for i, (w1, w2) in enumerate(bar2):
        i_w1 = word2idx[w1]
        i_w2 = word2idx[w2]
        u = Variable(EMBEDDINGS[i_w1].unsqueeze(0), requires_grad=True)
        v = Variable(EMBEDDINGS[i_w2].unsqueeze(0), requires_grad=True)
        negs = Variable(EMBEDDINGS[NEG_SAMPLES[epoch, i]], requires_grad=True)

        loss = torch.exp(-1 * distance(u, v)) / torch.exp(-1 * distance(u, negs)).sum()
        bar2.set_postfix(loss=loss.data[0, 0])
        loss.backward()
        if i/1000==0:
            print i
        r = epoch / EPOCHS
        LR = (1 - r) * START_LR + r * FINAL_LR
        EMBEDDINGS[NEG_SAMPLES[epoch, i]] -= LR * (((1 - negs.norm(dim=1) ** 2) ** 2) / 4).data.unsqueeze(
            1) * negs.grad.data
        EMBEDDINGS[i_w1] -= LR * (((1 - u.norm() ** 2) ** 2) / 4).data * u.grad.data
        EMBEDDINGS[i_w2] -= LR * (((1 - v.norm() ** 2) ** 2) / 4).data * v.grad.data

        EMBEDDINGS = proj(EMBEDDINGS)
