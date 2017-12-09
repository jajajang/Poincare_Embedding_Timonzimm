import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        if target in path:
            temp_targ=path.index(target)
            for i in range(temp_targ, len(path) - 1):
                hypernyms.append((noun, path[i]))
            del temp_targ
            

hypernyms = np.array(hypernyms)
lhds, rhds = list(hypernyms[:, 0]), list(hypernyms[:, 1])
targets = set(lhds+rhds)
print(len(targets), 'targets')

print(len(hypernyms), 'target hypernyms')
uniq_hypernyms = np.array(list(targets))
target_index=list(targets).index(target)
word2idx = {val: i for i, val in enumerate(uniq_hypernyms)}
random.shuffle(hypernyms)


def proj(params):
    norm = params.norm(p=2, dim=1).unsqueeze(1)
    for i, normy in enumerate(norm):
	if normy.numpy()>1:
		param[i]=param[i]/(normy+EPS)

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

#want to calculate isometry which moves 0 to v
def move(a,v):
    embeddings = (EMBEDDINGS.cpu()).numpy()
    targ_x, targ_y=embeddings[word2idx[a]]
    z=complex(targ_x,targ_y)
    temp_x, temp_y = embeddings[word2idx[v]]
    dz = -complex(temp_x, temp_y)
    moved = (z + dz) / (z * np.conjugate(dz) + 1)
    x = np.real(moved)
    y = np.imag(moved)
    return [x,y]

def plot(filename):
    embeddings = (EMBEDDINGS.cpu()).numpy()
    targets_ = ['mammal.n.01', 'beagle.n.01', 'canine.n.02', 'german_shepherd.n.01',
               'collie.n.01', 'border_collie.n.01',
               'carnivore.n.01', 'tiger.n.02', 'tiger_cat.n.01', 'domestic_cat.n.01',
               'squirrel.n.01', 'finback.n.01', 'rodent.n.01', 'elk.n.01',
               'homo_sapiens.n.01', 'orangutan.n.01', 'bison.n.01', 'antelope.n.01',
               'even-toed_ungulate.n.01', 'ungulate.n.01', 'elephant.n.01', 'rhinoceros.n.01',
               'odd-toed_ungulate.n.01', 'mustang.n.01', 'liger.n.01', 'lion.n.01', 'cat.n.01', 		'dog.n.01']
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.cla()  # clear things for fresh plot

    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((-1.1, 1.1))

    circle = plt.Circle((0, 0), 1., color='black', fill=False)
    ax.add_artist(circle)

    for n in targets_:
        x, y = move(wn.synset(n),target)
        ax.plot(x, y, 'o', color='y')
        ax.text(x + 0.01, y + 0.01, n, color='b')

    plt.savefig(filename)
    plt.close()

def are_you_hyper(dog, cat):
    big_dog=False
    big_cat=False
    hyper_path_d=dog.hypernym_paths()
    for path in hyper_path_d:
        if cat in path:
            big_dog=True

    hyper_path_c=cat.hypernym_paths()
    for path in hyper_path_c:
        if dog in path:
            big_cat=True
    return big_dog or big_cat

EPOCHS = 100
DIM = 2
START_LR = 0.1
FINAL_LR = 0.0001
NEG = 10

EMBEDDINGS = torch.Tensor(len(uniq_hypernyms), DIM)
nn.init.uniform(EMBEDDINGS, a=-0.001, b=0.001)

bar2 = hypernyms

for epoch in range(EPOCHS):
    filename='epoch_'+str(epoch)+'.png'
    plot(filename)
    for i, (w1, w2) in enumerate(bar2):
        i_w1 = word2idx[w1]
        i_w2 = word2idx[w2]
        u = Variable(EMBEDDINGS[i_w1].unsqueeze(0), requires_grad=True)
        v = Variable(EMBEDDINGS[i_w2].unsqueeze(0), requires_grad=True)
        NEG_SAMPLES_h=[]
        while len(NEG_SAMPLES_h)<NEG:
            pickme=np.random.randint(0, len(uniq_hypernyms))
            if (not are_you_hyper(uniq_hypernyms[pickme], uniq_hypernyms[i_w1])) :
                NEG_SAMPLES_h.append(pickme)
        NEG_SAMPLES = torch.from_numpy(np.array(NEG_SAMPLES_h))
        negs = Variable(EMBEDDINGS[NEG_SAMPLES], requires_grad=True)
        loss =-torch.log(torch.exp(-1 * distance(u, v)) / torch.exp(-1 * distance(u, negs)).sum())
        loss.backward()
        if i%1000==0:
            print str(epoch)+' epoch and '+str(i)+' attempt'
        r = epoch / EPOCHS
        LR = (1 - r) * START_LR + r * FINAL_LR
        EMBEDDINGS[NEG_SAMPLES] -= LR * (((1 - negs.norm(dim=1) ** 2) ** 2) / 4).data.unsqueeze(
            1) * negs.grad.data
        EMBEDDINGS[i_w1] -= LR * (((1 - u.norm() ** 2) ** 2) / 4).data * u.grad.data
        EMBEDDINGS[i_w2] -= LR * (((1 - v.norm() ** 2) ** 2) / 4).data * v.grad.data        
	EMBEDDINGS = proj(EMBEDDINGS)
        u.grad.data.zero_()
        v.grad.data.zero_()
        negs.grad.data.zero_()
