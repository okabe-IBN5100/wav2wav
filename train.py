import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from data import AudioDataset
from model import UnetGenerator, ConditionalDiscriminator
from criterion import GeneratorLoss, DiscriminatorLoss
from utils import Logger, weights_init
from config import *

device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

generator = UnetGenerator().to(device)
discriminator = ConditionalDiscriminator().to(device)
generator.apply(weights_init)
discriminator.apply(weights_init)

g_optimizer = torch.optim.Adam(generator.parameters(), lr=GLEARNING_RATE, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=DLEARNING_RATE, betas=(0.5, 0.999))

g_criterion = GeneratorLoss(alpha=100)
d_criterion = DiscriminatorLoss()

source_voice = 'SF2'
target_voice = 'TM1'

n_batches = 2000
dataset = AudioDataset(data_path, source_voice, target_voice)
sampler = RandomSampler(dataset, num_samples=BATCH_SIZE * n_batches)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)

GLoss = []
DLoss = []
logger = Logger()
for epoch in range(EPOCHS):
    ge_loss=0.
    de_loss=0.
    print(f"EPOCH {epoch} / {EPOCHS}")
    start = time.time()

    for batch_id, (x, real) in enumerate(dataloader):
        x = x.to(device)
        real = real.to(device)

        # Generator`s loss
        fake = generator(x)
        fake_pred = discriminator(fake, x)
        g_loss = g_criterion(fake, real, fake_pred)

        # Discriminator`s loss
        fake = generator(x).detach()
        fake_pred = discriminator(fake, x)
        real_pred = discriminator(real, x)
        d_loss = d_criterion(fake_pred, real_pred)

        # Generator`s params update
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # Discriminator`s params update
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        # add batch losses
        ge_loss += g_loss.item()
        de_loss += d_loss.item()

        GLoss.append(g_loss.item())
        DLoss.append(d_loss.item())

        if batch_id % 50 == 0:
            print("[Epoch %d/%d] [Batch %d/%d] [G loss: %.3f] [D loss: %.3f]" % (epoch+1, EPOCHS, batch_id, len(dataloader), g_loss, d_loss))

    # obttain per epoch losses
    g_loss = ge_loss/len(dataloader)
    d_loss = de_loss/len(dataloader)
    # count timeframe
    end = time.time()
    tm = (end - start)
    # logger.add_scalar('generator_loss', g_loss, epoch+1)
    # logger.add_scalar('discriminator_loss', d_loss, epoch+1)
    logger.save_weights(generator.state_dict(), 'generator')
    logger.save_weights(discriminator.state_dict(), 'discriminator')
logger.close()
print('End of training process!')

plt.plot(GLoss)
plt.plot(DLoss)
plt.show()