import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
from dataset import MapDataset
from generator_pix2pix_model import Generator
from discriminator_pix2pix_model import Discriminator 
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image

def train_fn():
    pass

def main():
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    

if __name__ == "__main__":
    main()