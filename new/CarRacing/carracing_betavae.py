import numpy as np
import os.path, sys
from torch.utils.data import Dataset
import torch
from skimage import io



sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from betavae import BetaVAE

INPUT_SHAPE = 96 * 96 * 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CarRacingDataset(Dataset):
    def __init__(self, file="carracing_observations.npz"):
        self.observations = np.load(file)["observations"] / 255.
        self.observations = np.transpose(self.observations, (0, 3, 1, 2)).astype(np.float32)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.observations[idx]

def main():
    carracing_dataset = CarRacingDataset()

    if True:
        bvae = BetaVAE(INPUT_SHAPE)
        bvae.train_model(carracing_dataset, "carracing_bvae")
    else:
        bvae = torch.load("carracing_bvae.pth")

        bvae.eval()
        reconstruction, z_sample, z_mu, z_var = bvae(torch.tensor([carracing_dataset[100]]).to(device))
        io.imsave("car_reconstruction.png", np.transpose(reconstruction.detach().cpu().numpy()[0, :], (1, 2, 0)))

if __name__ == "__main__":
    main()
