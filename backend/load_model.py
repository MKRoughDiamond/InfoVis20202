# git submodule update --init --recursive
# git clone --recursive https://github.com/MKRoughDiamond/InfoVis20202.git

import os
import sys
sys.path.append("model")

from model.disvae.utils.modelIO import load_model
from model.utils.datasets import MNIST

def get_model(model_name="betaB"):
    '''
    :param model_name: ["betaB", "betaH", "btcvae", "factor", "VAE"] 중 하나, 모르겠으면 디폴트 사용
    :return: model,
        model.latent_dim: latent dim 크기 (betaB: 10)
        model.img_size: 이미지 크기 ([1, 32, 32])
        self.num_pixels = self.img_size[1] * self.img_size[2]
        self.encoder = encoder(img_size, self.latent_dim)
        self.decoder = decoder(img_size, self.latent_dim)
    '''
    model_list = ["betaB", "betaH", "btcvae", "factor", "VAE"]
    assert model_name in model_list

    directory = os.path.join('model', 'results', model_name+'_mnist')
    return load_model(directory)

def get_dataset():
    return MNIST()


if __name__ == "__main__":
    import pdb
    pdb.set_trace()
    model = get_model()
    dataset = get_dataset()
    print('end')
