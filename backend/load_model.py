# git submodule update --init --recursive
# git clone --recursive https://github.com/MKRoughDiamond/InfoVis20202.git

import os
import sys
sys.path.append("model")
from torchvision.datasets import MNIST
from torchvision import transforms

def get_model(model_name="betaB", weight_path=None):
    '''
    :param model_name: ["betaB", "betaH", "btcvae", "factor", "VAE"] 중 하나, 모르겠으면 디폴트 사용
    :return: model,
        model.latent_dim: latent dim 크기 (betaB: 10)
        model.img_size: 이미지 크기 ([1, 32, 32])
        model.encoder: 모델 인코더
        model.decoder: 모델 디코더
        model.forward(): returns reconstruct, [mu, logvar], latent_sample
    '''
    pretrained_model_list = ["betaB", "betaH", "btcvae", "factor", "VAE"]
    if model_name in pretrained_model_list:
        from model.disvae.utils.modelIO import load_model
        directory = os.path.join('model', 'results', model_name+'_mnist')
        return load_model(directory)

    # user input, model_name: .py 파일 이름, weight_path: weight file 이름
    # ex) model_name: modelA.py, model_path: modelA.pt
    # 모델 파일에는 weight file 경로를 입력받아 학습된 model instance를 반환하는 load_model() 함수가 정의되어 있어야 함
    # 반환된 class instance는 위 docstring에서 정의된 attribute와 function을 가져야 함

    else:
        file_name, file_ext = os.path.splitext(model_name)
        assert file_ext == 'py'
        import importlib
        userModule = importlib.import_module("userModel."+file_name)
        return userModule.load_model(os.path.join('userModel', weight_path))

def get_dataset(model_name):
    pretrained_model_list = ["betaB", "betaH", "btcvae", "factor", "VAE"]

    if model_name in pretrained_model_list:
        return MNIST('model/data/mnist', transform=transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor()
        ]), train=False, download=True)

    else:
        return MNIST('model/data/mnist', train=False, download=True)


if __name__ == "__main__":
    import pdb
    pdb.set_trace()
    model = get_model()
    dataset = get_dataset()
    print('end')
