import pytest
import torch
import torch.nn as nn

from src.nn_mnist.model import NeuralNet
from src.nn_mnist.data import get_data_loaders


@pytest.fixture
def model_instance():
    """テスト用のNeuralNetモデルインスタンスを作成するフィクスチャ"""
    input_size = 784
    hidden_size = 500
    num_classes = 10
    # デバイス設定はテストでは不要（CPUで十分）
    return NeuralNet(input_size, hidden_size, num_classes)

# ユニットテスト 1: モデルの出力形状の確認 
def test_model_output_shape(model_instance):
    """モデルにダミー入力を与え、出力の形状が正しいか検証する"""
  
    dummy_input = torch.randn(10, 784)
    
    output = model_instance(dummy_input)
    
    assert output.shape == (10, 10)
    
# ユニットテスト 2: データローダーの機能確認 
def test_data_loader_setup():
    """データローダーが正しく初期化され、期待されるバッチサイズを返すか検証する"""
    
    test_batch_size = 64
    train_loader, test_loader = get_data_loaders(batch_size=test_batch_size)
    
    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert isinstance(test_loader, torch.utils.data.DataLoader)
    
    assert train_loader.batch_size == test_batch_size
    
    images, labels = next(iter(train_loader))
  
    assert images.shape == (test_batch_size, 1, 28, 28)
    assert labels.shape == (test_batch_size,)