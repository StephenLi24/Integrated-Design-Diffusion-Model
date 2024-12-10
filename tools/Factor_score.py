import torch
from random import randint, random
from tqdm import tqdm

def factor_score(model, rounds: int, dataset_size: int, noise_dim: int, device: torch.device):
    """
    Diffusion Model disentanglement metric using pure PyTorch.
    
    Args:
        model: A diffusion model with `generate` and `forward_encoder` methods.
        rounds: Number of evaluation rounds.
        dataset_size: Number of samples per round.
        noise_dim: Dimensionality of the latent noise (epsilon).
        device: PyTorch device (e.g., 'cpu' or 'cuda').

    Returns:
        Disentanglement score (accuracy of identifying fixed latent dimension).
    """
    correct_guesses = 0
    for _ in tqdm(range(rounds)):
        # 1. 固定潜变量维度
        y = randint(0, noise_dim - 1)  # 随机选择一个维度
        fixed_value = random() * 6 - 3  # 随机固定值
        epsilon = torch.randn((dataset_size, noise_dim), device=device)  # 随机噪声
        epsilon[:, y] = fixed_value  # 固定选择的维度

        # 2. 使用扩散模型生成数据
        x_gen = model.generate(epsilon, timesteps=1000)  # 扩散模型生成图像
        
        # 3. 编码生成数据回潜变量空间
        epsilon_hat = model.forward_encoder(x_gen)['epsilon']  # 编码器重建噪声

        # 4. 标准化重建噪声
        epsilon_hat_norm = epsilon / epsilon_hat.std(dim=0, keepdim=True)  # 标准化
        var_epsilon_hat_norm = epsilon_hat_norm.var(dim=0)  # 计算每个维度的方差
        
        # 5. 找出方差最小的维度
        y_hat = var_epsilon_hat_norm.argmin().item()  # 找出方差最小的维度
        correct_guesses += int(y == y_hat)  # 判断是否与固定维度一致

    # 6. 计算解耦得分
    return correct_guesses / rounds
