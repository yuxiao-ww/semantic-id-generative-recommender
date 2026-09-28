"""
S3a: 纯 AutoEncoder (RQ-VAE 的前身，不带量化)

结构: 384 -> Encoder -> 32维 latent -> Decoder -> 384
目的: 验证 encoder/decoder 能重建语义向量，拿到"无量化"的重建质量上界。
      S3b 会在 latent 和 Decoder 之间插入残差量化。

输入: data/processed/beauty/item_emb.npy  (num_items+1, 384)
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    384 -> 256 -> 128 -> 32 的 MLP。
    逐层收窄，把语义信息压进 32 维瓶颈。
    """
    def __init__(self, in_dim=384, latent_dim=32, hidden=(256, 128)):
        super().__init__()
        dims = [in_dim] + list(hidden)
        layers = []
        # 1. 逐层 Linear + ReLU
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        # 2. 最后一层压到 latent_dim，不加激活 (latent 要能取任意实数值)
        layers += [nn.Linear(dims[-1], latent_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)          # (B, latent_dim)


class Decoder(nn.Module):
    """
    Encoder 的镜像: 32 -> 128 -> 256 -> 384。
    """
    def __init__(self, out_dim=384, latent_dim=32, hidden=(128, 256)):
        super().__init__()
        dims = [latent_dim] + list(hidden)
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        # 1. 最后一层回到 384，不加激活 (重建目标是原始 embedding，实数)
        layers += [nn.Linear(dims[-1], out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)          # (B, out_dim)


class AutoEncoder(nn.Module):
    """S3a 的完整模型: Encoder + Decoder，中间没有量化。"""
    def __init__(self, in_dim=384, latent_dim=32):
        super().__init__()
        self.encoder = Encoder(in_dim, latent_dim)
        self.decoder = Decoder(in_dim, latent_dim)

    def forward(self, x):
        z = self.encoder(x)         # (B, latent_dim) 连续 latent
        recon = self.decoder(z)     # (B, in_dim) 重建
        return recon, z


"""
在 rqvae.py 里追加 ResidualVQ 层 (放在 AutoEncoder 类后面)。
这是把连续 latent 变成层次化离散 token 的核心组件。
"""

class VectorQuantizer(nn.Module):
    """
    单层向量量化: 维护 K 个码字，把输入替换成最近的码字。
    RQ 会堆叠多个这种层。
    """
    def __init__(self, num_codes=256, code_dim=32, beta=0.25):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.beta = beta                    # commitment loss 权重

        # 1. 码本: (K, code_dim) 可学习参数，每行一个码字
        self.codebook = nn.Embedding(num_codes, code_dim)
        # 2. 均匀初始化: 码字散开，降低初始就塌缩的风险
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)

    def forward(self, z):
        """
        z: (B, code_dim) 待量化的向量 (可能是 latent，也可能是残差)
        return:
          z_q: (B, code_dim) 量化后的向量 (STE 处理过，可回传梯度)
          indices: (B,) 每个样本选中的码字编号 -> 这就是 token
          loss: 标量，codebook loss + commitment loss
        """
        # 1. 算 z 到每个码字的欧氏距离平方
        #    展开 ||z - e||^2 = ||z||^2 - 2 z·e + ||e||^2
        #    (广播算，避免显式两层循环)
        codes = self.codebook.weight               # (K, code_dim)
        # (B,1) + (K,) - 2*(B,K) -> (B,K)
        dist = (z.pow(2).sum(1, keepdim=True)
                - 2 * z @ codes.t()
                + codes.pow(2).sum(1))             # (B, K)

        # 2. 最近码字的编号 = 距离最小的那个 -> 这就是 token
        indices = dist.argmin(dim=1)               # (B,)

        # 3. 取出对应码字作为量化结果
        z_q = self.codebook(indices)               # (B, code_dim)

        # 4. 两个辅助 loss (解决 codebook collapse):
        #    codebook loss: 推码字去靠近 encoder 输出 (sg 停在 z 上，只更新码字)
        codebook_loss = torch.nn.functional.mse_loss(z_q, z.detach())
        #    commitment loss: 推 encoder 输出去靠近码字 (sg 停在码字上，只更新 encoder)
        commit_loss = torch.nn.functional.mse_loss(z, z_q.detach())
        loss = codebook_loss + self.beta * commit_loss

        # 5. STE (straight-through estimator):
        #    前向: z + (z_q - z) = z_q，用的是量化值
        #    反向: (z_q - z).detach() 梯度为 0，梯度 = ∂z 直接流回 encoder
        #    这一行解决了 argmin 不可导的问题
        z_q = z + (z_q - z).detach()

        return z_q, indices, loss


class ResidualVQ(nn.Module):
    """
    残差量化: 堆叠 n_layers 个 VectorQuantizer，层层量化残差。

    z ≈ e_{k1} + e_{k2} + ... + e_{kn}
    Semantic ID = (k1, k2, ..., kn)

    每一层量化的是"上一层没表达完的残差"，不是原始 z。
    """
    def __init__(self, n_layers=3, num_codes=256, code_dim=32, beta=0.25):
        super().__init__()
        # 1. n_layers 个独立码本，每层一个 (码字不共享)
        self.layers = nn.ModuleList([
            VectorQuantizer(num_codes, code_dim, beta) for _ in range(n_layers)
        ])
        self.n_layers = n_layers

    def forward(self, z):
        """
        z: (B, code_dim)
        return:
          z_q: (B, code_dim) 所有层量化结果之和 (最终逼近 z)
          all_indices: (B, n_layers) 每列是一层的 token -> 拼起来就是 Semantic ID
          total_loss: 所有层的量化 loss 之和
        """
        residual = z                               # 1. 初始残差就是 z 本身
        z_q = torch.zeros_like(z)                  # 2. 累加各层量化结果
        indices_list = []
        total_loss = 0.0

        for layer in self.layers:
            # 3. 量化当前残差
            q, idx, loss = layer(residual)
            # 4. 累加到总量化结果
            z_q = z_q + q
            # 5. 更新残差: 减去这一层已经表达的部分
            #    注意用 q.detach()? 不 —— 这里 q 已经过 STE，直接减，
            #    让下一层量化"剩下没表达的部分"
            residual = residual - q
            indices_list.append(idx)
            total_loss = total_loss + loss

        # 6. 把每层 token 拼成 (B, n_layers) -> 每行是一个 item 的 Semantic ID
        all_indices = torch.stack(indices_list, dim=1)
        return z_q, all_indices, total_loss


"""
追加到 rqvae.py: 完整 RQVAE
结构: x -> Encoder -> z -> ResidualVQ -> z_q -> Decoder -> x_hat
     比 S3a 的 AutoEncoder 只多了中间 ResidualVQ 一步。
"""

class RQVAE(nn.Module):
    def __init__(self, in_dim=384, latent_dim=32,
                 n_layers=3, num_codes=256, beta=0.25):
        super().__init__()
        # 1. encoder/decoder 结构和 S3a 完全一样，复用
        self.encoder = Encoder(in_dim, latent_dim)
        self.decoder = Decoder(in_dim, latent_dim)
        # 2. 中间插入残差量化层 (这是相对 AutoEncoder 唯一的新增)
        self.rvq = ResidualVQ(n_layers, num_codes, latent_dim, beta)

    def forward(self, x):
        z = self.encoder(x)                    # (B, latent_dim) 连续 latent
        z_q, indices, vq_loss = self.rvq(z)    # 量化: 得到量化latent、SID、量化loss
        recon = self.decoder(z_q)              # (B, in_dim) 从量化latent重建
        return recon, indices, vq_loss

    @torch.no_grad()
    def get_semantic_ids(self, x):
        """
        推理接口: 只出 Semantic ID，不重建。S4 build SID 时用。
        x: (N, in_dim) -> indices: (N, n_layers)
        """
        z = self.encoder(x)
        _, indices, _ = self.rvq(z)
        return indices
