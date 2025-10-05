import torch
import torch.nn as nn
import tiktoken
import matplotlib.pyplot as plt
from attention import MultiHeadAttention


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])
        self.trfs = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.norm_layer = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, x):
        '''
        Args:
            x : [Batch_size, context_length]
        Returns:
            logits: [Batch_size, context_length, vocab_size]
        '''
        _, context_length = x.shape
        embedded = self.token_emb(x) + self.pos_emb(torch.arange(context_length, device=x.device))
        x = self.dropout(embedded)
        x = self.norm_layer(self.trfs(x))
        return self.out_head(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.emd_dim = cfg["emb_dim"]
        self.norm_layer1 = LayerNorm(emb_dim=self.emd_dim)
        self.norm_layer2 = LayerNorm(emb_dim=self.emd_dim)
        self.attention = MultiHeadAttention(d_in=self.emd_dim,
                                            d_out=self.emd_dim, 
                                            context_length=cfg["context_length"],
                                            dropout=cfg["drop_rate"],
                                            num_heads=cfg["n_heads"],
                                            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg=cfg)
        self.dropout_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm_layer1(x)
        x = self.attention(x)
        x = shortcut + self.dropout_shortcut(x)

        shortcut = x
        x = self.norm_layer2(x)
        x = self.ff(x)
        return shortcut + self.dropout_shortcut(x)


class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))  # [1, embedded]
        self.shift = nn.Parameter(torch.zeros(emb_dim))  
    
    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)   # [B, embedded]
        return self.scale * norm_x + self.shift  # [B, embedded]
    

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))
    

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )

    def forward(self, x):
        return self.layers(x)  # 只是中间嵌入了一层，维度没有发生改变，增加了若干可训练参数，可以提升泛化性能


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]  # 把idx截断为context_size的大小
        with torch.no_grad():
            logits = model(idx_cond)  # logits的大小应该为：[B, C, vocab_size]

        logits = logits[:, -1, :]  # 取最后一个context的输出，大小为[B, vocab_size]
        prabs = torch.softmax(logits, dim=-1)  # prabs是求softmax概率
        idx_next = torch.argmax(prabs, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)  # 在[B, C, vocab_size]的C这里进行拼接
    
    return idx


if __name__ == '__main__' :
    GPT_CONFIG_124M = {
        "vocab_size" : 50527,
        "context_length" : 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }

    tokenizer = tiktoken.get_encoding("gpt2")
    batch = torch.tensor([[6109, 3626, 6100, 345], [6109, 1110, 6622, 257]])

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)

    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # 额外增加一维
    
    model.eval()
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M["context_length"]
    )

    print(tokenizer.decode(out.squeeze(0).tolist()))
