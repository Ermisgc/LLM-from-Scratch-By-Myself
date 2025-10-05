import torch
import torch.nn as nn


def simple_self_attention(inputs):
    '''
    简单无训练权重的自注意力机制
    Args:
        inputs : 执行embed嵌入操作后的tensor, 大小为[context_length, output_dim]
    Returns:
        context_vec: 上下文向量，大小也为[context_length, output_dim]
    '''
    # Step1. 求score，torch.dot或者矩阵乘法
    attn_scores = inputs @ inputs.T  # [context_length, output_dim] * [output_dim, context_length] -- > [context_length, context_length]
    # Step2. 求weight, softmax
    attn_weight = torch.softmax(attn_scores, dim=1)  # 这里的-1等于最后一维，即列，对一行内的每个列取值进行的softmax，weight的size也为[context_length, context_length]
    # Step3. 求context_vec: sum(weight * input)
    context_vec = attn_weight @ inputs  # [context_length, output_dim]
    return context_vec


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads,  qkv_bias=False):
        '''
        多头注意力机制
        Args:
            d_in: token转化后的嵌入向量的维度
            d_out: 多头注意力机制输出的上下文向量的维度总和，假设头数为num_heads，单个头输出维度为x，那么d_out = num_heads * x
            context_length: 一个上下文的token长度
            dropout: dropout层的drop比例，一部分被屏蔽，另一部分相应增强
            num_heads: 头数，它应该刚好能被d_out整除
            qkv_bias: 注意力权重是否要带偏置值
        '''
        super().__init__()
        assert(d_out % num_heads == 0)

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)  # [d_in, d_out]
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # 相当于不同头的权重，在原本输出的基础上，加一个线性层进行权重变换
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, inputs):
        '''
        Args:
            inputs: 尺寸应该是[B, context_length, d_in]
        Returns:
            返回值tensor尺寸为[B, context_length, d_out]，这里的d_out是多头输出的总和 
        '''
        # inputs的尺寸一般是[B, d_in]，这里的d_in等于嵌入层的output_dim
        b, num_tokens, d_in = inputs.shape
        # Step1. 求所有向量通过权重矩阵变换后的的key, value
        keys   = self.W_key(inputs)  # [B, num_tokens,  d_out]
        values = self.W_value(inputs) 
        querys = self.W_query(inputs)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)  # 把keys拆分为[b, num_head, context_length, head_dim]
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        querys = querys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # Step2. 求每个查询向量与key的点积，然后用掩码制造序列因果顺序
        attn_score = querys @ keys.transpose(2, 3)  # [B, num_head, context_length, context_length]
        attn_score.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

        # Step3. 求softmax作为归一化的概率，这里采用嵌入维度的平方根以提升反向传播梯度
        attn_weight = torch.softmax(attn_score / keys.shape[-1] ** 0.5, dim=-1)  # 嵌入维度的平方根能够减缓提升反向传播的梯度，提高训练效率
        attn_weight = self.dropout(attn_weight)  # 用dropout屏蔽部分，并增强部分

        # Step4. 求上下文向量，通过概率与value加权求得
        context_vecs = attn_weight @ values  # [B, num_head, context_length, head_dim]
        context_vecs = context_vecs.transpose(1, 2)  # [B, context_length, num_head, head_dim]
        context_vecs = context_vecs.contiguous().view(b, num_tokens, self.d_out)  # 组合多头的输出,[B, context_length, d_out]

        # Step5. 多头权重修正
        return self.out_proj(context_vecs)  # [B, context_length, d_out]



if __name__ == '__main__':
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55]]
    )  # 大小为 [context_length, output_dim]，或者说[context_length, d_in]

    batch = torch.stack((inputs, inputs), dim=0)   # [B, context_length, d_in]

    d_in = inputs.shape[-1]
    d_out = 2
    torch.manual_seed(123)

    sa = MultiHeadAttention(d_in=d_in, d_out=d_out, context_length=batch.shape[1], dropout=0.0, num_heads=2)
    context_vecs = sa(batch)
    print(context_vecs)
    print(context_vecs.shape)