import tiktoken
import os
import urllib.request
import re  # 正则表达式
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(text)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            output_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(output_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(text, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")  # BPE分词
    dataset = GPTDatasetV1(text, tokenizer=tokenizer, max_length=max_length, stride=stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )


def embed(inputs, vocal_size=50257, output_dim=256):
    '''
    Args:
        inputs: 经过tokenizer后的tensor,大小应该是: [batch_size, context_length]
        context_length: 组成单个上下文的语境长度
        vocal_size: 词典的token数量
        output_dim: 嵌入向量的维度
    Return:
        返回嵌入向量，大小是[batch_size, context_length, output_dim]
    '''
    context_length = inputs.shape[-1]

    # token嵌入：token_id-> 1*n tensor的映射，层大小为[vocab_size, output_dim]
    # 本质上是将每一个tensor的token转化为{token, output_dim}的二元组，扩大了一维
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim) 
    token_embedded = token_embedding_layer(inputs)  # 这里的token嵌入应该只利用索引，非矩阵操作，返回的是[batch_size, context_length, output_dim]

    # 绝对位置嵌入：其实是给长度为context_length的每个位置都给一个值，层大小为[context_length, output_dim]
    # 即每一个位置都对应一个output_dim的一维向量
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)  # 大小为[context_length, output_dim]
    pos_embedded = pos_embedding_layer(torch.arange(context_length))

    return token_embedded + pos_embedded


def read_file_as_text(filename):
    with open("the-verdict.txt", "r", encoding='utf-8') as f:
        raw_text = f.read()    
    return raw_text


class SimpleTokenizer:
    def __init__(self, vocab: dict):
        '''
        A Simple Tokenizer
        Args:
            vocab (dict : str-> int) : the encoded vocabulary
        '''
        self.encode_vocab = vocab
        self.decode_vocab = {i:s for s, i in self.encode_vocab.items()}

    @classmethod
    def from_raw_text(cls, raw_text: str):
        '''
        build encode-decode dict through whole string
        '''
        preprocessed = cls.__str_split__(raw_text)
        all_words = sorted(set(preprocessed))  # set用于去重，sorted按字典序排序
        all_words.extend(["<|endoftext|>", "<|unk|>"])
        vocab = {token:integer for integer,token in enumerate(all_words)}  # 以token为key, integer为value构建词汇表vocab
        return cls(vocab)

    def encode(self, text : str) -> list[int]:
        # Tokenizer
        tokens = [token if token in self.encode_vocab else "<|unk|>" for token in self.__str_split__(text)]
        return [self.encode_vocab[token] for token in tokens]

    def decode(self, ids: list[int]) -> str:
        text = " ".join([self.decode_vocab[id] for id in ids])  # 用' '将每个token连接在一起
        return re.sub(r'\s+([,.?!"()\'])', r'\1', text)  # 查找空格+标点符号的匹配字符，然后sub掉
    
    @staticmethod
    def __str_split__(text : str) -> list[str]:
        #  文本分词
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)  # 分词
        return [item for item in preprocessed if item.strip()]   # 移除字符串首尾空格


if __name__ == '__main__':
    if not os.path.exists('the-verdict.txt'):   # 如果数据集文件不存在，就去官网下载这个文件，存在本地路径下
        url = ("https://raw.githubusercontent.com/rasbt/"
            "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
            "the-verdict.txt")
        file_path = 'the-verdict.txt'
        urllib.request.urlretrieve(url=url, filename=file_path)

    with open("the-verdict.txt", "r", encoding='utf-8') as f:
        raw_text = f.read()

    context_length = 4
    dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=context_length, stride=context_length, shuffle=False)
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)

    # 嵌入层，绝对位置嵌入
    vocab_size = 50257
    output_dim = 256
    torch.manual_seed(123)
    
    # token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)  # 嵌入层是token_id-> 1*n tensor的映射，由随机值组成
    # pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    # token_embedded = token_embedding_layer(inputs)  # token_embedded:[8, 4, 256]
    # pos_embedded = pos_embedding_layer(torch.arange(context_length)) # [4, 256]
    # input_embedded = token_embedded + pos_embedded  # [8, 4, 256]
    # print(input_embedded.shape)
    print(embed(inputs=inputs, vocal_size=vocab_size, output_dim=output_dim).shape)
