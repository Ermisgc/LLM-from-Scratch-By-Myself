import torch
import torch.nn as nn
from gpt import GPTModel, generate_text_simple
import tiktoken
from dataset import create_dataloader_v1, read_file_as_text


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # 用unsqueeze提升一维
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())
    

def calc_loss_batch(input, target, model, device):
    input = input.to(device)
    target = target.to(device)
    logits = model(input)
    return nn.functional.cross_entropy(torch.flatten(logits, 0, 1), torch.flatten(target, 0, 1))


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if(len(data_loader)) == 0:
        return float("nan")
    elif num_batches == None:
        num_batches = len(data_loader)
    else:
        num_batches = min(len(data_loader), num_batches)

    for i, (input, target) in enumerate(data_loader):
        if i >= num_batches:
            break
        else:
            loss = calc_loss_batch(input=input, target=target, model=model, device=device)
            total_loss += loss

    return total_loss / num_batches


def train_model_simple(model, train_loader, test_loader, optimizer, device, num_epochs, eval_freq, eval_itr, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input, target in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input=input, target=target, model=model, device=device)
            loss.backward()
            optimizer.step()
            tokens_seen += input.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model=model, train_loader=train_loader, val_loader=test_loader, device=device, eval_iter=eval_itr)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}")
        
        generate_and_print_sample(model=model, tokenizer=tokenizer, device=device, start_context=start_context)

    return train_losses, val_losses, track_tokens_seen
        

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model=model, device=device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model=model, device=device, num_batches=eval_iter)
    
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(text=start_context, tokenizer=tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model=model, idx=encoded, max_new_tokens=50, context_size=context_size)
    
    decoded_text = token_ids_to_text(token_ids=token_ids, tokenizer=tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()


if __name__ == "__main__":
    GPT_CONFIG_124M = {
        "vocab_size" : 50527,
        "context_length" : 256,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    tokenizer = tiktoken.get_encoding("gpt2")

    whole_text = read_file_as_text("the-verdict.txt")
    train_ratio = 0.90
    split_idx = int(train_ratio * len(whole_text))

    train_data = whole_text[:split_idx]
    val_data = whole_text[split_idx:]

    train_dataloader = create_dataloader_v1(
        text=train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )
    
    val_dataloader = create_dataloader_v1(
        text=val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    device = torch.device("cuda"
    if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0004,
        weight_decay=0.1
    )

    model.to(device)

    num_epoch = 10
    train_losses, val_losses, tokens_seen = train_model_simple(
        model=model,
        train_loader=train_dataloader,
        test_loader=val_dataloader,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epoch,
        eval_freq=5,
        eval_itr=5,
        start_context="Every effort moves you",
        tokenizer=tokenizer
    )
