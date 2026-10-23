import torch
import torch.nn.functional as F

from mlp import BLOCK_SIZE, read_names, build_vocab, build_dataset, split_dataset


def init_params_bn(vocab_size=27, block_size=BLOCK_SIZE, n_embd=10, n_hidden=200, seed=2147483647):
    C = torch.randn((vocab_size, n_embd), generator=torch.manual_seed(seed))
    W1 = torch.randn((block_size * n_embd, n_hidden), generator=torch.manual_seed(seed)) * 0.01
    W2 = torch.randn((n_hidden, vocab_size), generator=torch.manual_seed(seed)) * 0.01
    b2 = torch.zeros((vocab_size,))
    # no b1: batchnorm subtracts the batch mean right after, which cancels any bias

    bngain = torch.ones((1, n_hidden))
    bnbias = torch.zeros((1, n_hidden))
    bnmean_running = torch.zeros((1, n_hidden))  # updated by a running average, not by gradients
    bnvar_running = torch.ones((1, n_hidden))

    params = {
        'C': C, 'W1': W1, 'W2': W2, 'b2': b2,
        'bngain': bngain, 'bnbias': bnbias,
        'bnmean_running': bnmean_running, 'bnvar_running': bnvar_running,
    }
    # only the learnable tensors, and only after scaling (a scaled tensor is not a leaf)
    for name in ('C', 'W1', 'W2', 'b2', 'bngain', 'bnbias'):
        params[name].requires_grad_(True)
    return params


def forward_bn(params, X, Y=None, training=True, momentum=0.001, eps=1e-5):
    emb = params['C'][X]
    flat = emb.view(emb.shape[0], -1)
    hpreact = flat @ params['W1']

    if training:
        # normalize with this batch's statistics and track running estimates for evaluation
        batch_mean = hpreact.mean(0, keepdim=True)
        batch_var = hpreact.var(0, keepdim=True, unbiased=True)
        with torch.no_grad():
            params['bnmean_running'].data = (1 - momentum) * params['bnmean_running'].data + momentum * batch_mean.data
            params['bnvar_running'].data = (1 - momentum) * params['bnvar_running'].data + momentum * batch_var.data
    else:
        batch_mean = params['bnmean_running']
        batch_var = params['bnvar_running']

    hpreact_normalized = (hpreact - batch_mean) / torch.sqrt(batch_var + eps)
    hpreact = params['bngain'] * hpreact_normalized + params['bnbias']

    h = torch.tanh(hpreact)
    logits = h @ params['W2'] + params['b2']

    loss = F.cross_entropy(logits, Y) if Y is not None else None
    return logits, loss


def train_bn(params, Xtr, Ytr, steps=20000, batch_size=400, seed=2147483647, log_every=0):
    losses = []
    torch.manual_seed(seed)
    for step in range(steps):
        idx = torch.randint(0, Xtr.shape[0], (batch_size,))
        _, loss = forward_bn(params, Xtr[idx], Ytr[idx], training=True)

        for p in params.values():
            if p.requires_grad:
                p.grad = None
        loss.backward()

        lr = 0.1 if step < 10000 else 0.01  # lower the learning rate for the second half
        with torch.no_grad():
            for p in params.values():
                if p.requires_grad:  # skips the running statistics
                    p -= lr * p.grad

        losses.append(loss.item())
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"step {step:6d}/{steps} | loss {loss.item():.4f}")
    return losses


@torch.no_grad()
def split_loss_bn(params, X, Y):
    _, loss = forward_bn(params, X, Y, training=False)  # evaluation uses the running statistics
    return loss.item()


if __name__ == "__main__":
    words = read_names()
    stoi, itos = build_vocab(words)
    Xtr, Ytr, Xdev, Ydev, Xte, Yte = split_dataset(words, stoi)

    params = init_params_bn()
    print(f"dataset : {len(words):,} names, vocab {len(stoi)}")
    print(f"examples: train {Xtr.shape[0]:,} | dev {Xdev.shape[0]:,} | test {Xte.shape[0]:,}")
    print(f"params  : {sum(p.nelement() for p in params.values() if p.requires_grad):,} learnable")
    print("-" * 40)

    train_bn(params, Xtr, Ytr, steps=20000, log_every=2000)

    print("-" * 40)
    print(f"train loss: {split_loss_bn(params, Xtr, Ytr):.4f}")
    print(f"dev loss  : {split_loss_bn(params, Xdev, Ydev):.4f}")
    print(f"test loss : {split_loss_bn(params, Xte, Yte):.4f}")
