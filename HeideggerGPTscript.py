import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for prediction?
max_iters = 200
eval_interval = 300
lerning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

torch.manual_seed(1337)

with open("SeinUndZeit.txt",'r',encoding='utf-8') as f:
    text = f.read()

print("Length of dataset in characters : ", len(text))

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda j: ''.join([itos[c] for c in j])

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)

# split up the data into train and validation sets
n =  int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]



# %%
block_size = 8
train_data[:block_size+1]

# %%
x = train_data[:block_size]
y = train_data[1:block_size+1]

for t in range(block_size):
    predictors = x[:t+1]
    target = y[t]
    print(f"input : {predictors} target : {target}")




# %%
torch.manual_seed(100)
batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context size for predictions?

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x,y
xb, yb = get_batch('train')
print('inputs : ')
print(xb.shape)
print(xb)
print('targets :')
print(yb.shape)
print(yb)


for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} the target : {target}")




# %% [markdown]
# 

# %%
import torch
import torch.nn as nn 
from torch.nn import functional as F 
torch.manual_seed(1337)

class BigrammLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) #(B, T, C) 
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


n = BigrammLanguageModel(vocab_size)
logits, loss = n(xb,yb)
print(logits.shape)
print(loss)

# %%
idx = torch.zeros((1,1), dtype=torch.long)
print('\n Random Text')
print(decode(n.generate(idx, max_new_tokens = 400)[0].tolist()))


# %%
# create a Pytorch optimizer
optimizer = torch.optim.Adam(n.parameters(),lr=1e-3)

# %%
batch_size  = 32 
for steps in range(1000):
    # sample a batch of data
    xb , yb = get_batch('train')

    # evaluate the lass
    logits, loss = n(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())



# %%
idx = torch.zeros((1,1), dtype=torch.long)

print('\n After simplest possible optimization')
print(decode(n.generate(idx, max_new_tokens = 400)[0].tolist()))

# %% [markdown]
# 


