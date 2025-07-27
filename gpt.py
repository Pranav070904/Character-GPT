import torch
import torch.nn as nn
import torch.nn.functional as F


#wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
#--------------------------------------------------------------------------------
block_size = 256
batch_size = 64
epochs = 5000
lr = 3e-4
device = 'cuda' if torch.cuda.is_available() else "cpu"
print(device)
eval_iter = 200
eval_interval = 300
n_embed = 384
n_head = 6
n_layer = 6
dropout = .2
torch.manual_seed(42)

#-----------------------------------------------------------------------------------------

with open('input.txt','r',encoding='utf-8') as f:
  text = f.read()

#print(len(text))

chars = sorted(list(set(text)))
#print("".join(chars))
vocab_size = len(chars)
#print(vocab_size)
strtoi = {s:i for i,s in enumerate(chars)}
itostr = {i:s for i,s in enumerate(chars)}


encode = lambda s: [strtoi[c] for c in s]
decode = lambda l: "".join(itostr[i] for i in l)

#print(encode("Pranav"))
#print(decode(encode("Pranav")))

#----------------------------------------------------------------------------------------

data = torch.tensor(encode(text),dtype=torch.long)
#print(data.shape)
#print(data[:1000])

n =  int(.9*len(data))
train_data = data[:n]
val_data = data[n:]

#print(train_data.shape)
#print(val_data.shape)


#--------------------------------------------------------------------------


def get_batch(split):
  data = train_data if split == "train" else val_data
  ix = torch.randint(len(data)-block_size,(batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  x,y = x.to(device),y.to(device)
  return x,y

'''class LayerNorm1d:
  def __init__(self,dim,eps=1e-5):
    self.eps = eps
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)

  def __call__(self,x):
     xmean = x.mean(1,keepdim=True)#layer mean
     xvar = x.var(1,keepdim=True)#layer variance 
     xhat = (x-xmean)/torch.sqrt(xvar+self.eps) #normalize to unit variance 
     self.out = self.gamma*xhat+self.beta
     return self.out
  
  def parameters(self):
     return [self.gamma,self.beta]'''



class Head(nn.Module):
    '''Implements a single attention block'''
    def __init__(self,head_size):
        super().__init__()


        self.hs = head_size
        self.key = nn.Linear(n_embed,head_size,bias = False)
        self.query = nn.Linear(n_embed,head_size,bias=False)
        self.value = nn.Linear(n_embed,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))

        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
      B,T,C = x.shape
      k = self.key(x) #(B,T,C)
      q = self.query(x) #(B,T,C)
      v = self.value(x) #(B,T,C)

      wei = q @ k.transpose(-2,-1) * self.hs**(-.5) #(B,T,C)@(B,C,T)=>(B,T,T))    
      wei = wei.masked_fill(self.tril[:T,:T] == 0,float('-inf')) #(B,T,T)
      wei = F.softmax(wei,dim=-1) #(B,T,T)
      wei = self.dropout(wei)

      output = wei @ v #(B,T,T) @ (B,T,C) => (B,T,C)
      return output

class MultiHeadAttention(nn.Module):
  
    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) 
        self.proj = nn.Linear(n_embed,n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
      out = torch.cat([h(x) for h in self.heads],dim=-1)
      out = self.dropout(self.proj(out))
      

      return out
    

class FeedForward(nn.Module):
    def __init__(self,n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed,4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed,n_embed),
            nn.Dropout(dropout),
        )
    def forward(self,x):
       return self.net(x)


class Block(nn.Module):
    def __init__(self,n_embed,n_head):
      super().__init__()
      head_size = n_embed//n_head
      self.sa = MultiHeadAttention(n_head,head_size)
      self.ffw = FeedForward(n_embed)
      self.ln1 = nn.LayerNorm(n_embed)
      self.ln2 = nn.LayerNorm(n_embed)

    def forward(self,x):
       x = x + self.sa(self.ln1(x))
       x = x + self.ffw(self.ln2(x))
       return x


















class BigramLangModel(nn.Module):
  def __init__(self):
    super().__init__()

    self.token_embedding = nn.Embedding(vocab_size,n_embed)
    self.token_pos_embedding = nn.Embedding(block_size,n_embed)
    self.blocks = nn.Sequential(*[Block(n_embed,n_head=n_head) for _ in range(n_layer)])

    self.lm_head = nn.Linear(n_embed,vocab_size)
    self.ln_f = nn.LayerNorm(n_embed)

    self.apply(self._init_weights)

  def _init_weights(self,module):
    if isinstance(module,nn.Linear):
      torch.nn.init.normal_(module.weight,mean=0.0,std=.02)
      if module.bias is not None:
          torch.nn.init.zeros_(module.bias)
    elif isinstance(module,nn.Embedding):
        torch.nn.init.normal_(module.weight,mean=0.0,std=.02)

  def forward(self,idx,target=None):
    B,T = idx.shape
    tok_emd = self.token_embedding(idx) #B,T,C
    pos = self.token_pos_embedding(torch.arange(T,device=device))[None,:]# T,C
    x = tok_emd + pos #B,T,C
    x = self.blocks(x)
    x = self.ln_f(x)
    logits = self.lm_head(x) #B,T,Vocab_size
    if target == None:
      loss = None
    else:
      B,T,C = logits.shape
      logits = logits.view(B*T,C)
      target = target.view(B*T)

      loss = F.cross_entropy(logits,target)

    return logits,loss

  def generate(self,idx,max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx[:,-block_size:]#crop idx to last block size token
      logits,loss = self(idx_cond)

      logits = logits[:,-1,:]
      probs = F.softmax(logits,dim=1)
      idx_next = torch.multinomial(probs,num_samples=1) #(8,1)
      idx = torch.cat((idx,idx_next),dim=1)
    return idx


model = BigramLangModel()
model = model.to(device)
print(f"Model created and moved to {device}")

optimizer = torch.optim.Adam(model.parameters(),lr=lr)


@ torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ["train","val"]:
    losses = torch.zeros(eval_iter)
    for epoch in range(eval_iter):
      X,Y = get_batch(split)
      logits,loss = model(X,Y)
      losses[epoch] = loss.item()
    out[split] = losses.mean()
    model.train()
  return out


for epoch in range(epochs):

  if epoch % eval_interval == 0:
    losses = estimate_loss()
    print(f"step {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
  
  xb,yb = get_batch("train")
  logits,loss = model(xb,yb)

  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()


context = torch.zeros((1,1),dtype=torch.long,device=device)
print(decode(model.generate(context,max_new_tokens=500)[0].tolist()))