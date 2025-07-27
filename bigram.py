import torch
import torch.nn as nn
import torch.nn.functional as F


#wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
#--------------------------------------------------------------------------------
block_size = 8
batch_size = 4
epochs = 5000
lr = 1e-2
device = 'cuda' if torch.cuda.is_available() else "cpu"
print(device)
eval_iter = 200
eval_interaval = 300
n_embed = 32

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





with torch.no_grad():
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

class BigramLangModel(nn.Module):
  def __init__(self):
    super().__init__()

    self.token_embedding = nn.Embedding(vocab_size,n_embed)
    self.token_pos_embedding = nn.Embedding(block_size,n_embed)
    self.f_head = nn.Linear(n_embed,vocab_size)

  def forward(self,idx,target=None):
    B,T = idx.shape
    tok_emd = self.token_embedding(idx) #B,T,C
    pos = self.token_pos_embedding(torch.arange(T,device=device))[None,:]# T,C
    x = tok_emd + pos #B,T,C
    logits = self.f_head(x) #B,T,Vocab_size
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
      logits,loss = self(idx)

      logits = logits[:,-1,:]
      probs = F.softmax(logits,dim=1)
      idx_next = torch.multinomial(probs,num_samples=1) #(8,1)
      idx = torch.cat((idx,idx_next),dim=1)
    return idx


model = BigramLangModel()
model = model.to(device)
print(f"Model created and moved to {device}")

optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)


for epoch in range(epochs):

  if epoch % eval_interaval == 0:
    losses = estimate_loss()
    print(f"step {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
  
  xb,yb = get_batch("train")
  logits,loss = model(xb,yb)

  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()


context = torch.zeros((1,1),dtype=torch.long,device=device)
print(decode(model.generate(context,max_new_tokens=500)[0].tolist()))