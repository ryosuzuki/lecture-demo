"""Reproduce the tiny, learned decoder shipped with Transformer Studio.
Original synthetic corpus; no personal data or pretrained weights.
Requires Python, torch, numpy. CPU training, fixed seed.
"""
import torch, json, math, random, hashlib
from pathlib import Path
from torch import nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parent
torch.manual_seed(23);random.seed(23);torch.set_num_threads(2)
D,H,K,FF,LAYERS,MAX=8,2,2,16,2,8
river=['river','stream','creek','canal'];finance=['savings','central','commercial','investment']
dets=['the','a','this','that'];verbs=[['erodes','crumbles','slopes'],['lends','invests','profits']]
rows=[]
for di,det in enumerate(dets):
 for ci,ctx in enumerate(river+finance):
  split='validation' if (di*3+ci)%7==0 else 'train'
  for verb in verbs[int(ctx in finance)]: rows.append({'text':f'{det} {ctx} bank {verb} .','split':split,'category':'river' if ctx in river else 'finance'})
vocab=['<pad>']+sorted(set(' '.join(x['text'] for x in rows).split()));lookup={s:i for i,s in enumerate(vocab)}
class Block(nn.Module):
 def __init__(self):
  super().__init__();self.ln1=nn.LayerNorm(D);self.q=nn.Linear(D,H*K,bias=False);self.k=nn.Linear(D,H*K,bias=False);self.v=nn.Linear(D,H*K,bias=False);self.o=nn.Linear(H*K,D,bias=False);self.ln2=nn.LayerNorm(D);self.fc1=nn.Linear(D,FF);self.fc2=nn.Linear(FF,D)
 def forward(self,x,trace=False):
  z=self.ln1(x);B,T,_=x.shape
  q,k,v=[m(z).reshape(B,T,H,K).transpose(1,2) for m in [self.q,self.k,self.v]]
  scores=q@k.transpose(-1,-2)/math.sqrt(K);scores=scores.masked_fill(torch.ones(T,T,dtype=torch.bool).triu(1),-float('inf'));a=scores.softmax(-1)
  m=(a@v).transpose(1,2).reshape(B,T,H*K);delta=self.o(m);r=x+delta;n=self.ln2(r);mlp=self.fc2(F.relu(self.fc1(n)));out=r+mlp
  if trace:return out,{'input':x[0].tolist(),'normalized':z[0].tolist(),'q':q[0].tolist(),'k':k[0].tolist(),'v':v[0].tolist(),'attention':a[0].tolist(),'message':m[0].tolist(),'delta':delta[0].tolist(),'residual':r[0].tolist(),'mlp':mlp[0].tolist(),'output':out[0].tolist()}
  return out
class Model(nn.Module):
 def __init__(self):
  super().__init__();self.emb=nn.Embedding(len(vocab),D);self.pos=nn.Embedding(MAX,D);self.blocks=nn.ModuleList([Block() for _ in range(LAYERS)]);self.norm=nn.LayerNorm(D);self.readout=nn.Linear(D,len(vocab),bias=False)
 def forward(self,ids,trace=False):
  x=self.emb(ids)+self.pos(torch.arange(ids.shape[1]));tr=[]
  for block in self.blocks:
   if trace:x,t=block(x,True);tr.append(t)
   else:x=block(x)
  logits=self.readout(self.norm(x));return (logits,tr) if trace else logits
m=Model();opt=torch.optim.AdamW(m.parameters(),lr=.012,weight_decay=.005)
def data(split):
 ids=torch.tensor([[lookup[t] for t in r['text'].split()] for r in rows if r['split']==split]);return ids[:,:-1],ids[:,1:]
x,y=data('train');vx,vy=data('validation')
def weights():return {n:v.detach().tolist() for n,v in m.state_dict().items()}
@torch.no_grad()
def metrics():
 loss=F.cross_entropy(m(x).reshape(-1,len(vocab)),y.reshape(-1)).item();vloss=F.cross_entropy(m(vx).reshape(-1,len(vocab)),vy.reshape(-1)).item();good=0;n=0
 for det in dets:
  for ctx in river+finance:
   if (dets.index(det)*3+(river+finance).index(ctx))%7!=0:continue
   ids=torch.tensor([[lookup[w] for w in [det,ctx,'bank']]]);p=m(ids)[0,-1].softmax(-1);good+=int(vocab[p.argmax()] in verbs[int(ctx in finance)]);n+=1
 positions=F.cross_entropy(m(vx).transpose(1,2),vy,reduction='none').mean(0).tolist()
 return {'validationLossByTarget':dict(zip(['context','bank','verb','period'],positions)),'validationBankVerbLoss':positions[2],'trainLoss':loss,'validationLoss':vloss,'heldoutCategoryAccuracy':good/n,'heldoutPrefixes':n}
checkpoints=[]
for step in range(1201):
 if step in [0,30,100,300,700,1200]:
  stat=metrics();checkpoints.append({'step':step,'metrics':stat,'weights':weights()});print(step,stat,flush=True)
 if step==1200:break
 opt.zero_grad();logits=m(x);loss=F.cross_entropy(logits.reshape(-1,len(vocab)),y.reshape(-1));loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
refs=[]
with torch.no_grad():
 for prompt in ['the river bank','the savings bank','a creek bank','this central bank','the river bank erodes']:
  ids=torch.tensor([[lookup[t] for t in prompt.split()]]);logits,tr=m(ids,True);p=logits[0,-1].softmax(-1)
  refs.append({'prompt':prompt,'logits':logits[0].tolist(),'probabilities':p.tolist(),'blocks':tr});print(prompt,[(vocab[i],round(p[i].item(),3)) for i in p.argsort(descending=True)[:4]],flush=True)
model={'metadata':{'seed':23,'dModel':D,'heads':H,'dKey':K,'dValue':K,'dFF':FF,'layers':LAYERS,'maxTokens':MAX,'parameters':sum(p.numel() for p in m.parameters()),'activation':'relu','epsilon':1e-5,'trainingSteps':1200,'trainSequences':sum(r['split']=='train' for r in rows),'validationSequences':sum(r['split']=='validation' for r in rows),'note':'Original synthetic word-level corpus. This is a tiny trained decoder, not GPT or general English competence. All positions trained with next-token cross-entropy.'},'vocabulary':vocab,'checkpoints':checkpoints,'corpus':rows,'references':refs}
(ROOT/'model.json').write_text(json.dumps(model,separators=(',',':')))
(ROOT/'model.js').write_text('window.MODEL='+json.dumps(model,separators=(',',':'))+';\n')
print('parameters',model['metadata']['parameters'])
