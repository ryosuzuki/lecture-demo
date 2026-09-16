"""Real hidden-state recycling on a third-party Coconut replication checkpoint.
No execution of downloaded pickle code: torch.load(...weights_only=True).
This script does NOT claim the checkpoint is the original authors' release.
"""
from pathlib import Path
import json, time, hashlib, os
import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
ROOT=Path(__file__).resolve().parent
CHECKPOINT=Path(os.environ.get('COCONUT_CHECKPOINT','/tmp/openclaw/coconut-model/checkpoint_49'))
MODEL_ID='bmarti44/coconut-curriculum-checkpoints/coconut/checkpoint_49'
_MODEL=None

def load():
 global _MODEL
 if _MODEL is not None:return _MODEL
 torch.set_num_threads(4)
 tok=GPT2Tokenizer.from_pretrained(str(ROOT),local_files_only=True)
 tok.add_special_tokens({'additional_special_tokens':['<|start-latent|>','<|end-latent|>','<|latent|>']})
 cfg=GPT2Config.from_json_file(str(ROOT/'config.json'));cfg.vocab_size=len(tok);cfg._attn_implementation='eager'
 model=GPT2LMHeadModel(cfg)
 sd=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
 sd={k.removeprefix('base_causallm.'):v for k,v in sd.items() if k.startswith('base_causallm.')}
 missing, unexpected=model.load_state_dict(sd,strict=False)
 # Tied alias may be absent; no silently missing transformer weights allowed.
 if any(k!='lm_head.weight' for k in missing):raise RuntimeError(f'Missing weights: {missing}')
 model.eval()
 _MODEL=(model,tok)
 return _MODEL

def pca(vectors):
 a=np.asarray(vectors,dtype=float)
 if len(a)<2:return [[0.,0.,0.]]*len(a),[0,0,0]
 centered=a-a.mean(axis=0)
 u,s,v=np.linalg.svd(centered,full_matrices=False)
 xyz=centered@v[:3].T
 if xyz.shape[1]<3:xyz=np.pad(xyz,((0,0),(0,3-xyz.shape[1])))
 variance=(s*s)/(sum(s*s) or 1)
 return xyz.tolist(),(variance[:3].tolist()+[0,0,0])[:3]

@torch.inference_mode()
def run(question,steps=6,max_tokens=28):
 steps=int(steps)
 if not 0<=steps<=8:raise ValueError('Use 0–8 latent steps')
 if not isinstance(question,str) or not question.strip() or len(question)>12000:raise ValueError('Question must be 1–12000 characters')
 model,tok=load(); start=time.perf_counter()
 ids=tok.encode(question.strip()+'\n',add_special_tokens=False)
 if len(ids)>850:raise ValueError('Question too long (850 token limit)')
 ids+=[tok.convert_tokens_to_ids('<|start-latent|>')]
 emb=model.transformer.wte(torch.tensor([ids]))
 vectors=[];snapshots=[]
 for i in range(steps):
  out=model(inputs_embeds=emb,output_hidden_states=True,use_cache=False)
  h=out.hidden_states[-1][:,-1:,:]
  vec=h[0,0].float().tolist();vectors.append(vec)
  probabilities=torch.softmax(out.logits[0,-1].float(),dim=-1);vals,ix=torch.topk(probabilities,5)
  snapshots.append({'step':i+1,'norm':float(h.norm()),'vector':vec,'probe':[{'token':tok.decode([int(j)]),'p':float(p)} for p,j in zip(vals,ix)]})
  emb=torch.cat([emb,h],dim=1)
 end=torch.tensor([[tok.convert_tokens_to_ids('<|end-latent|>')]])
 emb=torch.cat([emb,model.transformer.wte(end)],dim=1)
 generated=[];first_probs=[];decode_steps=[]
 for i in range(max_tokens):
  out=model(inputs_embeds=emb,use_cache=False)
  probs=torch.softmax(out.logits[0,-1].float(),dim=-1)
  if i==0:
   vals,ix=torch.topk(probs,8);first_probs=[{'token':tok.decode([int(j)]),'p':float(p)} for p,j in zip(vals,ix)]
  vals,ix=torch.topk(probs,8)
  token=int(torch.argmax(probs));generated.append(token)
  decode_steps.append({'token':tok.decode([token]),'top':[{'token':tok.decode([int(j)]),'p':float(p)} for p,j in zip(vals,ix)]})
  if token==tok.eos_token_id:break
  emb=torch.cat([emb,model.transformer.wte(torch.tensor([[token]]))],dim=1)
 xyz,var=pca(vectors)
 for s,pos in zip(snapshots,xyz):s['position']=pos
 return {'question':question,'latentSteps':steps,'states':snapshots,'answer':tok.decode(generated,skip_special_tokens=True),'answerTokens':generated,'decodeSteps':decode_steps,'firstTokenProbabilities':first_probs,'elapsedSeconds':round(time.perf_counter()-start,3),'inputTokens':len(ids)-1,'projection':'PCA fitted to this trajectory only; axes may change between runs. Not semantic coordinates.','explainedVariance':var,'model':MODEL_ID,'hiddenSize':768,'mode':'real model inference','recordedAt':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}

if __name__=='__main__':
 data=json.loads((ROOT/'prosqa_test.json').read_text())
 samples=[]
 # Fixed first three held-out questions; no cherry-picking by accuracy.
 for i in range(3):
  r=run(data[i]['question'],6);r['expectedAnswer']=data[i]['answer'];r['datasetIndex']=i
  r['exactMatch']=r['answer'].strip().removeprefix('###').strip()==data[i]['answer'].strip()
  samples.append(r);print(i,r['answer'],r['elapsedSeconds'],flush=True)
  (ROOT/'recorded-traces.json').write_text(json.dumps({'provenance':{'model':MODEL_ID,'checkpointSha256':hashlib.file_digest(CHECKPOINT.open('rb'),'sha256').hexdigest(),'dataset':'facebookresearch/coconut/data/prosqa_test.json','inference':'Full-prefix forward pass; previous last hidden state inserted as next latent input embedding. Greedy decoding. No training.'},'traces':samples}))
