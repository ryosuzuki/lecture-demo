export const corpus = [
  [['<BOS>','the'],'cat'],[['<BOS>','the'],'dog'],[['<BOS>','the'],'sun'],
  [['the','cat'],'sat'],[['the','dog'],'ran'],[['the','sun'],'shines'],
  [['cat','sat'],'on'],[['dog','ran'],'to'],[['sun','shines'],'bright'],
  [['sat','on'],'the'],[['ran','to'],'the'],[['shines','bright'],'today'],
  [['on','the'],'mat'],[['to','the'],'park'],
  [['the','cat'],'sat'],[['the','dog'],'ran'],[['cat','sat'],'on'],[['sat','on'],'the']
];
export const vocab=[...new Set(corpus.flatMap(([xs,y])=>[...xs,y]))].sort();
export const examples=[
  {sentence:'the cat ___',context:['the','cat'],target:'sat'},
  {sentence:'the dog ___',context:['the','dog'],target:'ran'},
  {sentence:'the sun ___',context:['the','sun'],target:'shines'},
  {sentence:'cat sat ___',context:['cat','sat'],target:'on'},
  {sentence:'sun shines ___',context:['sun','shines'],target:'bright'},
  {sentence:'shines bright ___',context:['shines','bright'],target:'today'},
  {sentence:'on the ___',context:['on','the'],target:'mat'}
];
const D=3,H=4;
const mulberry32=a=>()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296};
const mat=(r,c,f)=>Array.from({length:r},()=>Array.from({length:c},f));
const vec=(n,f)=>Array.from({length:n},f);
export function fresh(){const rand=mulberry32(20260916),rv=()=> (rand()-.5)*1.2;return{E:mat(vocab.length,D,rv),W:mat(D,H,rv),b:vec(H,()=>0),O:mat(H,vocab.length,rv),c:vec(vocab.length,()=>0),step:0}}
export let params=fresh();
export function setParams(p){params=JSON.parse(JSON.stringify(p))}
export function resetParams(){params=fresh();return params}
const softmax=z=>{const m=Math.max(...z),e=z.map(x=>Math.exp(x-m)),s=e.reduce((a,b)=>a+b,0);return e.map(x=>x/s)};
export function forward(context,target){const tokens=Array.isArray(context)?context:[context],xs=tokens.map(t=>vocab.indexOf(t)),y=vocab.indexOf(target);const e=Array(D).fill(0).map((_,i)=>xs.reduce((s,x)=>s+params.E[x][i],0)/xs.length);const pre=Array(H).fill(0).map((_,j)=>params.b[j]+e.reduce((s,v,i)=>s+v*params.W[i][j],0));const h=pre.map(Math.tanh);const logits=vocab.map((_,k)=>params.c[k]+h.reduce((s,v,j)=>s+v*params.O[j][k],0));const probs=softmax(logits);return{xs,tokens,y,e:[...e],pre,h,logits,probs,loss:-Math.log(Math.max(1e-12,probs[y]))}}
function zerosLike(){return{E:mat(vocab.length,D,()=>0),W:mat(D,H,()=>0),b:vec(H,()=>0),O:mat(H,vocab.length,()=>0),c:vec(vocab.length,()=>0)}}
export function gradients(context,target){const f=forward(context,target),g=zerosLike(),dz=[...f.probs];dz[f.y]-=1;for(let k=0;k<vocab.length;k++){g.c[k]=dz[k];for(let j=0;j<H;j++)g.O[j][k]=f.h[j]*dz[k]}const dh=Array(H).fill(0).map((_,j)=>dz.reduce((s,v,k)=>s+v*params.O[j][k],0));const dp=dh.map((v,j)=>v*(1-f.h[j]**2));for(let j=0;j<H;j++){g.b[j]=dp[j];for(let i=0;i<D;i++)g.W[i][j]=f.e[i]*dp[j]}for(const x of f.xs)for(let i=0;i<D;i++)g.E[x][i]+=dp.reduce((s,v,j)=>s+v*params.W[i][j],0)/f.xs.length;return{f,g}}
export function trainPair(context,target,lr=.12){const{f,g}=gradients(context,target);for(const n of ['E','W','O'])for(let i=0;i<params[n].length;i++)for(let j=0;j<params[n][i].length;j++)params[n][i][j]-=lr*g[n][i][j];for(const n of ['b','c'])for(let i=0;i<params[n].length;i++)params[n][i]-=lr*g[n][i];params.step++;return{before:f,after:forward(context,target),g}}
export function trainCorpus(steps,lr=.12){for(let s=0;s<steps;s++){const[a,b]=corpus[s%corpus.length];trainPair(a,b,lr)}return params}
export function getRef(ref){let v=params[ref.group];for(const i of ref.idx)v=v[i];return v}
export function setRef(ref,val){let v=params[ref.group];for(let i=0;i<ref.idx.length-1;i++)v=v[ref.idx[i]];v[ref.idx.at(-1)]=val}
export function gradRef(g,ref){let v=g[ref.group];for(const i of ref.idx)v=v[i];return v}
export function lossSlice(context,target,a,b,range=1.1,n=25){const av=getRef(a),bv=getRef(b),out=[];for(let i=0;i<n;i++){const x=av-range+2*range*i/(n-1),row=[];setRef(a,x);for(let j=0;j<n;j++){const y=bv-range+2*range*j/(n-1);setRef(b,y);row.push({x,y,z:forward(context,target).loss})}out.push(row)}setRef(a,av);setRef(b,bv);return out}
export function surfaceRefs(primary){const a=primary||{group:'W',idx:[0,0]},same=(u,v)=>u.group===v.group&&u.idx.length===v.idx.length&&u.idx.every((x,i)=>x===v.idx[i]),candidates=[{group:'W',idx:[1,1]},{group:'W',idx:[2,2]},{group:'O',idx:[0,0]}];return{a,b:candidates.find(r=>!same(r,a))}}
export function state(){return JSON.parse(JSON.stringify(params))}
export const shape={D,H,V:vocab.length};
