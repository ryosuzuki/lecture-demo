/* Independent JavaScript inference implementation. PyTorch Linear weights: [out][in]. */
(function(root){
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0),add=(a,b)=>a.map((x,i)=>x+b[i]);
const linear=(x,w,b)=>w.map((r,i)=>dot(x,r)+(b?b[i]:0));
const softmax=a=>{const max=Math.max(...a),e=a.map(x=>Math.exp(x-max)),s=e.reduce((x,y)=>x+y,0);return e.map(x=>x/s)};
const ln=(x,g,b)=>{let m=x.reduce((a,b)=>a+b)/x.length,v=x.reduce((s,a)=>s+(a-m)**2,0)/x.length;return x.map((a,i)=>(a-m)/Math.sqrt(v+1e-5)*g[i]+b[i])};
function forward(model,ids,options={}){
let cp=options.checkpoint??model.checkpoints.length-1,w=model.checkpoints[cp].weights,M=model.metadata,T=ids.length;
if(!T||T>M.maxTokens||ids.some(i=>i<0||i>=model.vocabulary.length))throw Error('Use 1–8 known tokens.');
let tokenEmb=ids.map(i=>w['emb.weight'][i]),positions=ids.map((_,i)=>w['pos.weight'][i]),x=tokenEmb.map((v,i)=>add(v,positions[i])),input=x.map(v=>v.slice()),blocks=[];
for(let bi=0;bi<M.layers;bi++){
 const pre='blocks.'+bi+'.',inp=x.map(v=>v.slice()),z=x.map(v=>ln(v,w[pre+'ln1.weight'],w[pre+'ln1.bias']));
 const all={};for(const typ of ['q','k','v']){let l=z.map(v=>linear(v,w[pre+typ+'.weight']));all[typ]=Array.from({length:M.heads},(_,h)=>l.map(v=>v.slice(h*M.dKey,(h+1)*M.dKey)))}
 const intervention=options.layer===bi;
 if(intervention&&options.qOverride)all.q[options.head][options.query]=options.qOverride.slice();
 if(intervention&&options.vOverride)all.v[options.head][options.source]=options.vOverride.slice();
 let heads=[];
 for(let h=0;h<M.heads;h++){
  const scores=all.q[h].map((q,i)=>all.k[h].map((k,j)=>(options.masked!==false&&j>i)?-Infinity:dot(q,k)/Math.sqrt(M.dKey)));
  const a=scores.map(softmax),messages=a.map(row=>all.v[h][0].map((_,d)=>row.reduce((s,v,j)=>s+v*all.v[h][j][d],0)));
  const gain=intervention&&h===options.head?(options.gain??1):1;
  heads.push({q:all.q[h],k:all.k[h],v:all.v[h],scores,attention:a,messages,gain});
 }
 let message=ids.map((_,i)=>heads.flatMap(h=>h.messages[i].map(x=>x*h.gain))),delta=message.map(v=>linear(v,w[pre+'o.weight'])),residual=x.map((v,i)=>add(v,delta[i]));
 let n=residual.map(v=>ln(v,w[pre+'ln2.weight'],w[pre+'ln2.bias'])),hidden=n.map(v=>linear(v,w[pre+'fc1.weight'],w[pre+'fc1.bias']).map(x=>Math.max(0,x))),mlp=hidden.map(v=>linear(v,w[pre+'fc2.weight'],w[pre+'fc2.bias']));x=residual.map((v,i)=>add(v,mlp[i]));
 blocks.push({input:inp,normalized:z,heads,message,delta,residual,mlp,output:x.map(v=>v.slice())});
}
const final=x.map(v=>ln(v,w['norm.weight'],w['norm.bias'])),logits=final.map(v=>linear(v,w['readout.weight'])),probabilities=logits.map(v=>softmax(v.map(x=>x/(options.temperature??1))));return{tokenEmb,positions,input,blocks,final,logits,probabilities};
}
root.TransformerEngine={forward,linear,softmax,ln,dot,add};if(typeof module!=='undefined')module.exports=root.TransformerEngine;
})(typeof window!=='undefined'?window:globalThis);
