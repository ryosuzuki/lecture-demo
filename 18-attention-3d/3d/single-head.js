// Exact, low-dimensional teaching model. These matrices are deliberately chosen,
// not pretrained GPT weights. Column-vector convention matches the lecture.
export const tokens=['a','fluffy','blue','creature','roamed','the','verdant','forest'];
export const inputs=[[.1,0,0],[1,0,0],[0,1,0],[0,0,1],[-.6,0,.2],[.2,0,0],[.7,.7,0],[0,0,.8]];
export const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
export const mv=(m,v)=>m.map(r=>dot(r,v));
export function singleHead({focus=0,key=1,value=1,mask=true,query=3}={}){
 const WQ=[[.1,0,2+focus],[0,.1,2-focus],[0,0,.2]],WK=[[2,0,0],[0,2*key,0],[0,0,.2]],WV=[[1.8,.1,0],[.4,.5,0],[0,2.2*value,0]],WO=[[1,0,0],[0,1,0],[0,0,1]];
 const Q=inputs.map(x=>mv(WQ,x)),K=inputs.map(x=>mv(WK,x)),V=inputs.map(x=>mv(WV,x));
 const raw=K.map(k=>dot(Q[query],k)),scaled=raw.map(x=>x/Math.sqrt(3));
 const logits=scaled.map((x,i)=>mask&&i>query?-Infinity:x),max=Math.max(...logits),exp=logits.map(x=>x===-Infinity?0:Math.exp(x-max)),Z=exp.reduce((a,b)=>a+b,0),weights=exp.map(x=>x/Z);
 const contributions=V.map((v,i)=>v.map(x=>x*weights[i])),z=[0,1,2].map(d=>contributions.reduce((s,v)=>s+v[d],0)),delta=mv(WO,z),output=inputs[query].map((x,d)=>x+delta[d]);
 return {WQ,WK,WV,WO,Q,K,V,raw,scaled,logits,max,exp,Z,weights,contributions,z,delta,output,input:inputs[query],query};
}
