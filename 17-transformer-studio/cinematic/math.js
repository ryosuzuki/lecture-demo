export const words=['青い','猫','が','眠る'];
// Deliberately small illustrative activations, already including position information.
export const X=[[1,.2,-.4],[.2,1,.3],[-.3,.1,.8],[.6,.8,-.2]];
export const WQ=[[[1,.2],[.3,1],[-.4,.5]],[[.2,1],[1,-.3],[.6,.2]],[[1,-.6],[-.4,.2],[.3,1]]];
export const WK=[[[.4,1],[1,.1],[-.3,-.5]],[[1,.3],[-.2,1],[.5,-.4]],[[.1,1],[.9,.3],[-.6,.4]]];
export const WV=[[[1,.2,-.3],[.1,1,.4],[.3,-.2,.7]],[[.2,1,.1],[1,.3,-.4],[-.2,.4,1]],[[.8,-.1,.5],[.2,.9,-.3],[-.4,.2,.8]]];
export const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
export const mul=(A,B)=>A.map(row=>B[0].map((_,j)=>row.reduce((s,v,k)=>s+v*B[k][j],0)));
export function softmax(a){const m=Math.max(...a),e=a.map(x=>Math.exp(x-m)),z=e.reduce((a,b)=>a+b,0);return e.map(x=>x/z)}
export function trace({head=0,mask=true,temp=1,gain=1}={}){const Q=mul(X,WQ[head]),K=mul(X,WK[head]),V=mul(X,WV[head]);V[1]=V[1].map(x=>x*gain);const scores=Q.map(q=>K.map(k=>dot(q,k)/Math.sqrt(2)));const logits=scores.map((r,i)=>r.map((v,j)=>mask&&j>i?-Infinity:v/temp));const A=logits.map(softmax),O=mul(A,V);return {X,Q,K,V,scores,logits,A,O}}
export function multi(options={}){const heads=[0,1,2].map(head=>trace({...options,head}));const concat=X.map((_,i)=>heads.flatMap(h=>h.O[i]));const WO=Array.from({length:9},(_,i)=>Array.from({length:3},(_,j)=>(i%3===j?.55:.08)*(i<6?1:-.5)));const delta=mul(concat,WO);return {heads,concat,WO,delta,residual:X.map((r,i)=>r.map((v,j)=>v+delta[i][j]))}}
