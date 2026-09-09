// Hand-designed instructional model, not learned weights or measured embeddings.
export const words=['a','fluffy','blue','creature'];
export const X=[[-.2,-.2,0],[1,0,0],[0,1,0],[0,0,1]];
export const WQ=[[[0,0],[0,0],[2,2]],[[0,0],[0,0],[2,0]],[[0,0],[0,0],[0,2]]];
export const WK=[[[2,0],[0,2],[0,0]],[[2,0],[0,2],[0,0]],[[2,0],[0,2],[0,0]]];
export const WV=[[[1,0],[0,1],[0,0]],[[.5,0],[0,.2],[0,0]],[[.2,0],[0,.5],[0,0]]];
export const UP=[[[1,0,0],[0,1,0]],[[1,0,0],[0,1,0]],[[1,0,0],[0,1,0]]];
export const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
export const mul=(A,B)=>A.map(row=>B[0].map((_,j)=>row.reduce((s,v,k)=>s+v*B[k][j],0)));
export function softmax(a){const m=Math.max(...a),e=a.map(x=>Math.exp(x-m)),z=e.reduce((a,b)=>a+b,0);return e.map(x=>x/z)}
export function trace({head=0,mask=true,temp=1,gain=1}={}){const Q=mul(X,WQ[head]),K=mul(X,WK[head]),V=mul(X,WV[head]);V[1]=V[1].map(x=>x*gain);const scores=Q.map(q=>K.map(k=>dot(q,k)/Math.sqrt(2)));const logits=scores.map((r,i)=>r.map((v,j)=>mask&&j>i?-Infinity:v/temp));const A=logits.map(softmax),O=mul(A,V),lifted=mul(V,UP[head]),delta=mul(O,UP[head]),residual=X.map((r,i)=>r.map((v,j)=>v+delta[i][j]));return {X,Q,K,V,scores,logits,A,O,lifted,delta,residual,UP:UP[head]}}
export function multi(options={}){const heads=[0,1,2].map(head=>trace({...options,head}));const concat=X.map((_,i)=>heads.flatMap(h=>h.O[i]));const WO=UP.flat(),delta=mul(concat,WO);return {heads,concat,WO,delta,residual:X.map((r,i)=>r.map((v,j)=>v+delta[i][j]))}}
