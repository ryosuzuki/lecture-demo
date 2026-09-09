// Small, hand-designed matrices. Not trained weights or measured semantic axes.
export const words=['a','fluffy','blue','creature','roamed','the','verdant','forest'];
export const E=[[0,0,0,0],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,0],[0,0,0,0],[.7,.7,0,0],[0,0,1,0]];
export const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
export const mul=(a,b)=>a.map(r=>b[0].map((_,j)=>r.reduce((s,x,k)=>s+x*b[k][j],0)));
export const softmax=a=>{const m=Math.max(...a),e=a.map(x=>Math.exp(x-m)),z=e.reduce((s,x)=>s+x,0);return e.map(x=>x/z)};
export function trace(s={},head=s.head??0){
 const {pos=1,mask=true,temp=1,gain=1,qx=2,qy=2,query=3}=s;
 const X=E.map((e,i)=>e.map((x,j)=>j===3?i/7*pos:x));
 const WQ=[[0,0],[0,0],[head===2?.2:2,head===1?.2:2],[.35,.35]],WK=[[2,0],[0,2],[0,0],[.35,.35]],WV=[[head===2?.2:1,0],[0,head===1?.2:1],[0,0],[0,0]],UP=[[1,0,0,0],[0,1,0,0]];
 const Q=mul(X,WQ),K=mul(X,WK),V=mul(X,WV);
 // Explicit single-query intervention; default offsets leave projection intact.
 Q[query][0]+=qx-2;Q[query][1]+=qy-2;V[1]=V[1].map(x=>x*gain);
 const S=Q.map(q=>K.map(k=>dot(q,k)/Math.sqrt(2))),L=S.map((r,i)=>r.map((x,j)=>mask&&j>i?-Infinity:x/temp)),A=L.map(softmax),O=mul(A,V),delta=mul(O,UP),Y=X.map((r,i)=>r.map((x,j)=>x+delta[i][j]));
 return {X,WQ,WK,WV,UP,Q,K,V,S,L,A,O,delta,Y};
}
export function multi(s={}){const heads=[0,1,2].map(h=>trace(s,h)),enabled=s.enabled??[true,true,true],concat=words.map((_,i)=>heads.flatMap((h,j)=>h.O[i].map(x=>enabled[j]?x:0))),WO=heads.flatMap(h=>h.UP),delta=mul(concat,WO);return {heads,concat,WO,delta,Y:heads[0].X.map((r,i)=>r.map((x,j)=>x+delta[i][j]))}}
export function cross(s={}){const keys=[[1,0],[0,1],[.6,.6]],queries=[[s.qx??2,s.qy??2],[0,2]],values=[[1,0,0],[0,1,0],[0,0,1]],A=queries.map(q=>softmax(keys.map(k=>dot(q,k)/Math.sqrt(2))));return {keys,queries,values,A,O:mul(A,values)}}
export const parameters=(d,h,k,l)=>({perHead:4*d*k,block:4*d*k*h,total:4*d*k*h*l,fullValue:d*d,factoredValue:2*d*k});
