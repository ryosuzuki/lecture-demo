export const words=['a','fluffy','blue','creature','roamed','the','verdant','forest'];
// Read from Chapter 6, 09:25. The two high scores use reconstructed sub-decimal
// precision consistent with the displayed 93.0 / 93.4 and 0.42 / 0.58.
// The unpublished full-precision random source values are not claimed to be known.
export const sourceScores=[
[.7,-83.7,-24.7,-27.8,-5.2,-89.3,-45.2,-36.1],
[-73.4,2.9,-5.4,93.04,-48.2,-87.3,-49.7,7.8],
[-53.4,-5.7,1.8,93.36,-55.6,-56,-26.1,-62.1],
[-21.5,-29.7,-56.1,4.9,-32.4,-92.3,-9.5,-28.1],
[-20.1,-40.9,-87.8,-55.4,.6,-64.7,-96.7,-18.9],
[-87.9,-33.3,-22.6,-31.4,5.5,.6,-4.6,-96.8],
[-41.2,-55.5,-42.3,-59.8,-79,-97.9,3.7,93.8],
[-58.9,-75.5,-91.1,-90.6,-75.6,-89,-70.8,4.7]];
export const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
export const add=(a,b)=>a.map((x,i)=>x+b[i]);
export const scale=(a,k)=>a.map(x=>x*k);
export function softmax(v,T=1){if(T===0){const m=Math.max(...v);const n=v.filter(x=>x===m).length;return v.map(x=>x===m?1/n:0)}const m=Math.max(...v);const e=v.map(x=>x===-Infinity?0:Math.exp((x-m)/T));const s=e.reduce((a,b)=>a+b,0);if(!s)throw Error('No unmasked score');return e.map(x=>x/s)}
export function trace({mask=false,scaled=false,bias=0,value=1,head=0}={}){
 const scores=sourceScores.map(r=>[...r]);scores[2][3]+=bias;
 if(head===1){for(let i=0;i<8;i++)for(let j=0;j<8;j++)scores[i][j]=i===j-1?7:0;}
 if(head===2){for(let i=0;i<8;i++)for(let j=0;j<8;j++)scores[i][j]=i===0?5:0;}
 if(head===3){for(let i=0;i<8;i++)for(let j=0;j<8;j++)scores[i][j]=i===j?5:-1;}
 const logits=scores.map((r,i)=>r.map((x,j)=>mask&&i>j?-Infinity:x/(scaled?Math.sqrt(128):1)));
 const attention=Array.from({length:8},()=>Array(8).fill(0));
 for(let j=0;j<8;j++){const col=softmax(logits.map(r=>r[j]));col.forEach((p,i)=>attention[i][j]=p)}
 const values=[[0,0,0],[1.8,.4,0],[.1,.5,2.2*value],[0,0,0],[.2,0,.1],[0,0,0],[0,1.8,1],[.2,.2,0]];
 const base=[[-.4,0,0],[.1,.2,0],[0,0,.3],[1,.5,-.8],[0,1,0],[0,0,0],[0,.4,.1],[-.5,.8,.4]];
 const delta=words.map((_,j)=>values.reduce((sum,v,i)=>add(sum,scale(v,attention[i][j])),[0,0,0]));
 return {scores,logits,attention,values,base,delta,output:base.map((b,j)=>add(b,delta[j]))};
}
export const sourceLogits=[-.8,-5,.5,1.5,3.5,-2.3,4];
export const parameterCounts={embedding:50257*12288,query:128*12288,key:128*12288,valueFull:12288*12288,head:4*128*12288,block:96*4*128*12288,attention:96*96*4*128*12288};
export function outputEquivalence(){const small=[[.42,.58],[.2,.8]],up=[[[1,2],[0,1],[3,-1]],[[2,0],[-1,2],[1,1]]];const proposed=up.map((u,h)=>u.map(r=>dot(r,small[h])));const summed=add(...proposed);const concat=small.flat();const matrix=up[0].map((r,i)=>[...r,...up[1][i]]);return {small,up,proposed,summed,concat,matrix,fused:matrix.map(r=>dot(r,concat))}}
// Chapter 6, 12:15: values displayed in the separate masking demonstration.
export const maskingScores=[[3.53,.80,1.96,4.48,3.74,-1.95],[1.90,-.30,-.21,.82,.29,2.91],[1.52,.24,.89,.67,2.99,-.41],[.63,-1.71,-5.11,1.31,1.73,-1.48],[4.54,-2.91,.09,-.37,3.07,2.94],[.31,.76,-1.78,-3.96,-.70,.31]];
export function maskingTrace(mask=true,after=false){const logits=maskingScores.map((r,i)=>r.map((v,j)=>mask&&!after&&i>j?-Infinity:v));const a=Array.from({length:6},()=>Array(6));for(let j=0;j<6;j++){const c=softmax(logits.map(r=>r[j]));for(let i=0;i<6;i++)a[i][j]=mask&&after&&i>j?0:c[i];}return {logits,attention:a,sums:a[0].map((_,j)=>a.reduce((s,r)=>s+r[j],0))}}
