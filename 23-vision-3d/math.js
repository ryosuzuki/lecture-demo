export const kernels={blur:Array(9).fill(1/9),vertical:[-1,0,1,-1,0,1,-1,0,1],horizontal:[-1,-1,-1,0,0,0,1,1,1],sharpen:[0,-1,0,-1,5,-1,0,-1,0]};
export const sigmoid=x=>1/(1+Math.exp(-x));
export function image(kind='ring'){return Array.from({length:64},(_,i)=>{let x=i%8,y=Math.floor(i/8);if(kind==='edge')return x>=4?1:.08;if(kind==='diagonal')return Math.abs(x-y)<2?.95:.05;return ((x===2||x===5)&&y>=1&&y<=6||(y===1||y===6)&&x>=2&&x<=5)?1:.03;});}
export function patch(a,r,c,n=8){return Array.from({length:9},(_,i)=>a[(r+Math.floor(i/3))*n+c+i%3]);}
export function correlate(a,k,n=8){let m=n-2;return Array.from({length:m*m},(_,i)=>patch(a,Math.floor(i/m),i%m,n).reduce((s,x,j)=>s+x*k[j],0));}
export const relu=a=>a.map(x=>Math.max(0,x));
export function pool(a,n=6){let m=Math.floor(n/2);return Array.from({length:m*m},(_,i)=>{let r=Math.floor(i/m)*2,c=i%m*2;return Math.max(a[r*n+c],a[r*n+c+1],a[(r+1)*n+c],a[(r+1)*n+c+1]);});}
export function network(a,gain=1,bias=0){let weights=Array.from({length:4},(_,j)=>Array.from({length:8},(_,i)=>Math.sin((j+1)*(i+1)*1.7)*gain));let h=weights.map(w=>sigmoid(w.reduce((s,x,i)=>s+x*a[i],bias)));let ow=[[-1,.5,1,-.5],[.8,-1,.2,1]];let o=ow.map(w=>sigmoid(w.reduce((s,x,i)=>s+x*h[i],0)));return {weights,h,o,ow};}
