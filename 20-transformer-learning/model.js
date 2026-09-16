// A real, deliberately tiny decoder. Scalar reverse-mode autodiff; no dependencies.
// One causal attention head, residuals, tanh MLP. No layer normalization.
export const examples=[
 {text:'After lunch the cat sat on the',tokens:['after','lunch','the','cat','sat','on','the'],target:'mat'},
 {text:'In the morning the sun',tokens:['in','the','morning','the','sun'],target:'shines'},
 {text:'After lunch the dog ran to the',tokens:['after','lunch','the','dog','ran','to','the'],target:'park'},
 {text:'At night the cat sleeps on the',tokens:['at','night','the','cat','sleeps','on','the'],target:'mat'},
];
export const vocab=[...new Set(examples.flatMap(e=>[...e.tokens,e.target]))].sort();
class V{
 constructor(value,parents=[],back=()=>{}){this.value=value;this.grad=0;this.parents=parents;this.back=back}
 add(b){b=val(b);const o=new V(this.value+b.value,[this,b],()=>{this.grad+=o.grad;b.grad+=o.grad});return o}
 mul(b){b=val(b);const o=new V(this.value*b.value,[this,b],()=>{this.grad+=b.value*o.grad;b.grad+=this.value*o.grad});return o}
 exp(){const o=new V(Math.exp(this.value),[this],()=>this.grad+=o.value*o.grad);return o}
 log(){const o=new V(Math.log(this.value),[this],()=>this.grad+=o.grad/this.value);return o}
 inv(){const o=new V(1/this.value,[this],()=>this.grad-=o.grad/(this.value*this.value));return o}
 tanh(){const o=new V(Math.tanh(this.value),[this],()=>this.grad+=(1-o.value*o.value)*o.grad);return o}
}
const val=x=>x instanceof V?x:new V(x),sum=a=>a.reduce((s,x)=>s.add(x),val(0));
const mm=(x,w)=>w[0].map((_,j)=>sum(x.map((v,i)=>v.mul(w[i][j]))));
const softmax=z=>{const max=Math.max(...z.map(v=>v.value)),ex=z.map(v=>v.add(-max).exp()),inv=sum(ex).inv();return ex.map(v=>v.mul(inv))};
const nums=a=>a.map(v=>Array.isArray(v)?nums(v):v.value);
export class TinyTransformer{
 constructor(seed=42){this.seed=seed;this.reset()}
 reset(){let a=this.seed;const rand=()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296};this.params=[];const matrix=(name,r,c,scale=.5)=>Array.from({length:r},(_,i)=>Array.from({length:c},(_,j)=>{const p=new V((rand()-.5)*2*scale);p.name=`${name}[${i},${j}]`;this.params.push(p);return p}));
 this.E=matrix('Embedding',vocab.length,4);this.P=matrix('Position',8,4,.1);this.Q=matrix('Query',4,4);this.K=matrix('Key',4,4);this.V=matrix('Value',4,4);this.A=matrix('AttentionOut',4,4);this.W1=matrix('MLP1',4,8);this.W2=matrix('MLP2',8,4);this.U=matrix('Unembedding',4,vocab.length);this.steps=0;
 }
 forward(tokens,target){if(!tokens.length||tokens.length>8)throw Error('Use 1–8 vocabulary tokens');if(tokens.some(t=>!vocab.includes(t))||!vocab.includes(target))throw Error('Unknown token');
 const x=tokens.map((t,i)=>this.E[vocab.indexOf(t)].map((v,d)=>v.add(this.P[i][d]))),q=x.map(v=>mm(v,this.Q)),k=x.map(v=>mm(v,this.K)),v=x.map(v=>mm(v,this.V));
 const attention=q.map((qi,i)=>softmax(k.slice(0,i+1).map(kj=>sum(qi.map((z,d)=>z.mul(kj[d]))).mul(.5))));
 const mixed=attention.map(row=>Array.from({length:4},(_,d)=>sum(row.map((a,j)=>a.mul(v[j][d])))));
 const attn=mixed.map((z,i)=>mm(z,this.A).map((a,d)=>a.add(x[i][d])));
 const hidden=attn.map(z=>mm(mm(z,this.W1).map(a=>a.tanh()),this.W2).map((a,d)=>a.add(z[d])));
 const logits=mm(hidden.at(-1),this.U),probs=softmax(logits);this.lossNode=probs[vocab.indexOf(target)].log().mul(-1);
 return {tokens:[...tokens],x:nums(x),attention:nums(attention).map((r,i)=>r.concat(Array(tokens.length-i-1).fill(0))),attn:nums(attn),hidden:nums(hidden),logits:nums(logits),probs:nums(probs),loss:this.lossNode.value};
 }
 backward(tokens,target){const trace=this.forward(tokens,target),seen=new Set(),order=[];function visit(v){if(seen.has(v))return;seen.add(v);for(const p of v.parents)visit(p);order.push(v)}visit(this.lossNode);for(const p of this.params)p.grad=0;for(const n of order)n.grad=0;this.lossNode.grad=1;for(let i=order.length-1;i>=0;i--)order[i].back();return trace}
 step(tokens,target,lr=.05){if(!Number.isFinite(lr)||lr<0)throw Error('Invalid learning rate');const before=this.backward(tokens,target);for(const p of this.params)p.value-=lr*p.grad;this.steps++;return{before,after:this.forward(tokens,target)}}
}
