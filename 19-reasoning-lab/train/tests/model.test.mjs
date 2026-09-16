import assert from 'node:assert/strict';
import {examples,forward,gradients,getRef,setRef,gradRef,resetParams,trainPair,state,setParams,surfaceRefs,vocab} from '../model.js';
resetParams();const e=examples[1],f=forward(e.context,e.target),initialState=JSON.stringify(state());resetParams();assert.equal(JSON.stringify(state()),initialState);assert.equal(f.probs.length,vocab.length);assert.ok(Math.abs(f.probs.reduce((a,b)=>a+b,0)-1)<1e-12);assert.ok(Math.abs(f.loss+Math.log(f.probs[f.y]))<1e-12);
const {g}=gradients(e.context,e.target);const refs=[{group:'W',idx:[0,0]},{group:'O',idx:[2,3]},{group:'E',idx:[vocab.indexOf(e.context[0]),1]},{group:'b',idx:[2]},{group:'c',idx:[4]}];for(const r of refs){const x=getRef(r),eps=1e-5;setRef(r,x+eps);const p=forward(e.context,e.target).loss;setRef(r,x-eps);const m=forward(e.context,e.target).loss;setRef(r,x);const numeric=(p-m)/(2*eps),analytic=gradRef(g,r);assert.ok(Math.abs(numeric-analytic)<2e-5,`${r.group} ${r.idx}: ${numeric} != ${analytic}`)}
const before=forward(e.context,e.target).loss;for(let i=0;i<80;i++)trainPair(e.context,e.target,.1);const after=forward(e.context,e.target).loss;assert.ok(after<before*.3,`${after} !< ${before}`);
const saved=state(),old=forward(e.context,e.target).loss;trainPair(e.context,e.target,.1);setParams(saved);assert.ok(Math.abs(forward(e.context,e.target).loss-old)<1e-12);
const pair=surfaceRefs({group:'W',idx:[1,1]});assert.notDeepEqual(pair.a,pair.b);
console.log(JSON.stringify({ok:true,probSum:f.probs.reduce((a,b)=>a+b,0),initialLoss:before,trainedLoss:after,gradientChecks:refs.length},null,2));
