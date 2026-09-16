import assert from 'node:assert/strict';
import {examples,vocab,params,forward,gradients,resetParams,getRef,setRef,gradRef,state} from '../train/model.js';
resetParams();
const ex=examples[0],y=vocab.indexOf(ex.target),w={group:'O',idx:[0,y]},b={group:'c',idx:[y]};
const beforeState=state(),before=forward(ex.context,ex.target),g=gradients(ex.context,ex.target).g;
for(const ref of [w,b]){
  const old=getRef(ref),eps=1e-4;
  setRef(ref,old+eps);const plus=forward(ex.context,ex.target).loss;
  setRef(ref,old-eps);const minus=forward(ex.context,ex.target).loss;
  setRef(ref,old);
  assert.ok(Math.abs((plus-minus)/(2*eps)-gradRef(g,ref))<1e-6,'analytic gradient matches finite difference');
}
const lr=.35;setRef(w,getRef(w)-lr*gradRef(g,w));setRef(b,getRef(b)-lr*gradRef(g,b));
const after=forward(ex.context,ex.target),afterState=state();
assert.ok(after.loss<before.loss,'one two-parameter SGD step lowers chosen-example loss');
assert.ok(after.probs[after.y]>before.probs[before.y],'target probability rises');
const changed=[];for(const group of ['E','W','O','b','c'])JSON.stringify(beforeState[group])!==JSON.stringify(afterState[group])&&changed.push(group);
assert.deepEqual(changed,['O','c'],'only output matrix and bias groups changed');
let scalarChanges=0;for(let j=0;j<beforeState.O.length;j++)for(let k=0;k<beforeState.O[j].length;k++)scalarChanges+=beforeState.O[j][k]!==afterState.O[j][k];for(let k=0;k<beforeState.c.length;k++)scalarChanges+=beforeState.c[k]!==afterState.c[k];
assert.equal(scalarChanges,2,'exactly two scalar parameters changed');
console.log(JSON.stringify({beforeLoss:before.loss,afterLoss:after.loss,beforeProbability:before.probs[before.y],afterProbability:after.probs[after.y],gradients:[gradRef(g,w),gradRef(g,b)],scalarChanges},null,2));
