// Coordinate slice of this same model's loss. All non-selected parameters stay fixed.
export function sampleLandscape(model,tokens,target,indexA,indexB,{radius=2,size=25,center=null}={}){
 if(indexA===indexB)throw Error('Choose two different parameters');
 const a=model.params[indexA],b=model.params[indexB],original=[a.value,b.value];center=center||original;
 const points=[];try{for(let i=0;i<size;i++){const row=[];a.value=center[0]-radius+2*radius*i/(size-1);for(let j=0;j<size;j++){b.value=center[1]-radius+2*radius*j/(size-1);row.push({x:a.value,z:b.value,loss:model.forward(tokens,target).loss})}points.push(row)}}finally{a.value=original[0];b.value=original[1]}
 return{points,center:[...center],radius,size,axes:[a.name,b.name],current:{x:a.value,z:b.value,loss:model.forward(tokens,target).loss}};
}
export function coordinateStep(model,tokens,target,indexA,indexB,lr){
 if(indexA===indexB)throw Error('Choose two different parameters');
 const before=model.backward(tokens,target),a=model.params[indexA],b=model.params[indexB],gradient=[a.grad,b.grad];a.value-=lr*a.grad;b.value-=lr*b.grad;model.steps++;return{before,after:model.forward(tokens,target),gradient,position:[a.value,b.value]};
}
