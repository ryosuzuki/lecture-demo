import * as THREE from 'three';
import {OrbitControls} from '../vendor/OrbitControls.js';
// Independent Q/K space: rotating it never tilts the HTML arithmetic or the Value view.
export function queryScene(host){
 const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setClearColor('#000');host.append(renderer.domElement);
 const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(37,1,.1,100),controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=false;controls.minDistance=3;controls.maxDistance=30;
 const grid=new THREE.GridHelper(6,6,0x556677,0x253039);scene.add(grid);
 for(let i=0;i<3;i++){const a=new THREE.Vector3(),b=new THREE.Vector3();a.setComponent(i,-3);b.setComponent(i,3);scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([a,b]),new THREE.LineBasicMaterial({color:0x666666})));}
 let group=new THREE.Group();scene.add(group);let labels=[];
 function reset(){camera.position.set(5,4,10);controls.target.set(.6,.6,.2);controls.update();}
 function arrow(v,color,name){const end=new THREE.Vector3(...v);if(end.length()>1e-6)group.add(new THREE.ArrowHelper(end.clone().normalize(),new THREE.Vector3(),end.length(),color,.18,.09));const label=document.createElement('span');label.className='qk-label';label.style.color=color;label.textContent=name;host.append(label);labels.push({label,end});}
 function update(q,k,receiver,source,score){group.traverse(o=>{o.geometry?.dispose();o.material?.dispose()});scene.remove(group);group=new THREE.Group();scene.add(group);labels.forEach(l=>l.label.remove());labels=[];arrow(q,'#ffff62','Q('+receiver+')');arrow(k,'#5cd0b3','K('+source+')');
  const qq=q.reduce((a,x)=>a+x*x,0);if(qq>1e-8){const projected=q.map(x=>x*score/qq);const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...k),new THREE.Vector3(...projected)]),new THREE.LineDashedMaterial({color:0x888888,dashSize:.1,gapSize:.07}));line.computeLineDistances();group.add(line);}
 }
 function render(){const r=host.getBoundingClientRect();if(!r.width||!r.height)return;const size=new THREE.Vector2();renderer.getSize(size);if(size.x!==r.width||size.y!==r.height){renderer.setSize(r.width,r.height,false);camera.aspect=r.width/r.height;camera.updateProjectionMatrix();}controls.update();renderer.render(scene,camera);labels.forEach(({label,end},i)=>{const p=end.clone().project(camera);label.style.display=Math.abs(p.x)>1||Math.abs(p.y)>1||p.z>1?'none':'';label.style.left=Math.max(4,Math.min(r.width-label.offsetWidth-5,(p.x+1)*r.width/2+8))+'px';label.style.top=Math.max(4,Math.min(r.height-30,(1-p.y)*r.height/2+(i?14:-22)))+'px';});}
 reset();return {update,render,reset,camera};
}
