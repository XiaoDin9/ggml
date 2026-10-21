/* Shared by the standalone HTML and SVG; no DOM or external dependencies. */
(() => {
  'use strict';
  const W=360,H=130,DX=440,DY=152;
  function index(edges){
    const result=new Map();
    for(const edge of edges){if(!result.has(edge.source))result.set(edge.source,[]);result.get(edge.source).push(edge);}
    for(const list of result.values())list.sort((a,b)=>a.site.line-b.site.line||a.id.localeCompare(b.id,undefined,{numeric:true}));
    return result;
  }
  const rootKey=id=>'root:'+id;
  function occurrence(id,key,parent,edge,ancestors,out){
    const recursive=ancestors.includes(id);
    return {id,key,parent,edge,depth:ancestors.length,recursive,count:recursive?0:(out.get(id)||[]).length,children:[],omitted:0,w:W,h:H};
  }
  function build(rootId,expanded,out,limit=1200){
    let count=0;
    function visit(id,key,parent,edge,ancestors){
      const node=occurrence(id,key,parent,edge,ancestors,out);count++;
      if(!node.recursive&&expanded.has(key)){
        const edges=out.get(id)||[];
        for(let i=0;i<edges.length;i++){
          if(count>=limit){node.omitted=edges.length-i;break;}
          const e=edges[i];node.children.push(visit(e.target,key+'/'+e.id,node,e,[...ancestors,id]));
        }
      }
      return node;
    }
    return visit(rootId,rootKey(rootId),null,null,[]);
  }
  function flatten(root){const nodes=[];function walk(n){nodes.push(n);n.children.forEach(walk);}walk(root);return nodes;}
  function layout(root){
    function measure(n){n.span=n.children.length?Math.max(H,n.children.reduce((s,c)=>s+measure(c),0)+(n.children.length-1)*(DY-H)):H;return n.span;}
    function place(n,top){n.x=60+n.depth*DX;n.y=top+(n.span-H)/2;let next=top;for(const child of n.children){place(child,next);next+=child.span+(DY-H);}}
    measure(root);place(root,80);return flatten(root);
  }
  function path(root,target,out){
    if(root===target)return [];
    const queue=[root],parents=new Map([[root,null]]);
    for(let i=0;i<queue.length;i++){
      for(const edge of out.get(queue[i])||[]){
        if(parents.has(edge.target))continue;parents.set(edge.target,edge);queue.push(edge.target);
        if(edge.target===target){const result=[];let current=target;while(current!==root){const e=parents.get(current);result.push(e);current=e.source;}return result.reverse();}
      }
    }
    return null;
  }
  function fromPath(rootId,edges,out){
    const root=occurrence(rootId,rootKey(rootId),null,null,[],out);let current=root;const ancestors=[rootId];
    for(const edge of edges){const next=occurrence(edge.target,current.key+'/'+edge.id,current,edge,ancestors,out);current.children.push(next);current=next;ancestors.push(next.id);}
    return root;
  }
  function chainLayout(root){
    const nodes=flatten(root),columns=Math.max(1,Math.ceil(nodes.length/4)),rows=Math.ceil(nodes.length/columns);
    nodes.forEach((n,i)=>{const col=Math.floor(i/rows),step=i%rows,row=col%2?rows-1-step:step;n.x=60+col*DX;n.y=80+row*178;});
    return nodes;
  }
  function route(parent,child,chain=false){
    if(chain&&parent.x===child.x){
      const down=child.y>parent.y,x=parent.x+W/2,y1=parent.y+(down?H:0),y2=child.y+(down?0:H);
      return `M ${x} ${y1} L ${x} ${y2}`;
    }
    const x1=parent.x+W,y1=parent.y+H/2,x2=child.x,y2=child.y+H/2,m=(x1+x2)/2;
    return `M ${x1} ${y1} C ${m} ${y1},${m} ${y2},${x2} ${y2}`;
  }
  function expandPath(root,edges,expanded){let key=rootKey(root);for(const edge of edges){expanded.add(key);key+='/'+edge.id;}return key;}
  function collapse(key,expanded){for(const candidate of [...expanded])if(candidate===key||candidate.startsWith(key+'/'))expanded.delete(candidate);}
  globalThis.CallTree={W,H,DX,DY,index,rootKey,build,flatten,layout,path,fromPath,chainLayout,route,expandPath,collapse};
})();
