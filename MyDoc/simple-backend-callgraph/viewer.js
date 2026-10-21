'use strict';
const DATA=JSON.parse(document.getElementById('data').textContent);
const $=id=>document.getElementById(id),NS='http://www.w3.org/2000/svg',T=CallTree;
const ALL={...DATA.functions,...DATA.nodes},OUT=T.index(DATA.edges),viewCache=new Map();
const state={view:DATA.views[0],root:'main',expanded:new Set(),mode:'tree',chain:[],selected:'main',selectedKey:T.rootKey('main'),edge:null,edgeKey:null,scale:1,x:0,y:0,history:[],drag:null,nodes:[]};
let resizeTimer;
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function svg(tag,attrs={},children=[]){const e=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);for(const c of children)e.append(c);return e;}
function text(x,y,value,attrs={}){const e=svg('text',{x,y,...attrs});e.textContent=value;return e;}
function toast(message){$('toast').textContent=message;$('toast').style.display='block';clearTimeout(toast.timer);toast.timer=setTimeout(()=>$('toast').style.display='none',2500);}
function codeURI(path,line){return 'vscode://file/'+encodeURI(DATA.root+'/'+path).replace(/#/g,'%23').replace(/\?/g,'%3F')+':'+line+':1';}
function openCode(node){window.location.href=node.code;}
function applyTransform(){$('world').setAttribute('transform',`translate(${state.x} ${state.y}) scale(${state.scale})`);$('zoom').textContent=Math.round(state.scale*100)+'%';}
function bounds(){return {w:Math.max(...state.nodes.map(n=>n.x+n.w))+60,h:Math.max(...state.nodes.map(n=>n.y+n.h))+60};}
function fit(){const r=$('graph').getBoundingClientRect(),b=bounds();state.scale=Math.max(.045,Math.min(1,(r.width-36)/b.w,(r.height-140)/b.h));state.x=(r.width-b.w*state.scale)/2;state.y=90+(r.height-145-b.h*state.scale)/2;applyTransform();}
function zoom(factor,cx,cy){const r=$('graph').getBoundingClientRect();cx=cx??r.width/2;cy=cy??r.height/2;const s=Math.max(.04,Math.min(2.5,state.scale*factor));state.x=cx-(cx-state.x)*s/state.scale;state.y=cy-(cy-state.y)*s/state.scale;state.scale=s;applyTransform();}
function wrapName(name){if(name.length<=39)return[name];const cut=name.lastIndexOf('_',39);const n=cut>16?cut+1:39;return[name.slice(0,n),name.slice(n)];}
function occurrence(key=state.selectedKey){return state.nodes.find(n=>n.key===key);}
function renderGraph(){
  const root=state.mode==='chain'?T.fromPath(state.root,state.chain,OUT):T.build(state.root,state.expanded,OUT);
  state.nodes=state.mode==='chain'?T.chainLayout(root):T.layout(root);
  $('edges').replaceChildren();$('nodes').replaceChildren();
  for(const n of state.nodes){
    if(!n.parent)continue;
    const e=n.edge,color=e.kind==='callback'?'#b79be9':e.kind==='direct'?'#7e97b3':'#d7b075',d=T.route(n.parent,n,state.mode==='chain');
    const g=svg('g',{'class':'edge','data-edge':e.id,'data-child':n.key,tabindex:0,role:'link','aria-label':`${e.source} → ${e.target}`});
    const title=svg('title');title.textContent=`${e.source} → ${e.target}\n${e.label}\n${e.note}\n单击查看被调实现；双击在 VS Code 打开`;g.append(title);
    g.append(svg('path',{d,'class':'hit'}),svg('path',{d,'class':'wire',stroke:color,'stroke-dasharray':e.kind==='direct'?'none':'7 5','marker-end':e.kind==='callback'?'url(#arrow-callback)':'url(#arrow)'}));
    if(e.kind!=='direct'||e.note){
      const p=n.parent,vertical=state.mode==='chain'&&p.x===n.x;
      const label=e.kind==='callback'?'接口':e.kind==='phase'?'阶段':e.kind==='lifecycle'?'首次':e.note==='此源码阶段内的调用'?'调用':'条件';
      const x=vertical?p.x+p.w/2+8:(p.x+p.w+n.x)/2-12;
      const y=vertical?(p.y+p.h/2+n.y+n.h/2)/2:(p.y+p.h/2+n.y+n.h/2)/2-6;
      g.append(text(x,y,label,{'class':'label'}));
    }
    g.addEventListener('click',()=>selectEdge(e,n.key));g.addEventListener('dblclick',ev=>{ev.preventDefault();openCode(ALL[n.id]);});
    g.addEventListener('keydown',ev=>{if(ev.key==='Enter'){ev.preventDefault();selectEdge(e,n.key);}});$('edges').append(g);
  }
  for(const[nIndex,n]of state.nodes.entries()){
    const fn=DATA.nodes[n.id],color=DATA.groups[fn.group].color,open=state.mode==='tree'&&state.expanded.has(n.key);
    const g=svg('g',{'class':'node','data-node':n.id,'data-key':n.key,transform:`translate(${n.x} ${n.y})`,tabindex:0,role:'link','aria-label':fn.name});
    const title=svg('title');title.textContent=`${fn.name}\n${fn.note}\n${fn.path}:${fn.line}\n${n.recursive?'递归回到祖先，停止展开':'点击查看源码；＋ 展开下一层'}`;g.append(title);
    const body=svg('g',{'class':'node-body'});
    body.append(svg('rect',{width:n.w,height:n.h,rx:9,fill:'#1b2635',stroke:'#3a4a61','class':'box'}),svg('rect',{x:0,y:12,width:4,height:n.h-24,rx:2,fill:color}));
    wrapName(fn.name).forEach((line,i)=>body.append(text(14,25+i*18,line,{fill:color,'font-family':'Consolas,monospace','font-size':13})));
    body.append(text(14,69,fn.note.length>26?fn.note.slice(0,25)+'…':fn.note,{fill:'#c7d4e4','font-size':12}));
    body.append(text(14,89,fn.path.split('/').pop()+':'+fn.line,{fill:'#8fa4bd','font-family':'Consolas,monospace','font-size':11}));
    if(n.recursive)body.append(text(14,116,'↩ 递归回到祖先 · 停止展开',{fill:'#d7b075','font-size':11}));
    else if(!n.count)body.append(text(14,116,'图中已到末端',{fill:'#7189a4','font-size':11}));
    if(state.mode==='chain')body.append(text(n.w-56,116,String(nIndex+1).padStart(2,'0')+' / '+state.nodes.length,{fill:'#abbed4','font-size':10}));
    g.append(body);
    if(n.count){
      const toggle=svg('g',{'class':'tree-toggle','data-toggle':n.key,tabindex:0,role:'button','aria-expanded':open?'true':'false','aria-label':(open?'收起 ':'展开 ')+fn.name});
      toggle.append(svg('rect',{x:12,y:99,width:145,height:23,rx:5,fill:open?'#34445a':'#243c54',stroke:'#516c89'}));
      toggle.append(text(21,115,open?'− 收起此分支':`＋ ${n.count} 个调用`,{fill:'#c8e2fb','font-size':11}));
      const handler=ev=>{ev.stopPropagation();toggleNode(n.key);};toggle.addEventListener('click',handler);toggle.addEventListener('dblclick',ev=>ev.stopPropagation());
      toggle.addEventListener('keydown',ev=>{if(ev.key==='Enter'||ev.key===' '){ev.preventDefault();handler(ev);}});g.append(toggle);
    }
    if(n.omitted)g.append(text(164,116,'请收起其它分支再展开',{fill:'#e0b77a','font-size':10}));
    g.addEventListener('click',()=>selectFunction(n.id,{occurrence:n.key}));g.addEventListener('dblclick',ev=>{ev.preventDefault();openCode(fn);});
    g.addEventListener('keydown',ev=>{if(ev.target!==g)return;if(ev.key==='Enter')selectFunction(n.id,{occurrence:n.key});if(ev.key==='ArrowRight'&&!open)toggleNode(n.key);if(ev.key==='ArrowLeft'&&open)toggleNode(n.key);});
    $('nodes').append(g);
  }
  $('stats').textContent=state.mode==='chain'?`完整调用链 · ${state.nodes.length} 个沿途节点 · 只显示这一条路径`:`调用树 · ${state.nodes.length} 个可见节点 · ＋ 展开下一层；同名节点按分支独立显示`;
  $('return-tree').disabled=state.mode!=='chain';
  $('chain-hint').textContent=state.mode==='chain'?'显示 main 到目标的一条最短路径；沿途条件仍需满足。':'也可选中任意函数，点击右侧「main → 此函数」。';
  updateHighlight();applyTransform();
}
function updateHighlight(){
  for(const el of $('nodes').children)el.classList.toggle('selected',el.dataset.key===state.selectedKey);
  for(const el of $('edges').children)el.classList.toggle('selected',el.dataset.child===state.edgeKey);
}
function activateView(){for(const b of $('views').children)b.classList.toggle('active',b.dataset.view===state.view.id);}
function setView(id,doFit=true){
  viewCache.set(state.view.id,{root:state.root,expanded:new Set(state.expanded)});
  state.view=DATA.views.find(v=>v.id===id)||DATA.views[0];const cached=viewCache.get(state.view.id);
  state.root=cached?.root||state.view.root;state.expanded=cached?.expanded||new Set();state.mode='tree';state.chain=[];state.selectedKey=T.rootKey(state.root);state.edgeKey=null;
  activateView();renderGraph();selectFunction(state.root,{silent:true,occurrence:state.selectedKey});if(doFit)requestAnimationFrame(fit);
}
function positionNode(n,minimumScale=false){
  const r=$('graph').getBoundingClientRect();if(minimumScale)state.scale=Math.max(.65,Math.min(1,state.scale));
  state.x=Math.max(50,r.width*.12)-n.x*state.scale;state.y=r.height/2-(n.y+n.h/2)*state.scale;applyTransform();
}
function changeTree(anchor,change){
  const before=occurrence(anchor),screen=before?{x:state.x+before.x*state.scale,y:state.y+before.y*state.scale}:null;
  change();renderGraph();const after=occurrence(anchor);
  if(screen&&after){state.x=screen.x-after.x*state.scale;state.y=screen.y-after.y*state.scale;applyTransform();}
}
function toggleNode(key){
  let n=occurrence(key);if(!n||!n.count)return;
  const wasChain=state.mode==='chain',wasOnlyRoot=state.nodes.length===1,opening=!state.expanded.has(key);
  if(wasChain){T.expandPath(state.root,state.chain,state.expanded);state.mode='tree';state.expanded.add(key);renderGraph();n=occurrence(key);positionNode(n,true);}
  else {
    changeTree(key,()=>{if(state.expanded.has(key))T.collapse(key,state.expanded);else state.expanded.add(key);});
    const current=occurrence(key),r=$('graph').getBoundingClientRect();
    if(wasOnlyRoot&&opening){state.scale=Math.min(.85,Math.max(.55,(r.width-80)/(T.W*2+80)));state.x=30-60*state.scale;state.y=100-Math.min(...state.nodes.map(n=>n.y))*state.scale;applyTransform();}
    else if(opening&&state.x+(current.x+T.W+T.DX)*state.scale>r.width)positionNode(current,true);
  }
  selectFunction(n.id,{silent:true,occurrence:key});
}
function expandLevel(){
  if(state.mode==='chain'){returnTree();return;}
  const anchor=occurrence()?.key||T.rootKey(state.root),leaves=state.nodes.filter(n=>n.count&&!state.expanded.has(n.key));
  if(!leaves.length){toast('当前可见末端均已展开。');return;}
  changeTree(anchor,()=>leaves.forEach(n=>state.expanded.add(n.key)));
  if(state.nodes.length<18)fit();
}
function collapseBranch(){const n=occurrence();if(!n)return;if(state.mode==='chain')returnTree();changeTree(n.key,()=>T.collapse(n.key,state.expanded));selectFunction(n.id,{silent:true,occurrence:n.key});}
function collapseAll(){state.mode='tree';state.chain=[];state.expanded.clear();state.selectedKey=T.rootKey(state.root);renderGraph();selectFunction(state.root,{silent:true,occurrence:state.selectedKey});fit();}
function findPath(id,root=state.root){return T.path(root,id,OUT);}
function openChain(id){
  const path=findPath(id,'main');if(path===null){toast('此辅助函数未纳入 main 调用树；源码可在右侧阅读。');return false;}
  viewCache.set(state.view.id,{root:state.root,expanded:new Set(state.expanded)});
  state.view=DATA.views[0];state.root='main';state.mode='chain';state.chain=path;state.edgeKey=null;
  state.selectedKey=T.expandPath('main',path,state.expanded);activateView();renderGraph();
  selectFunction(id,{silent:true,occurrence:state.selectedKey});requestAnimationFrame(fit);return true;
}
function returnTree(){if(state.mode!=='chain')return;const key=T.expandPath(state.root,state.chain,state.expanded);state.mode='tree';state.selectedKey=key;renderGraph();const n=occurrence(key);if(n)positionNode(n,true);}
function reveal(id){
  const existing=state.nodes.find(n=>n.id===id&&n.key===state.selectedKey)||state.nodes.find(n=>n.id===id);
  if(existing)return existing;
  let path=findPath(id);
  if(path===null){path=findPath(id,'main');if(path===null)return null;state.view=DATA.views[0];state.root='main';state.expanded=new Set();activateView();}
  state.mode='tree';const key=T.expandPath(state.root,path,state.expanded);renderGraph();return occurrence(key);
}
function locate(id=state.selected){const n=reveal(id);if(!n){toast('图外辅助实现已显示在右侧。');return;}state.selectedKey=n.key;positionNode(n,true);updateHighlight();}
function selectEdge(edge,key){
  if(!key){
    let parent=occurrence();if(parent?.id!==edge.source)parent=state.nodes.find(n=>n.id===edge.source)||reveal(edge.source);
    if(parent&&!parent.recursive){if(state.mode==='chain'){T.expandPath(state.root,state.chain,state.expanded);state.mode='tree';}state.expanded.add(parent.key);renderGraph();key=parent.key+'/'+edge.id;}
  }
  selectFunction(edge.target,{edge,occurrence:key});
}
function highlight(line){const el=$('source-code').querySelector(`[data-line="${line}"]`);if(el){el.classList.add('marked');const wrap=$('source-wrap');wrap.scrollTop=Math.max(0,el.offsetTop-$('source-code').offsetTop-wrap.clientHeight/3);}}
function renderSource(node,line){
  let inComment=false;const keywords=new Set(['return','if','else','for','while','switch','case','break','continue','static','const','struct','enum','void','int','bool','size_t','true','false','NULL','nullptr','new','delete','auto','sizeof']);
  function syntax(s){let result='',i=0;while(i<s.length){
    if(inComment){const end=s.indexOf('*/',i),stop=end<0?s.length:end+2;result+='<span class="comment">'+esc(s.slice(i,stop))+'</span>';i=stop;if(end>=0)inComment=false;continue;}
    if(s.startsWith('//',i)){result+='<span class="comment">'+esc(s.slice(i))+'</span>';break;}
    if(s.startsWith('/*',i)){inComment=true;continue;}
    if(s[i]==='"'||s[i]==="'"){const quote=s[i];let end=i+1;while(end<s.length){if(s[end]==='\\'){end+=2;continue;}if(s[end++]===quote)break;}result+='<span class="str">'+esc(s.slice(i,end))+'</span>';i=end;continue;}
    const match=/^[A-Za-z_]\w*/.exec(s.slice(i));if(match){const token=match[0];if(ALL[token])result+=`<a href="#fn=${encodeURIComponent(token)}" class="fn-link" data-function="${esc(token)}">${esc(token)}</a>`;else if(keywords.has(token))result+='<span class="kw">'+token+'</span>';else result+=esc(token);i+=token.length;continue;}result+=esc(s[i++]);
  }return result;}
  $('source-code').innerHTML=node.source.split('\n').map((s,i)=>{const no=node.line+i;return `<div class="source-line" data-line="${no}"><a class="line-number" href="${esc(codeURI(node.path,no))}" title="VS Code 打开此行">${no}</a><code class="source-text">${syntax(s)}</code></div>`;}).join('');$('source-wrap').scrollTop=0;if(line)requestAnimationFrame(()=>highlight(line));
}
function renderRelations(node){
  const edges=OUT.get(node.id)||[];$('relation-summary').textContent=`下一层调用 / 内部阶段（${edges.length}）`;$('relation-items').replaceChildren();
  for(const e of edges){const b=document.createElement('button');b.innerHTML=esc(ALL[e.target].name)+`<br><small>${esc(e.kind==='callback'?e.label:e.kind==='phase'?'函数内部阶段':e.kind==='lifecycle'?'首次初始化':'直接调用')}${e.note?' · '+esc(e.note):''}</small>`;b.onclick=()=>selectEdge(e);$('relation-items').append(b);}
  if(!edges.length)$('relation-items').innerHTML='<div class="subtitle">图中未继续展开；可点击源码中的函数名。</div>';
}
function selectFunction(id,opts={}){
  const node=ALL[id];if(!node)return;
  if(!opts.noHistory&&(state.selected!==id||state.selectedKey!==opts.occurrence))state.history.push({id:state.selected,key:state.selectedKey,edge:state.edge,edgeKey:state.edgeKey});
  state.selected=id;state.selectedKey=opts.occurrence||state.nodes.find(n=>n.id===id)?.key||null;state.edge=opts.edge?.id||null;state.edgeKey=opts.edge?state.selectedKey:null;
  $('node-group').textContent=node.kind==='phase'?'函数内部阶段 · 源码块':(DATA.groups[node.group]?.title||'辅助实现')+' · 完整实现';
  $('node-name').textContent=node.name;$('node-note').textContent=node.note;$('source-path').textContent=node.path+':'+node.line+' — '+node.endLine;$('source-path').href=node.code;$('open-code').href=node.code;
  $('back').disabled=!state.history.length;$('locate').disabled=!DATA.nodes[id];$('show-chain').disabled=!DATA.nodes[id];$('call-info').hidden=!opts.edge;
  if(opts.edge)renderCallInfo(opts.edge);renderSource(node,opts.line);renderRelations(node);if(opts.reveal)locate(id);updateHighlight();
  try{history.replaceState(null,'','#'+new URLSearchParams({fn:id,view:state.view.id,mode:state.mode}));}catch{}
  if(!opts.silent&&$('auto-open').checked)openCode(node);
}
function renderCallInfo(edge){
  const src=ALL[edge.source],dst=ALL[edge.target];$('call-info').innerHTML=`<div class="smallcaps">${edge.kind==='callback'?'函数指针分派':edge.kind==='phase'?'函数内部阶段':edge.kind==='lifecycle'?'首次初始化':'函数调用'}</div><p>${esc(src.name)} → <strong>${esc(dst.name)}</strong></p><p>${esc(edge.note||'目标实现已显示在下方。')}</p><button id="show-caller">查看调用处 · L${edge.site.line}</button><a href="${esc(dst.code)}">VS Code 打开被调实现 ↗</a>`;
  const current=occurrence();$('show-caller').onclick=()=>selectFunction(src.id,{line:edge.site.line,silent:true,occurrence:current?.parent?.key});
  if(edge.sites.length>1){const p=document.createElement('p');p.textContent='调用行：'+edge.sites.join('、');$('call-info').append(p);}
  if(edge.binding){const b=document.createElement('button');b.textContent='查看接口绑定 · L'+edge.binding.line;b.onclick=()=>showFileLine(edge.binding.path,edge.binding.line);$('call-info').append(b);const a=document.createElement('a');a.href=edge.binding.code;a.textContent='VS Code 打开绑定 ↗';$('call-info').append(a);}
}
function showFileLine(path,line){const known=Object.values(ALL).find(n=>n.path===path&&n.line<=line&&n.endLine>=line);if(known){selectFunction(known.id,{line,silent:true});return;}const lines=DATA.files[path].text.split('\n'),first=Math.max(1,line-10),last=Math.min(lines.length,line+14),id='file-region';ALL[id]={id,name:'接口表绑定',path,line:first,endLine:last,source:lines.slice(first-1,last).join('\n'),code:codeURI(path,line),group:'aux',kind:'phase',note:'函数指针的赋值位置；此处是接口表数据。'};selectFunction(id,{line,silent:true});}
function search(){
  const query=$('search').value.trim().toLowerCase();$('results').replaceChildren();if(!query)return;
  const matches=Object.values(ALL).filter(n=>!n.id.includes('@')&&(n.name+' '+n.path+' '+n.note).toLowerCase().includes(query));matches.sort((a,b)=>(a.name===query?-1:b.name===query?1:0)||Number(!!DATA.nodes[b.id])-Number(!!DATA.nodes[a.id])||a.name.localeCompare(b.name));
  for(const n of matches.slice(0,35)){const b=document.createElement('button');b.className='result';b.setAttribute('role','option');b.innerHTML=`<strong>${esc(n.name)}</strong><small>${esc(n.path)}:${n.line}${DATA.nodes[n.id]?' · 自动展开到此':' · 辅助实现'}</small>`;b.onclick=()=>{selectFunction(n.id,{reveal:true});$('results').replaceChildren();$('search').value='';};$('results').append(b);}
  if(!matches.length)$('results').innerHTML='<div class="empty">没有匹配的已索引实现。</div>';
}
function exportSVG(download=true){
  const clone=$('graph').cloneNode(true),b=bounds();clone.removeAttribute('style');clone.setAttribute('width',b.w);clone.setAttribute('height',b.h);clone.setAttribute('viewBox',`0 0 ${b.w} ${b.h}`);clone.querySelector('#world').removeAttribute('transform');
  const background=svg('rect',{width:b.w,height:b.h,fill:'#10151d'});clone.insertBefore(background,clone.firstChild);
  const title=text(60,32,state.mode==='chain'?'simple-backend · 完整调用链':'simple-backend · 当前展开的调用树',{fill:'#e6edf8','font-size':18});clone.insertBefore(title,clone.querySelector('#world'));
  for(const n of state.nodes){const group=[...clone.querySelectorAll('[data-key]')].find(g=>g.dataset.key===n.key),a=svg('a',{href:ALL[n.id].code});a.append(group.querySelector('.node-body'));group.prepend(a);group.querySelector('.tree-toggle')?.remove();}
  for(const group of clone.querySelectorAll('[data-child]')){const e=DATA.edges.find(e=>e.id===group.dataset.edge),a=svg('a',{href:ALL[e.target].code});while(group.firstChild)a.append(group.firstChild);group.append(a);}
  const style=svg('style');style.textContent='.wire{fill:none;stroke-width:1.6}.hit{fill:none;stroke:transparent;stroke-width:14}.label{fill:#a9bdd6;font:10px Consolas,monospace;paint-order:stroke;stroke:#10151d;stroke-width:4px}';clone.insertBefore(style,clone.firstChild);
  const content=new XMLSerializer().serializeToString(clone);
  if(download){const url=URL.createObjectURL(new Blob([content],{type:'image/svg+xml'})),a=document.createElement('a');a.href=url;a.download='callgraph-current.svg';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);toast('已导出当前展开状态，保留源码链接。');}return content;
}
for(const view of DATA.views){const b=document.createElement('button');b.textContent=view.title;b.dataset.view=view.id;b.onclick=()=>setView(view.id);$('views').append(b);}
for(const chain of DATA.chains){const o=document.createElement('option');o.value=chain.target;o.textContent=chain.title;$('chain-target').append(o);}
$('open-chain').onclick=()=>openChain($('chain-target').value);$('return-tree').onclick=returnTree;$('show-chain').onclick=()=>openChain(state.selected);
$('expand-level').onclick=expandLevel;$('collapse-branch').onclick=collapseBranch;$('collapse-all').onclick=collapseAll;$('fit').onclick=fit;$('zoom-in').onclick=()=>zoom(1.25);$('zoom-out').onclick=()=>zoom(.8);$('export-svg').onclick=()=>exportSVG();$('locate').onclick=()=>locate();
$('help').onclick=()=>$('help-dialog').showModal();$('close-help').onclick=()=>$('help-dialog').close();
$('back').onclick=()=>{const item=state.history.pop();if(item)selectFunction(item.id,{noHistory:true,silent:true,occurrence:item.key,edge:DATA.edges.find(e=>e.id===item.edge)});};
$('copy-location').onclick=async()=>{const n=ALL[state.selected],value=DATA.root+'/'+n.path+':'+n.line;try{await navigator.clipboard.writeText(value);toast('已复制源码位置。');}catch{prompt('复制源码位置',value);}};
$('search').addEventListener('input',search);$('search').addEventListener('keydown',e=>{if(e.key==='Escape')$('results').replaceChildren();if(e.key==='Enter')$('results').querySelector('button')?.click();});document.addEventListener('click',e=>{if(!e.target.closest('.search'))$('results').replaceChildren();});
$('source-code').addEventListener('click',e=>{const a=e.target.closest('[data-function]');if(a){e.preventDefault();const edge=(OUT.get(state.selected)||[]).find(e=>e.target===a.dataset.function);if(edge)selectEdge(edge);else selectFunction(a.dataset.function);}});
$('graph').addEventListener('wheel',e=>{e.preventDefault();const r=$('graph').getBoundingClientRect();zoom(Math.exp(-e.deltaY*.0015),e.clientX-r.left,e.clientY-r.top);},{passive:false});
$('graph').addEventListener('pointerdown',e=>{if(e.button!==0||e.target.closest('.node,.edge'))return;state.drag={x:e.clientX,y:e.clientY,ox:state.x,oy:state.y};$('graph').setPointerCapture(e.pointerId);$('graph').classList.add('dragging');});
$('graph').addEventListener('pointermove',e=>{if(!state.drag)return;state.x=state.drag.ox+e.clientX-state.drag.x;state.y=state.drag.oy+e.clientY-state.drag.y;applyTransform();});
function endDrag(){state.drag=null;$('graph').classList.remove('dragging');}$('graph').addEventListener('pointerup',endDrag);$('graph').addEventListener('pointercancel',endDrag);
$('graph').addEventListener('keydown',e=>{if(e.key==='0')fit();if(e.key==='+'||e.key==='=')zoom(1.2);if(e.key==='-')zoom(1/1.2);});
$('divider').addEventListener('pointerdown',e=>{if(innerWidth<=700)return;e.preventDefault();const start=e.clientX,width=$('inspector').getBoundingClientRect().width;function move(event){$('inspector').style.width=Math.max(320,Math.min(innerWidth*.75,width+start-event.clientX))+'px';}function done(){document.removeEventListener('pointermove',move);document.removeEventListener('pointerup',done);}document.addEventListener('pointermove',move);document.addEventListener('pointerup',done);});
new ResizeObserver(()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(fit,100);}).observe($('graph'));
const hash=new URLSearchParams(location.hash.slice(1)),initial=hash.get('fn');setView(hash.get('view')||'overview');
if(initial&&ALL[initial]){if(hash.get('mode')==='chain')openChain(initial);else selectFunction(initial,{silent:true,reveal:!!DATA.nodes[initial],noHistory:true});}
window.callgraph={data:DATA,state,selectFunction,selectEdge,setView,locate,fit,showFileLine,toggleNode,expandLevel,collapseAll,openChain,returnTree,exportSVG};
