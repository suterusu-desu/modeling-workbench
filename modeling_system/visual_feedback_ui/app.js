'use strict';
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let board = null, regionId = null, reviewId = null, revising = null, dragStart = null;
let feedbackDraft = null;
let motionDraft = null, motionClipId = null, motionRegion = null;
let workflow = null;
const reviewViews = new Map();
const rows = collection => board?.[collection] || [];
const record = id => Object.values(board || {}).flatMap(v => Array.isArray(v) ? v : []).find(v => v.record === id);
const current = () => board?.current[regionId];
const target = () => record(current()?.target);
const plan = () => record(current()?.plan);
const review = () => record(reviewId);
const imageUrl = id => `/api/image/${board.board}/${id}`;
const data = form => Object.fromEntries(new FormData(form).entries());
const lines = value => value.split('\n').map(s => s.trim()).filter(Boolean);
const selected = form => Array.from(form.elements.references.selectedOptions, o => o.value);
const fact = (wording, date, source) => ({actor:'user', wording, date, source});
const scope = () => {
  if (!board) throw Error('Create or open a feedback board first.');
  return {board:board.board, expected_revision:board.revision};
};
function message(text, error=false) {
  $('message').textContent = text; $('message').className = error ? 'error' : ''; $('message').style.display='block';
}
async function loadWorkflow() {
  workflow=await request('/workflow.json');
  $('workflow-cycle').innerHTML=workflow.cycle.map((s,i)=>`<article><h3>${i+1} · ${esc(s.title)}</h3><p>${esc(s.instruction)}</p></article>`).join('');
  $('workflow-scope').textContent=workflow.evidence_scope;
  $('workflow-methods').innerHTML=workflow.methods.map(m=>`<article class="card"><span class="badge">${esc(m.status)}</span><h3>${esc(m.title)}</h3><p>${esc(m.when)}</p><p><b>What helped:</b> ${esc(m.basis)}</p><p><b>Method:</b> ${esc(m.method)}</p><p><b>Use these references:</b> ${esc(m.reference)}</p><p class="hint"><b>Limits:</b> ${esc(m.limits)}</p><details><summary>Existing modeling tools</summary><p class="meta">${esc(m.operations.join(', '))}</p></details></article>`).join('');
  $('workflow-experiments').innerHTML=workflow.experimental.map(m=>`<article class="card"><span class="badge">${esc(m.status)}</span><h3>${esc(m.title)}</h3><p>${esc(m.instruction)}</p></article>`).join('');
  $('workflow-choice').innerHTML='<option value="">Choose an approach</option>'+workflow.methods.map(m=>`<option value="${m.id}">${esc(m.title)} · ${esc(m.status)}</option>`).join('');
}
$('workflow-choice').onchange=()=>{
  const method=workflow?.methods.find(m=>m.id===$('workflow-choice').value);
  $('workflow-choice-summary').textContent=method?method.when+' '+method.limits:'';
};
$('apply-workflow-method').onclick=()=>{
  const method=workflow?.methods.find(m=>m.id===$('workflow-choice').value);
  if(!method){message('Choose an approach for this target first.',true);return;}
  $('plan-form').elements.method.value=method.title+' ['+method.id+']\n'+method.method+'\nEvidence: '+method.status+'. '+method.limits;
  message('Method draft added. Adapt it to the marked target, name preserved features, and link useful reference images before saving.');
};
async function request(path, body) {
  const response = await fetch(path, body === undefined ? {} : {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  const value = await response.json();
  if (!response.ok || ['failed','conflicting'].includes(value.status)) throw Error(value.summary || value.reason || 'Request failed');
  return value;
}
async function operation(name, arguments_) { return request('/api/operation', {operation:'visual_feedback_'+name, arguments:arguments_}); }
async function loadBoards() {
  const value = await request('/api/boards');
  const visible=value.boards.filter(b=>!b.archived||$('show-archived-boards').checked||b.board===board?.board);
  $('boards').innerHTML = '<option value="">Choose a board</option>' + visible.map(b=>`<option value="${b.board}">${esc(b.title)}${b.archived?' · archived':''}</option>`).join('');
  if (board) $('boards').value = board.board;
}
async function reopen(id) {
  const changedBoard=board?.board!==id;
  board = await request('/api/board?board='+encodeURIComponent(id));
  if(changedBoard){$('other-images').open=false;$('archived-images').open=false;}
  if (!board.current[regionId]) regionId = Object.keys(board.current)[0] || null;
  revising = null;
  $('workspace').hidden = false;
  $('export').hidden = false; $('export').href='/api/summary?board='+board.board; $('export').download='visual-target-summary.md';
  render(); await loadBoards();
}
function bindForm(id, handler) {
  $(id).addEventListener('submit', async event => {
    event.preventDefault();
    const button = event.submitter; if (button) button.disabled = true;
    try { await handler($(id)); }
    catch (error) { message(error.message+' Your form text remains available.',true); }
    finally { if (button) button.disabled=false; }
  });
}
function chooseTab(name) {
  document.querySelectorAll('.tab').forEach(e=>e.hidden=e.id!==name+'-tab');
  document.querySelectorAll('[data-tab]').forEach(e=>e.classList.toggle('active', e.dataset.tab===name));
  requestAnimationFrame(initializeReviewViews);
}
document.querySelectorAll('[data-tab]').forEach(e=>e.onclick=()=>chooseTab(e.dataset.tab));
$('boards').onchange = async()=> { if ($('boards').value) try {regionId=null; reviewId=null; await reopen($('boards').value);message('Opened '+board.title+'.');}catch(e){message(e.message,true);} };
$('reopen').onclick = async()=> {try {if (board) await reopen(board.board); message('Reopened from the persistent Workbench store.');}catch(e){message(e.message,true);} };
$('show-archived-boards').onchange=()=>loadBoards().catch(e=>message(e.message,true));
$('archive-board').onclick=async()=>{
  try {await operation('presentation',{...scope(),collection:'board',record:board.board,
    archived:!board.presentation?.board_archived,reason:'User selected recoverable board archive/restore in the local UI'});
    await reopen(board.board);message('Board presentation updated. All images, targets and history are preserved.');
  }catch(e){message(e.message,true);}
};
bindForm('new-board', async form => {
  const result = await operation('create',{title:data(form).title,idempotency_key:crypto.randomUUID()});
  await reopen(result.board); message('Board created. Import a source image, then mark the target region.');
});
bindForm('upload', async form => {
  const fields = data(form), file = form.elements.file.files[0];
  if (!file || file.size>20*1024*1024) throw Error('Choose an image smaller than 20 MiB.');
  const bytes = new Uint8Array(await file.arrayBuffer()); let binary='';
  for (let i=0;i<bytes.length;i+=8192) binary+=String.fromCharCode(...bytes.subarray(i,i+8192));
  const metadata = Object.fromEntries(['label','source','source_version','camera','display_state','pose','lighting','framing','view_role'].map(k=>[k,fields[k]||null]));
  metadata.captured_at=fields.captured_at;
  const result = await request('/api/upload',{...scope(),filename:file.name,data:btoa(binary),metadata});
  await reopen(board.board); $('target-form').elements.image.value=result.added.images[0]; updateMarkImage();
  message('Image bytes and source/version metadata pinned. Mark the exact region.');
});
function options(selector, collection, label) {
  document.querySelectorAll(selector).forEach(select=>{
    const previous=Array.from(select.selectedOptions,o=>o.value);
    select.innerHTML=(select.classList.contains('context-select')?'<option value="">Not supplied</option>':'')+collection.map(r=>`<option value="${r.record}">${esc(label(r))}</option>`).join('');
    if (select.multiple) Array.from(select.options).forEach(o=>o.selected=previous.includes(o.value));
    else if (previous[0]===''||previous.includes(select.value) || collection.some(r=>r.record===previous[0])) select.value=previous[0];
  });
}
function regionFields(value) {
  const form=$('target-form'); ['x','y','w','h'].forEach((key,i)=>form.elements[key].value=value[i]); updateRect();
}
function updateRect() {
  const f=$('target-form'), rectangle=['x','y','w','h'].map(k=>Number(f.elements[k].value));
  ['x','y','width','height'].forEach((k,i)=>$('mark-rect').setAttribute(k,String(rectangle[i])));
}
function updateMarkImage() {
  const id=$('target-form').elements.image.value;
  $('mark-image').hidden=!id;
  if (id) $('mark-image').src=imageUrl(id);
}
function point(event) {
  const rect=$('mark-canvas').getBoundingClientRect();
  return [Math.max(0,Math.min(1,(event.clientX-rect.left)/rect.width)),Math.max(0,Math.min(1,(event.clientY-rect.top)/rect.height))];
}
$('mark-canvas').onpointerdown=event=>{if (!$('target-form').elements.image.value)return;dragStart=point(event);$('mark-canvas').setPointerCapture(event.pointerId);event.preventDefault();};
$('mark-canvas').onpointermove=event=>{if (!dragStart)return;const p=point(event);regionFields([Math.min(p[0],dragStart[0]),Math.min(p[1],dragStart[1]),Math.abs(p[0]-dragStart[0]),Math.abs(p[1]-dragStart[1])].map(v=>Number(v.toFixed(6))));};
$('mark-canvas').onpointerup=()=>{dragStart=null;};
$('target-form').elements.image.onchange=updateMarkImage;
['x','y','w','h'].forEach(k=>$('target-form').elements[k].oninput=updateRect);
function newRegion() {
  revising=null; $('target-form').reset(); setDates(); $('target-form-title').textContent='Mark a new region and retain the exact words';
  $('cancel-revision').hidden=true; $('target-details').open=true; chooseTab('target'); updateMarkImage(); updateRect();
}
$('new-region').onclick=newRegion; $('cancel-revision').onclick=newRegion;
bindForm('target-form',async form=>{
  const f=data(form), result=await operation('target',{...scope(),image:f.image,label:f.label,wording:f.wording,date:f.date,source:f.source,
    region:['x','y','w','h'].map(k=>Number(f[k])),...(revising?{target_id:revising.target_id,supersedes:revising.record}:{})});
  await reopen(board.board);regionId=record(result.added.targets[0]).target_id;reviewId=null;render(); $('target-details').open=false;
  message('Target revision saved with exact wording, marked image and provenance. Previous revisions remain in history.');
});
function planFields(form, row) {
  if (!row) {form.reset(); return;}
  for (const key of ['interpretation','method','expected_appearance']) form.elements[key].value=row[key];
  form.elements.preserved_features.value=row.preserved_features.join('\n');
  Array.from(form.elements.references.options).forEach(o=>o.selected=row.references.includes(o.value));
}
function planData(form) {const f=data(form);return {interpretation:f.interpretation,preserved_features:lines(f.preserved_features),method:f.method,expected_appearance:f.expected_appearance,references:selected(form)};}
bindForm('plan-form',async form=>{
  if (!target())throw Error('Save a target region first.');
  await operation('plan',{...scope(),target:target().record,...planData(form),supersedes:current().plan});
  await reopen(board.board); $('plan-details').open=false; message('Interpretation revision saved. Prior agreement stays on its exact earlier revision.');
});
bindForm('reference-form',async form=>{
  await operation('reference',{...scope(),...data(form)});await reopen(board.board);message('Local appearance role saved without clearing the overall guide rejection or adopting geometry.');
});
bindForm('agreement-form',async form=>{
  if (!plan())throw Error('Record the target interpretation first.');const f=data(form);
  await operation('agreement',{...scope(),target:target().record,plan:plan().record,fact:fact(f.wording,f.date,f.source)});
  await reopen(board.board);message('Explicit agreement recorded for this target and interpretation only.');
});
bindForm('compare-form',async form=>{
  if (!plan())throw Error('Record a target interpretation first.');const f=data(form);
  const context_views=[];
  for(const [role,prefix] of [['eye_context','eye'],['whole_face','face']]){
    if(Boolean(f[prefix+'_baseline'])!==Boolean(f[prefix+'_trial']))throw Error('Supply both LEFT and RIGHT images for '+role.replace('_',' ')+', or leave both empty.');
    if(f[prefix+'_baseline'])context_views.push({role,baseline:f[prefix+'_baseline'],trial:f[prefix+'_trial']});
  }
  const result=await operation('compare',{...scope(),target:target().record,plan:plan().record,baseline:f.baseline,trial:f.trial,
    baseline_caption:f.baseline_caption,trial_caption:f.trial_caption,changed:f.changed,unchanged:f.unchanged,
    baseline_region:JSON.parse(f.baseline_region),trial_region:JSON.parse(f.trial_region),
    baseline_approval:f.approval_wording?fact(f.approval_wording,f.approval_date,f.approval_source):null,context_views});
  reviewId=result.added.comparisons[0];await reopen(board.board);$('compare-details').open=false;
  message('Fixed baseline-left / trial-right comparison saved. Matching reflects recorded descriptors; result approval is separate.');
});
$('review-select').onchange=()=>{reviewId=$('review-select').value;renderComparison();};
$('feedback-form').elements.correct.onchange=()=>{
  const checked=$('feedback-form').elements.correct.checked;$('correction-fields').hidden=!checked;
  if (checked) planFields($('feedback-form'),record(review()?.plan));
};
function feedbackBinding() {
  const r=review();
  return r?{board:board.board,comparison:r.record,label:record(r.target).label,result:record(r.trial).metadata.label}:null;
}
function showFeedbackDraft() {
  const binding=feedbackBinding(), changed=feedbackDraft&&
    (feedbackDraft.board!==binding?.board||feedbackDraft.comparison!==binding?.comparison);
  $('feedback-draft-status').hidden=!changed;
  $('feedback-draft-status').textContent=changed?`This draft belongs to ${feedbackDraft.label} · ${feedbackDraft.result}. Return to that comparison to submit it, or clear the draft before writing for another result.`:'';
  $('discard-feedback-draft').hidden=!feedbackDraft;
  $('motion-draft-scope').hidden=!motionDraft;
  if(motionDraft){const clip=record(motionDraft.video);$('motion-draft-scope').textContent=`Pinned motion draft: ${clip?.metadata.label||motionDraft.video} · frame ${motionDraft.frame_index} · ${motionDraft.timestamp_seconds}s · ${motionDraft.panel} · region ${JSON.stringify(motionDraft.region)}. Playback and speed selection do not change this draft.`;}
}
function bindFeedbackDraft() {
  if (!feedbackDraft) feedbackDraft=feedbackBinding();
  showFeedbackDraft();
}
$('feedback-form').addEventListener('input',bindFeedbackDraft);
$('feedback-form').addEventListener('change',bindFeedbackDraft);
$('discard-feedback-draft').onclick=()=>{
  feedbackDraft=null;motionDraft=null;$('feedback-form').reset();$('correction-fields').hidden=true;setDates();showFeedbackDraft();
};
bindForm('feedback-form',async form=>{
  const r=review();if (!r)throw Error('Display a comparison first.');const f=data(form);
  const binding=feedbackBinding();
  if(feedbackDraft&&(feedbackDraft.board!==binding.board||feedbackDraft.comparison!==binding.comparison))
    throw Error('The displayed comparison changed after this draft began. Return to its original comparison or clear the draft.');
  if(form.elements.include_motion.checked&&!motionDraft)throw Error('Pause a clip, mark its region and pin that frame first.');
  await operation('submit',{...scope(),comparison:r.record,target:r.target,plan:r.plan,inspection:r.inspection,trial:r.trial,
    source_version:r.source_version,result_version:r.result_version,wording:f.wording,date:f.date,source:f.source,
    correction:form.elements.correct.checked?planData(form):null,historical:form.elements.historical.checked,
    motion:form.elements.include_motion.checked?motionDraft:null});
  feedbackDraft=null;motionDraft=null;form.reset();setDates();await reopen(board.board);$('correction-fields').hidden=true;
  message('Version-bound feedback saved. A corrected interpretation is a new revision; its agreement and next comparison are explicit.');
});
const choices={execution:['proposed','running','built'],technical_verification:['unknown','passed','failed'],visually_useful:['unknown','yes','no'],retained:['unknown','yes','no'],owner_acceptance:['unknown','accepted','rejected']};
function stateChoices(){const f=$('state-form');f.elements.value.innerHTML=choices[f.elements.facet.value].map(v=>`<option>${v}</option>`).join('');ownerFact();}
function ownerFact(){$('owner-fact').hidden=!($('state-form').elements.facet.value==='owner_acceptance'&&$('state-form').elements.value.value!=='unknown');}
$('state-form').elements.facet.onchange=stateChoices;$('state-form').elements.value.onchange=ownerFact;stateChoices();
bindForm('state-form',async form=>{
  if (!review())throw Error('Display a comparison first.');const f=data(form);
  await operation('state',{...scope(),comparison:reviewId,facet:f.facet,value:f.value,evidence:f.evidence,
    fact:f.facet==='owner_acceptance'&&f.value!=='unknown'?fact(f.wording,f.date,f.source):null});
  await reopen(board.board);message('Independent fact recorded for the displayed result. Other state facets are unchanged.');
});
function provenance(row){return `<details><summary>Exact provenance and revision</summary><pre class="provenance">${esc(JSON.stringify(row,null,2))}</pre></details>`;}
function referenceCards(refs){return `<div class="reference-cards">${refs.map(id=>{const r=record(id);return r?`<article class="card"><h3>${esc(r.role)}</h3><img src="${imageUrl(r.image)}" alt="${esc(r.role)} reference"><p><b>Overall guide:</b> ${esc(r.overall_status)} · ${esc(r.rejection_reason||'No rejection supplied')}</p><p><b>Supported:</b> ${esc(r.supported)}</p><p><b>Excluded:</b> ${esc(r.excluded)}</p><p><b>Unknown:</b> ${esc(r.unknown)}</p><p><b>Qualification:</b> ${esc(r.qualification)}</p><p class="hint">Local role only. Geometry adoption: none.</p>${provenance(r)}</article>`:'';}).join('')}</div>`;}
function renderImages() {
  const active=current(), t=target(), p=plan(), r=record(active?.comparison);
  const relevant=new Set([t?.image,r?.baseline,r?.trial,...(r?.context_views||[]).flatMap(v=>[v.baseline,v.trial]),...(p?.references||[]).map(id=>record(id)?.image)].filter(Boolean));
  const archived=new Set(board.presentation?.archived_images||[]);
  const thumbnail=i=>`<div class="saved-image"><div class="thumbnail"><img src="${imageUrl(i.record)}" alt=""><span>${esc(i.metadata.label)}</span></div><button type="button" data-image-archive="${i.record}" data-archived="${archived.has(i.record)}">${archived.has(i.record)?'Restore image':'Archive image'}</button>${archived.has(i.record)&&relevant.has(i.record)?'<p class="hint">Archived, still used by this target.</p>':''}</div>`;
  $('image-list').innerHTML=rows('images').filter(i=>relevant.has(i.record)).map(thumbnail).join('')||'<p class="hint">Mark a region to collect its current evidence here.</p>';
  const other=rows('images').filter(i=>!relevant.has(i.record)&&!archived.has(i.record));
  $('other-images').querySelector('summary').textContent=`Other saved images (${other.length})`;
  $('other-image-list').innerHTML=other.map(thumbnail).join('')||'<p class="hint">None.</p>';
  $('archived-images').querySelector('summary').textContent=`Archived images (${archived.size})`;
  $('archived-image-list').innerHTML=rows('images').filter(i=>archived.has(i.record)).map(thumbnail).join('')||'<p class="hint">None.</p>';
  document.querySelectorAll('[data-image-archive]').forEach(b=>b.onclick=async()=>{
    try{await operation('presentation',{...scope(),collection:'images',record:b.dataset.imageArchive,
      archived:b.dataset.archived!=='true',reason:'User selected recoverable image archive/restore in the local UI'});
      await reopen(board.board);message('Image presentation updated. Pinned bytes and all references/history are preserved.');
    }catch(e){message(e.message,true);}
  });
  // New selections favor working images. Existing referenced/selected evidence
  // remains selectable even if archived, without clearing historical bindings.
  options('.image-select',rows('images').filter(i=>!archived.has(i.record)||relevant.has(i.record)||
    Array.from(document.querySelectorAll('.image-select')).some(s=>s.value===i.record)),i=>i.metadata.label+' · '+i.metadata.captured_at+' · '+(i.metadata.view_role?.replace('_',' ')||'extent unknown'));
}
function render(){
  $('archive-board').hidden=false;
  $('archive-board').textContent=board.presentation?.board_archived?'Restore this board':'Archive this board';
  $('regions').innerHTML=Object.entries(board.current).map(([id,a])=>{const t=record(a.target);return `<button class="region-button ${id===regionId?'selected':''}" data-region="${id}">${esc(t.label)}<br><span class="hint">${esc(t.date)} · ${rows('targets').filter(r=>r.target_id===id).length} target revision(s)</span></button>`;}).join('')||'<p class="hint">Import an inspection image and mark the first region.</p>';
  document.querySelectorAll('[data-region]').forEach(b=>b.onclick=()=>{regionId=b.dataset.region;reviewId=null;render();});
  renderImages();
  options('.reference-select',rows('reference_roles'),r=>r.role+' · overall '+r.overall_status);
  const t=target(),p=plan();
  const agreement=record(current()?.agreement);
  const targetHtml=t?`<article class="card"><span class="eyebrow">EXACT USER TARGET · ${esc(t.label)}</span><blockquote class="target-wording">${esc(t.wording)}</blockquote><p class="meta">${esc(t.date)} · ${esc(t.source)}<br>Inspection version: ${esc(t.source_version)}</p><span class="badge">${agreement?'Explicit target agreement recorded':'No agreement recorded for this revision'}</span>${agreement?`<p class="hint">${esc(agreement.fact.wording)} · ${esc(agreement.fact.date)} · ${esc(agreement.fact.source)}<br>Target / interpretation agreement only.</p>`:''}${provenance(t)}</article>`:'<p>Start with the exact words and a marked source image.</p>';
  $('target-summary').innerHTML=targetHtml+(t?'<button id="revise-target" type="button">Revise wording / region</button>':'');$('review-target').innerHTML=targetHtml;
  if(t)$('target-summary').insertAdjacentHTML('beforeend',`<div id="saved-target-image">${panel(t.image,'Pinned inspection · '+t.label,t.region,'Saved target region')}</div>`);
  if(t)$('revise-target').onclick=()=>{revising=t;const f=$('target-form');for(const k of ['image','label','wording','date','source'])f.elements[k].value=t[k];regionFields(t.region);updateMarkImage();$('target-details').open=true;$('target-form-title').textContent='Revise this target · old wording and region remain in history';$('cancel-revision').hidden=false;};
  $('plan-summary').innerHTML=p?`<article class="card"><span class="eyebrow">CURRENT INTERPRETATION ${p.correction_from_comparison?'· CORRECTED FROM FEEDBACK':''}</span><h2>${esc(p.interpretation)}</h2><p><b>Preserve:</b> ${esc(p.preserved_features.join('; '))}</p><p><b>Operation:</b> ${esc(p.method)}</p><p><b>Expected appearance:</b> ${esc(p.expected_appearance)}</p>${referenceCards(p.references)}${provenance(p)}</article>`:'<p class="hint">Save the region, then record what you understand and what the operation must preserve.</p>';
  planFields($('plan-form'),p);
  if (t){const f=$('compare-form');f.elements.baseline_region.value=JSON.stringify(t.region);f.elements.trial_region.value=JSON.stringify(t.region);}
  const reviews=rows('comparisons').filter(r=>r.target_id===regionId);
  if(!reviews.some(r=>r.record===reviewId))reviewId=current()?.comparison||reviews.at(-1)?.record||null;
  $('review-select').innerHTML=reviews.map((r,i)=>`<option value="${r.record}">Review ${i+1} · ${esc(record(r.trial).metadata.label)} · ${esc(r.matching.status)}${r.record===current()?.comparison?' · current':' · historical'}</option>`).join('');
  if(reviewId)$('review-select').value=reviewId;
  renderComparison();renderHistory();updateMarkImage();setDates();
  if(!t)$('target-details').open=true;
}
function panel(imageId,caption,region,heading,viewing=false){
  const i=record(imageId),m=i.metadata;
  const annotation=region?`<svg class="target-outline" viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true"><rect x="${region[0]}" y="${region[1]}" width="${region[2]}" height="${region[3]}"></rect></svg>`:'';
  const image=`<img src="${imageUrl(imageId)}" alt="${esc(caption)}" draggable="false">${annotation}`;
  const frame=viewing?`<div class="review-viewport" data-width="${i.size[0]}" data-height="${i.size[1]}"><div class="review-stage">${image}</div></div>`:`<div class="image-frame">${image}</div><button type="button" data-inspection-outline aria-pressed="false">Show target outline</button>`;
  const descriptor=value=>esc(typeof value==='object'&&value!==null?JSON.stringify(value):value||'unknown');
  return `<article class="panel"><div class="panel-title"><h3>${esc(heading)}</h3></div>${frame}<div class="caption">${esc(caption)}</div><div class="meta">${esc(m.captured_at)}<br>Version: ${esc(m.source_version)}<br>Camera: ${descriptor(m.camera)}<br>Light: ${descriptor(m.lighting)}<br>State: ${descriptor(m.display_state)}<br>Scale / framing: ${descriptor(m.framing)}<br>Pose: ${descriptor(m.pose)}</div>${provenance(i)}</article>`;
}
function displayMatching(pair){
  const result={...pair.matching,mismatched:[...(pair.matching?.mismatched||[])],unknown:[...(pair.matching?.unknown||[])]};
  // Old immutable comparisons predate explicit light/framing descriptors. Show
  // their missing qualification without rewriting or invalidating their history.
  for(const field of ['lighting','framing']){
    const left=record(pair.baseline).metadata[field],right=record(pair.trial).metadata[field];
    if(!left||!right)result.unknown.push(field);
    else if(JSON.stringify(left)!==JSON.stringify(right))result.mismatched.push(field);
  }
  result.unknown=[...new Set(result.unknown)];result.mismatched=[...new Set(result.mismatched)];
  result.status=result.mismatched.length?'unmatched':result.unknown.length?'unknown':'matched declared inputs';
  return result;
}
function reviewPair(r,pair,label,key,annotations=false){
  const match=displayMatching(pair),state=reviewViews.get(key)||{zoom:1,x:0,y:0,outlines:false};reviewViews.set(key,state);
  const sameVersion=record(pair.baseline).metadata.source_version===record(r.baseline).metadata.source_version;
  const left=r.baseline_approval&&sameVersion?'LEFT · Owner-approved baseline (recorded user fact)':'LEFT · Baseline — approval unknown';
  return `<section class="review-pair" data-view-key="${esc(key)}" data-view-role="${esc(pair.role||'original')}"><h3>${esc(label)}</h3><p class="${match.status==='matched declared inputs'?'match':'warning'}"><b>${esc(match.status.toUpperCase())}</b> ${esc([...match.mismatched.map(s=>'Different: '+s),...match.unknown.map(s=>'Unknown: '+s)].join(' · '))}</p><div class="view-controls"><button type="button" data-fit-view>Fit entire image</button><label>Both panels · zoom<input type="range" data-zoom-view min="1" max="4" step="0.1" value="${state.zoom}" aria-label="${esc(label)} synchronized zoom"></label><output>${Math.round(state.zoom*100)}%</output>${annotations?'<button type="button" data-outline-view aria-pressed="false">Show target outline</button>':''}</div><p class="hint">Both panels use the same viewing zoom and pan. Drag to pan after zooming. ${annotations?'Target outline is hidden for skin review.':'Target coordinates remain on the pinned inspection; no annotation is inferred in this camera.'}</p><div class="panels">${panel(pair.baseline,r.baseline_caption,annotations?r.baseline_region:null,left,true)}${panel(pair.trial,r.trial_caption,annotations?r.trial_region:null,'RIGHT · Isolated trial',true)}</div></section>`;
}
function reviewEvidence(r){
  const supplied=r.context_views||[],used=new Set(),parts=[];
  const primaryRole=record(r.baseline).metadata.view_role;
  for(const [role,label] of [['eye_context','Eye and surrounding face'],['whole_face','Whole face']]){
    let pair=supplied.find(v=>v.role===role);
    if(!pair&&primaryRole===role&&record(r.trial).metadata.view_role===role)pair={...r,role};
    if(pair){parts.push(reviewPair(r,pair,label,r.record+':'+role,pair.baseline===r.baseline&&pair.trial===r.trial));used.add(pair.baseline+'|'+pair.trial);}
    else parts.push(`<p class="warning" data-missing-context="${role}">${label} pair not supplied for these exact versions. A tight crop cannot establish this context. Add broader source captures without replacing the saved target or comparison history.</p>`);
  }
  if(!used.has(r.baseline+'|'+r.trial)){
    const original=reviewPair(r,r,'Full original source images · recorded extent '+(primaryRole==='detail'?'closeup':primaryRole||'unknown'),r.record+':original',true);
    if(!primaryRole&&!supplied.length)parts.push(original);
    else parts.push(`<details class="detail-review"><summary>Optional closeup / original comparison</summary>${original}</details>`);
  }
  return parts.join('');
}
function applyReviewView(pair){
  const state=reviewViews.get(pair.dataset.viewKey);if(!state)return;
  const titles=Array.from(pair.querySelectorAll('.panel-title'));
  titles.forEach(title=>title.style.minHeight='');
  const titleHeight=Math.max(...titles.map(title=>title.offsetHeight));
  titles.forEach(title=>title.style.minHeight=titleHeight+'px');
  pair.classList.toggle('outlines-visible',state.outlines);
  pair.querySelector('output').textContent=Math.round(state.zoom*100)+'%';
  const toggle=pair.querySelector('[data-outline-view]');
  if(toggle){toggle.setAttribute('aria-pressed',String(state.outlines));toggle.textContent=state.outlines?'Hide target outline':'Show target outline';}
  for(const viewport of pair.querySelectorAll('.review-viewport')){
    const stage=viewport.querySelector('.review-stage'),ratio=Number(viewport.dataset.width)/Number(viewport.dataset.height);
    const width=Math.min(viewport.clientWidth,viewport.clientHeight*ratio),height=width/ratio;
    stage.style.width=width+'px';stage.style.height=height+'px';
    stage.style.transform=`translate(-50%,-50%) translate(${state.x*viewport.clientWidth}px,${state.y*viewport.clientHeight}px) scale(${state.zoom})`;
  }
}
function initializeReviewViews(){
  document.querySelectorAll('.review-pair').forEach(pair=>{
    applyReviewView(pair);
    pair.querySelector('[data-fit-view]').onclick=()=>{const s=reviewViews.get(pair.dataset.viewKey);Object.assign(s,{zoom:1,x:0,y:0});pair.querySelector('[data-zoom-view]').value=1;applyReviewView(pair);};
    pair.querySelector('[data-zoom-view]').oninput=e=>{reviewViews.get(pair.dataset.viewKey).zoom=Number(e.target.value);applyReviewView(pair);};
    const toggle=pair.querySelector('[data-outline-view]');if(toggle)toggle.onclick=()=>{const s=reviewViews.get(pair.dataset.viewKey);s.outlines=!s.outlines;applyReviewView(pair);};
    for(const viewport of pair.querySelectorAll('.review-viewport')){
      let start=null;
      viewport.onpointerdown=e=>{const s=reviewViews.get(pair.dataset.viewKey);if(s.zoom<=1)return;start={px:e.clientX,py:e.clientY,x:s.x,y:s.y};viewport.setPointerCapture(e.pointerId);e.preventDefault();};
      viewport.onpointermove=e=>{if(!start)return;const s=reviewViews.get(pair.dataset.viewKey),limit=(s.zoom-1)/2;s.x=Math.max(-limit,Math.min(limit,start.x+(e.clientX-start.px)/viewport.clientWidth));s.y=Math.max(-limit,Math.min(limit,start.y+(e.clientY-start.py)/viewport.clientHeight));applyReviewView(pair);};
      viewport.onpointerup=viewport.onpointercancel=()=>{start=null;};
    }
  });
}
new ResizeObserver(()=>document.querySelectorAll('.review-pair').forEach(applyReviewView)).observe($('workspace'));
document.addEventListener('toggle',event=>{if(event.target.matches('.detail-review'))initializeReviewViews();},true);
document.addEventListener('click',event=>{
  const button=event.target.closest('[data-inspection-outline]');if(!button)return;
  const shown=button.getAttribute('aria-pressed')!=='true';button.setAttribute('aria-pressed',String(shown));
  button.closest('.panel').classList.toggle('outlines-visible',shown);button.textContent=shown?'Hide target outline':'Show target outline';
});
function renderComparison(){
  const r=review();if(!r){$('comparison').innerHTML='<p>No comparison for this region yet. Add source-bound baseline and isolated trial images.</p>';$('feedback-scope').textContent='No comparison selected.';showFeedbackDraft();renderMotion();return;}
  const t=record(r.target),p=record(r.plan),active=r.record===current()?.comparison;
  const matching=displayMatching(r);
  const state={};rows('state_facts').filter(f=>f.comparison===r.record).forEach(f=>state[f.facet]=f);
  $('comparison').innerHTML=`${active?'':'<div class="warning">Historical comparison. This target or interpretation has changed. Feedback stays on these exact versions; a new comparison is needed for the current interpretation.</div>'}<div class="${matching.status==='matched declared inputs'?'match':'warning'}"><b>${esc(matching.status.toUpperCase())}</b><br>${esc([...matching.mismatched.map(s=>'Different: '+s),...matching.unknown.map(s=>'Unknown: '+s)].join(' · '))}<span class="hint">${esc(matching.basis)}</span></div><p><b>Marked target:</b> ${esc(t.wording)}<br><b>Interpretation at this review:</b> ${esc(p.interpretation)}</p>${reviewEvidence(r)}<div class="states">${Object.entries(choices).map(([facet])=>`<div><strong>${esc(facet.replaceAll('_',' '))}</strong>${esc(state[facet]?.value||'unknown')}</div>`).join('')}</div><article class="card"><p><b>Changed:</b> ${esc(r.changed)}</p><p><b>Unchanged / unresolved:</b> ${esc(r.unchanged)}</p><p class="hint">An isolated result is not owner-approved without its own explicit user fact. Target agreement and technical verification do not imply acceptance.</p>${r.baseline_approval?`<p class="meta">Baseline user fact: ${esc(r.baseline_approval.wording)} · ${esc(r.baseline_approval.date)} · ${esc(r.baseline_approval.source)}</p>`:''}${provenance(r)}</article>`;
  $('feedback-scope').innerHTML=`<strong>${esc(t.label)} · ${esc(record(r.trial).metadata.label)}</strong><br>Target revision: ${r.target.slice(0,12)} · interpretation: ${r.plan.slice(0,12)}<br>Inspection source version: ${esc(r.source_version)}<br>Result version: ${esc(r.result_version)}<br>${active?'Current exact comparison':'Historical — correction cannot activate from this review'}`;
  initializeReviewViews();showFeedbackDraft();renderMotion();
}
function renderHistory(){
  const entries=[];
  for(const collection of ['targets','plans','comparisons','videos','motion_reviews','feedback','agreements','reference_roles','state_facts'])for(const r of rows(collection)){
    if(r.target_id&&r.target_id!==regionId)continue;
    let content='';
    if(collection==='targets')content=`<h3>Exact target wording</h3><blockquote>${esc(r.wording)}</blockquote><p class="hint">${esc(r.label)} · ${esc(r.date)} · ${esc(r.source)}${r.supersedes?' · supersedes a prior target revision':''}</p>`;
    if(collection==='plans')content=`<h3>${r.correction_from_comparison?'Corrected interpretation':'Interpretation / method'}</h3><p>${esc(r.interpretation)}</p><p><b>Preserve:</b> ${esc(r.preserved_features.join('; '))}<br><b>Method:</b> ${esc(r.method)}<br><b>Expected:</b> ${esc(r.expected_appearance)}</p>${r.correction_wording?`<blockquote>${esc(r.correction_wording)}</blockquote>`:''}`;
    if(collection==='comparisons')content=`<h3>Baseline / trial review · ${esc(r.matching.status)}</h3><p>LEFT: ${esc(r.baseline_caption)}<br>RIGHT: ${esc(r.trial_caption)}</p>`;
    if(collection==='feedback')content=`<h3>${r.interpretation_correction?'User correction of interpretation':'Version-bound feedback'}</h3><blockquote>${esc(r.wording)}</blockquote><p class="hint">${esc(r.date)} · ${esc(r.source)}<br>${esc(r.applicability)}</p>`;
    if(collection==='videos')content=`<h3>Pinned motion · ${esc(r.metadata.label)}</h3><p>${esc(r.metadata.speed_label)} · ${r.frame_count} decoded frames · ${esc(r.metadata.result_version||'Result version unknown')}</p>`;
    if(collection==='motion_reviews')content=`<h3>Motion linked to exact comparison · ${esc(r.matching.status)}</h3><p>${esc(r.moment_mapping)}</p>`;
    if(collection==='feedback'&&r.motion)content+=`<p><b>Motion:</b> ${esc(record(r.motion.video)?.metadata.label||r.motion.video)} · frame ${r.motion.frame_index} · ${r.motion.timestamp_seconds}s<br>Region ${esc(JSON.stringify(r.motion.region))} · ${esc(r.motion.panel)}<br>Source moment: ${esc(r.motion.moment_id||'unknown; no cross-speed inference')}</p>`;
    if(collection==='agreements')content=`<h3>Explicit target / interpretation agreement</h3><blockquote>${esc(r.fact.wording)}</blockquote><p class="hint">${esc(r.scope)}</p>`;
    if(collection==='reference_roles')content=`<h3>Qualified reference role · ${esc(r.role)}</h3><p>Overall: ${esc(r.overall_status)} · ${esc(r.rejection_reason)}<br>Supported: ${esc(r.supported)}<br>Excluded: ${esc(r.excluded)}<br>Unknown: ${esc(r.unknown)}</p>`;
    if(collection==='state_facts')content=`<h3>${esc(r.facet.replaceAll('_',' '))} · ${esc(r.value)}</h3><p>${esc(r.evidence)}</p>${r.fact?`<blockquote>${esc(r.fact.wording)}</blockquote>`:''}`;
    const activeField={targets:'target',plans:'plan',comparisons:'comparison',agreements:'agreement'}[collection];
    const status=activeField?`<span class="badge">${r.record===current()?.[activeField]?'Current revision':'Historical revision'}</span>`:'';
    const scopeText=r.target_id?`<p class="meta">Region: ${esc(record(r.target||r.record)?.label||target()?.label)}${r.trial?' · Result: '+esc(record(r.trial)?.metadata.label):''}${r.result_version?'<br>Result version: '+esc(r.result_version):''}</p>`:'';
    entries.push({time:r.recorded_at,html:`<article class="history-entry ${r.correction_from_comparison||r.interpretation_correction?'corrected':''}">${status}${content}${scopeText}${provenance(r)}</article>`});
  }
  $('history').innerHTML=entries.sort((a,b)=>b.time.localeCompare(a.time)).map(e=>e.html).join('')||'<p>No history yet.</p>';
}
bindForm('video-upload',async form=>{
  const file=form.elements.file.files[0];
  if(!file||file.size>20*1024*1024)throw Error('Choose an MP4 smaller than 20 MiB; larger clips can use the service.');
  const metadata=JSON.parse(form.elements.metadata.value),bytes=new Uint8Array(await file.arrayBuffer());let binary='';
  for(let i=0;i<bytes.length;i+=8192)binary+=String.fromCharCode(...bytes.subarray(i,i+8192));
  const result=await request('/api/video-upload',{...scope(),filename:file.name,data:btoa(binary),metadata});
  await reopen(board.board);$('motion-link').elements.first.value=result.added.videos[0];
  message('Clip bytes and decoded frame timestamps pinned. Link the clip to the exact comparison.');
});
bindForm('motion-link',async form=>{
  if(!review())throw Error('Display the exact baseline/trial comparison first.');
  const f=data(form);await operation('motion',{...scope(),comparison:review().record,videos:[f.first,...(f.second?[f.second]:[])]});
  await reopen(board.board);message('Motion linked to this exact comparison. Unknown and unmatched provenance remains visible.');
});
function activeMotion(){return rows('motion_reviews').filter(m=>m.comparison===reviewId).at(-1);}
function renderMotion(){
  options('.motionclip-select',rows('videos'),v=>v.metadata.label+' · '+v.metadata.speed_label+' · '+v.record.slice(0,12));
  const m=activeMotion();
  if(!m){$('motion-player').innerHTML='<p class="hint">No motion clip is linked to this comparison. Import a producer-supplied clip and its source metadata below. Existing image feedback remains available.</p>';return;}
  if(!m.videos.includes(motionClipId))motionClipId=m.videos[0];
  const clip=record(motionClipId),meta=clip.metadata;motionRegion=null;
  const layout=meta.layout==='baseline_left_trial_right'?'LEFT · '+esc(review().baseline_caption)+' | RIGHT · '+esc(review().trial_caption):meta.layout?esc(meta.layout.replace('_only',''))+' only; other side absent':'Baseline/trial layout unknown; no side inferred';
  $('motion-player').innerHTML=`<article class="card"><p class="${m.matching.status==='matched declared inputs'?'match':'warning'}"><b>${esc(m.matching.status.toUpperCase())}</b><br>${esc([...m.matching.mismatched.map(x=>'Different: '+x),...m.matching.unknown.map(x=>'Unknown: '+x)].join(' · '))}</p><p class="hint">${esc(m.moment_mapping)}. ${esc(m.matching.basis)}</p><label>Playback clip / speed<select id="motion-speed">${m.videos.map(id=>{const v=record(id);return `<option value="${id}">${esc(v.metadata.label)} · ${esc(v.metadata.speed_label)}</option>`;}).join('')}</select></label><p id="motion-layout">${layout}</p><p class="meta">Baseline version: ${esc(meta.baseline_version||'unknown')}<br>Trial version: ${esc(meta.result_version||'unknown')}<br>Camera / light / state / scale: ${esc([meta.camera,meta.lighting,meta.display_state,meta.framing].map(x=>typeof x==='object'?JSON.stringify(x):x||'unknown').join(' · '))}<br>Extent: ${esc(meta.view_role?.replace('_',' ')||'unknown')} · Pinned clip SHA256 ${clip.asset.sha256}</p><div class="motion-frame"><video id="motion-video" controls playsinline preload="auto" src="/api/video/${board.board}/${clip.record}" aria-label="${esc(meta.label)}"></video><svg id="motion-mark" viewBox="0 0 1 1" preserveAspectRatio="none" hidden><rect id="motion-rect" x="0" y="0" width="0" height="0"/></svg></div><div class="motion-controls"><button id="motion-fit" type="button">Fit complete frame</button><button id="motion-prev" type="button">Previous frame</button><button id="motion-next" type="button">Next frame</button><button id="motion-mark-toggle" type="button">Pause and mark this frame</button><button id="motion-pin" type="button">Pin frame + region to feedback</button></div><label>Decoded frame (zero-based)<input id="motion-frame-index" type="range" min="0" max="${clip.frame_count-1}" step="1" value="0"></label><p id="motion-time" class="scope">Waiting for decoded playback frame.</p><label>Marked side<select id="motion-panel">${meta.layout==='baseline_left_trial_right'?'<option value="trial">RIGHT · isolated trial</option><option value="baseline">LEFT · baseline</option><option value="unresolved">Side unresolved</option>':meta.layout?`<option value="${meta.layout.replace(/_only$/,'')}">${esc(meta.layout.replace(/_only$/,''))}</option>`:'<option value="unresolved">Unresolved</option>'}</select></label><p class="hint">The complete contextual frame fits by default. Marking pauses playback; a thin outline has no filled overlay. Feedback pins that exact clip/frame/region. No mark is converted into geometry.</p>${provenance(clip)}${provenance(m)}</article>`;
  $('motion-speed').value=motionClipId;
  const video=$('motion-video'),slider=$('motion-frame-index'),svg=$('motion-mark');let shownIndex=null,start=null;
  // Property assignment obeys this local UI's strict self-only style policy;
  // HTML inline style attributes are intentionally not enabled.
  video.closest('.motion-frame').style.aspectRatio=String(clip.size[0]/clip.size[1]);
  video.closest('.motion-frame').style.setProperty('--motion-aspect',String(clip.size[0]/clip.size[1]));
  const indexAt=t=>{let i=0;while(i+1<clip.playback_times.length&&clip.playback_times[i+1]<=t+0.00001)i++;return i;};
  const displayFrame=i=>{shownIndex=i;slider.value=i;$('motion-time').textContent=`Frame ${i} / ${clip.frame_count-1} · video timestamp ${clip.frame_times[i]}s · playback ${clip.playback_times[i]}s · source moment ${meta.frame_ids?.[i]||'unknown'}`;};
  if(video.requestVideoFrameCallback){const update=(_,info)=>{if(!video.isConnected)return;displayFrame(indexAt(info.mediaTime));video.requestVideoFrameCallback(update);};video.requestVideoFrameCallback(update);}
  else {video.addEventListener('seeked',()=>displayFrame(indexAt(video.currentTime)));video.addEventListener('timeupdate',()=>displayFrame(indexAt(video.currentTime)));}
  const hideMark=()=>{svg.setAttribute('hidden','');video.controls=true;};
  const seek=i=>{video.pause();hideMark();motionRegion=null;shownIndex=null;video.currentTime=clip.playback_times[Math.max(0,Math.min(clip.frame_count-1,i))]+0.000001;};
  slider.oninput=()=>seek(Number(slider.value));$('motion-prev').onclick=()=>seek((shownIndex??Number(slider.value))-1);$('motion-next').onclick=()=>seek((shownIndex??Number(slider.value))+1);
  $('motion-fit').onclick=()=>{video.style.objectFit='contain';hideMark();video.closest('.motion-frame').scrollIntoView({block:'center'});};
  $('motion-mark-toggle').onclick=()=>{video.pause();const hidden=svg.hasAttribute('hidden');video.controls=!hidden;svg.toggleAttribute('hidden',!hidden);$('motion-mark-toggle').textContent=hidden?'Hide region outline':'Pause and mark this frame';};
  video.addEventListener('play',()=>{hideMark();motionRegion=null;});
  const point=e=>{const b=svg.getBoundingClientRect();return[Math.max(0,Math.min(1,(e.clientX-b.left)/b.width)),Math.max(0,Math.min(1,(e.clientY-b.top)/b.height))];};
  svg.onpointerdown=e=>{if(!video.paused||shownIndex===null)return;start=point(e);svg.setPointerCapture(e.pointerId);e.preventDefault();};
  svg.onpointermove=e=>{if(!start)return;const p=point(e);motionRegion=[Math.min(p[0],start[0]),Math.min(p[1],start[1]),Math.abs(p[0]-start[0]),Math.abs(p[1]-start[1])];['x','y','width','height'].forEach((k,i)=>$('motion-rect').setAttribute(k,motionRegion[i]));};svg.onpointerup=()=>{start=null;};
  $('motion-pin').onclick=()=>{
    try{
      if(!video.paused||shownIndex===null||!motionRegion||motionRegion[2]<=0||motionRegion[3]<=0)throw Error('Pause a decoded frame and drag a region on it first.');
      const binding=feedbackBinding();if(feedbackDraft&&(feedbackDraft.board!==binding.board||feedbackDraft.comparison!==binding.comparison))throw Error('Return to the existing draft comparison or clear that draft first.');
      if(motionDraft)throw Error('This draft already has a pinned frame. Clear it before choosing another moment.');
      bindFeedbackDraft();motionDraft={review:m.record,video:clip.record,frame_index:shownIndex,timestamp_seconds:clip.frame_times[shownIndex],region:[...motionRegion],panel:$('motion-panel').value};
      $('feedback-form').elements.include_motion.checked=true;$('feedback-details').open=true;showFeedbackDraft();message('Video frame and region pinned to this feedback draft. Playback will not retarget it.');
    }catch(e){message(e.message,true);}
  };
  $('motion-speed').onchange=()=>{
    const other=record($('motion-speed').value),moment=meta.frame_ids?.[shownIndex];
    const mapped=m.speed_matching?.mapping_allowed&&moment?other.metadata.frame_ids?.indexOf(moment):-1;
    motionClipId=other.record;renderMotion();
    if(mapped>=0){$('motion-video').addEventListener('loadedmetadata',()=>{$('motion-video').currentTime=other.playback_times[mapped]+0.000001;},{once:true});message('Speed changed using the same explicit source moment; each clip keeps its own timestamp.');}
    else message('Speed changed. Moment correspondence is unknown or unmatched; no timestamp was copied.');showFeedbackDraft();
  };
}
function setDates(){
  const now=new Date(), day=now.toISOString().slice(0,10);
  document.querySelectorAll('input[type=date]').forEach(e=>{if(!e.value)e.value=day;});
  document.querySelectorAll('input[type=datetime-local]').forEach(e=>{if(!e.value)e.value=new Date(now-now.getTimezoneOffset()*60000).toISOString().slice(0,16);});
}
setDates();Promise.all([loadBoards(),loadWorkflow()]).catch(e=>message(e.message,true));
