'use strict';
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let board = null, regionId = null, reviewId = null, revising = null, dragStart = null;
let feedbackDraft = null;
let workflow = null;
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
  const metadata = Object.fromEntries(['label','source','source_version','camera','display_state','pose'].map(k=>[k,fields[k]||null]));
  metadata.captured_at=fields.captured_at;
  const result = await request('/api/upload',{...scope(),filename:file.name,data:btoa(binary),metadata});
  await reopen(board.board); $('target-form').elements.image.value=result.added.images[0]; updateMarkImage();
  message('Image bytes and source/version metadata pinned. Mark the exact region.');
});
function options(selector, collection, label) {
  document.querySelectorAll(selector).forEach(select=>{
    const previous=Array.from(select.selectedOptions,o=>o.value);
    select.innerHTML=collection.map(r=>`<option value="${r.record}">${esc(label(r))}</option>`).join('');
    if (select.multiple) Array.from(select.options).forEach(o=>o.selected=previous.includes(o.value));
    else if (previous.includes(select.value) || collection.some(r=>r.record===previous[0])) select.value=previous[0];
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
  const result=await operation('compare',{...scope(),target:target().record,plan:plan().record,baseline:f.baseline,trial:f.trial,
    baseline_caption:f.baseline_caption,trial_caption:f.trial_caption,changed:f.changed,unchanged:f.unchanged,
    baseline_region:JSON.parse(f.baseline_region),trial_region:JSON.parse(f.trial_region),
    baseline_approval:f.approval_wording?fact(f.approval_wording,f.approval_date,f.approval_source):null});
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
}
function bindFeedbackDraft() {
  if (!feedbackDraft) feedbackDraft=feedbackBinding();
  showFeedbackDraft();
}
$('feedback-form').addEventListener('input',bindFeedbackDraft);
$('feedback-form').addEventListener('change',bindFeedbackDraft);
$('discard-feedback-draft').onclick=()=>{
  feedbackDraft=null;$('feedback-form').reset();$('correction-fields').hidden=true;setDates();showFeedbackDraft();
};
bindForm('feedback-form',async form=>{
  const r=review();if (!r)throw Error('Display a comparison first.');const f=data(form);
  const binding=feedbackBinding();
  if(feedbackDraft&&(feedbackDraft.board!==binding.board||feedbackDraft.comparison!==binding.comparison))
    throw Error('The displayed comparison changed after this draft began. Return to its original comparison or clear the draft.');
  await operation('submit',{...scope(),comparison:r.record,target:r.target,plan:r.plan,inspection:r.inspection,trial:r.trial,
    source_version:r.source_version,result_version:r.result_version,wording:f.wording,date:f.date,source:f.source,
    correction:form.elements.correct.checked?planData(form):null,historical:form.elements.historical.checked});
  feedbackDraft=null;form.reset();setDates();await reopen(board.board);$('correction-fields').hidden=true;
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
  const relevant=new Set([t?.image,r?.baseline,r?.trial,...(p?.references||[]).map(id=>record(id)?.image)].filter(Boolean));
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
    Array.from(document.querySelectorAll('.image-select')).some(s=>s.value===i.record)),i=>i.metadata.label+' · '+i.metadata.captured_at);
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
function panel(imageId,caption,region,heading){
  const i=record(imageId),m=i.metadata;
  return `<article class="panel"><div class="panel-title"><h3>${heading}</h3></div><div class="image-frame"><img src="${imageUrl(imageId)}" alt="${esc(caption)}"><svg viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden="true"><rect x="${region[0]}" y="${region[1]}" width="${region[2]}" height="${region[3]}"></rect></svg></div><div class="caption">${esc(caption)}</div><div class="meta">${esc(m.captured_at)}<br>Camera: ${esc(typeof m.camera==='object'?JSON.stringify(m.camera):m.camera||'unknown')}<br>State: ${esc(typeof m.display_state==='object'?JSON.stringify(m.display_state):m.display_state||'unknown')}<br>Pose: ${esc(typeof m.pose==='object'?JSON.stringify(m.pose):m.pose||'unknown')}</div>${provenance(i)}</article>`;
}
function renderComparison(){
  const r=review();if(!r){$('comparison').innerHTML='<p>No comparison for this region yet. Add source-bound baseline and isolated trial images.</p>';$('feedback-scope').textContent='No comparison selected.';showFeedbackDraft();return;}
  const t=record(r.target),p=record(r.plan),active=r.record===current()?.comparison;
  const state={};rows('state_facts').filter(f=>f.comparison===r.record).forEach(f=>state[f.facet]=f);
  $('comparison').innerHTML=`${active?'':'<div class="warning">Historical comparison. This target or interpretation has changed. Feedback stays on these exact versions; a new comparison is needed for the current interpretation.</div>'}<div class="${r.matching.status==='matched declared inputs'?'match':'warning'}"><b>${esc(r.matching.status.toUpperCase())}</b><br>${esc([...r.matching.mismatched.map(s=>'Different: '+s),...r.matching.unknown.map(s=>'Unknown: '+s)].join(' · '))}<span class="hint">${esc(r.matching.basis)}</span></div><p><b>Marked target:</b> ${esc(t.wording)}<br><b>Interpretation at this review:</b> ${esc(p.interpretation)}</p><div class="panels">${panel(r.baseline,r.baseline_caption,r.baseline_region,r.baseline_approval?'LEFT · Owner-approved baseline (recorded user fact)':'LEFT · Baseline — approval unknown')}${panel(r.trial,r.trial_caption,r.trial_region,'RIGHT · Isolated trial')}</div><div class="states">${Object.entries(choices).map(([facet])=>`<div><strong>${esc(facet.replaceAll('_',' '))}</strong>${esc(state[facet]?.value||'unknown')}</div>`).join('')}</div><article class="card"><p><b>Changed:</b> ${esc(r.changed)}</p><p><b>Unchanged / unresolved:</b> ${esc(r.unchanged)}</p><p class="hint">An isolated result is not owner-approved without its own explicit user fact. Target agreement and technical verification do not imply acceptance.</p>${r.baseline_approval?`<p class="meta">Baseline user fact: ${esc(r.baseline_approval.wording)} · ${esc(r.baseline_approval.date)} · ${esc(r.baseline_approval.source)}</p>`:''}${provenance(r)}</article>`;
  $('feedback-scope').innerHTML=`<strong>${esc(t.label)} · ${esc(record(r.trial).metadata.label)}</strong><br>Target revision: ${r.target.slice(0,12)} · interpretation: ${r.plan.slice(0,12)}<br>Inspection source version: ${esc(r.source_version)}<br>Result version: ${esc(r.result_version)}<br>${active?'Current exact comparison':'Historical — correction cannot activate from this review'}`;
  showFeedbackDraft();
}
function renderHistory(){
  const entries=[];
  for(const collection of ['targets','plans','comparisons','feedback','agreements','reference_roles','state_facts'])for(const r of rows(collection)){
    if(r.target_id&&r.target_id!==regionId)continue;
    let content='';
    if(collection==='targets')content=`<h3>Exact target wording</h3><blockquote>${esc(r.wording)}</blockquote><p class="hint">${esc(r.label)} · ${esc(r.date)} · ${esc(r.source)}${r.supersedes?' · supersedes a prior target revision':''}</p>`;
    if(collection==='plans')content=`<h3>${r.correction_from_comparison?'Corrected interpretation':'Interpretation / method'}</h3><p>${esc(r.interpretation)}</p><p><b>Preserve:</b> ${esc(r.preserved_features.join('; '))}<br><b>Method:</b> ${esc(r.method)}<br><b>Expected:</b> ${esc(r.expected_appearance)}</p>${r.correction_wording?`<blockquote>${esc(r.correction_wording)}</blockquote>`:''}`;
    if(collection==='comparisons')content=`<h3>Baseline / trial review · ${esc(r.matching.status)}</h3><p>LEFT: ${esc(r.baseline_caption)}<br>RIGHT: ${esc(r.trial_caption)}</p>`;
    if(collection==='feedback')content=`<h3>${r.interpretation_correction?'User correction of interpretation':'Version-bound feedback'}</h3><blockquote>${esc(r.wording)}</blockquote><p class="hint">${esc(r.date)} · ${esc(r.source)}<br>${esc(r.applicability)}</p>`;
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
function setDates(){
  const now=new Date(), day=now.toISOString().slice(0,10);
  document.querySelectorAll('input[type=date]').forEach(e=>{if(!e.value)e.value=day;});
  document.querySelectorAll('input[type=datetime-local]').forEach(e=>{if(!e.value)e.value=new Date(now-now.getTimezoneOffset()*60000).toISOString().slice(0,16);});
}
setDates();Promise.all([loadBoards(),loadWorkflow()]).catch(e=>message(e.message,true));
