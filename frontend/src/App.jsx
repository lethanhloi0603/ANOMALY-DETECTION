import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { Activity, Database, Play, ShieldAlert, UserRoundCog, Zap } from "lucide-react";
import "./styles.css";

const API = "http://localhost:5117";

function ResultCard({ result }) {
  if (!result) return null;
  const routeClass = result.baselineRoute === "personalized" ? "personalized" : "global";
  return (
    <div className="card result">
      <div className="row between"><h2>Latest detection result</h2><span className={`pill ${routeClass}`}>{result.baselineRoute}</span></div>
      <div className="grid2">
        <Metric label="User ID" value={result.userId} />
        <Metric label="Role / Dept" value={`${result.role || "UNKNOWN"} / ${result.department || "UNKNOWN"}`} />
        <Metric label="Log count" value={result.logCount} />
        <Metric label="Active days" value={result.activeDaysCount} />
        <Metric label="Safe personal threshold ready" value={String(result.personalizedReady)} />
        <Metric label="Personal threshold updated" value={String(result.personalizedTrainingCompleted)} />
        <Metric label="Raw score" value={Number(result.score).toFixed(6)} />
        <Metric label="Selected threshold" value={Number(result.threshold).toFixed(6)} />
        <Metric label="Global ratio" value={result.globalRatio == null ? "-" : Number(result.globalRatio).toFixed(3)} />
        <Metric label="Role ratio" value={result.roleRatio == null ? "-" : Number(result.roleRatio).toFixed(3)} />
        <Metric label="Personal ratio" value={result.personalRatio == null ? "-" : Number(result.personalRatio).toFixed(3)} />
        <Metric label="Final anomaly index" value={result.finalAnomalyIndex == null ? "-" : Number(result.finalAnomalyIndex).toFixed(3)} />
        <Metric label="Is anomaly" value={String(result.isAnomaly)} />
      </div>
      {result.warning && <div className="warning">{result.warning}</div>}
    </div>
  );
}
function Metric({ label, value }) { return <div className="metric"><span>{label}</span><strong>{value ?? "-"}</strong></div>; }

function App() {
  const [summary, setSummary] = useState(null), [detections, setDetections] = useState([]), [result, setResult] = useState(null), [status, setStatus] = useState(null);
  const [holdout, setHoldout] = useState([]), [selectedHoldout, setSelectedHoldout] = useState(""), [loading, setLoading] = useState(false);
  const [form, setForm] = useState({ userId:"U-DEMO-001", pcId:"PC-DEMO-001", eventType:"logon", activity:"Logon", timestamp:new Date().toISOString(), content:"normal login" });
  const [bulk, setBulk] = useState({ userId:"U-DEMO-001", pcId:"PC-DEMO-001", count:30 });
  async function api(path, options={}) { const res=await fetch(`${API}${path}`, { headers:{"Content-Type":"application/json"}, ...options }); if(!res.ok) throw new Error(await res.text()); return res.json(); }
  async function refresh(){ try{ const [s,d]=await Promise.all([api("/api/dashboard/summary"), api("/api/detections")]); setSummary(s); setDetections(d); }catch(e){console.error(e);} }
  useEffect(()=>{refresh();},[]);
  async function trainGlobal(){ setLoading(true); try{ const r=await api("/api/training/global",{method:"POST"}); alert(`Global role/group model trained. Windows: ${r.windows}, active days: ${r.active_days || r.activeDays}`); await refresh(); }catch(e){alert(`Train global failed: ${e.message}`);} finally{setLoading(false);} }
  async function submitManual(e){ e.preventDefault(); setLoading(true); try{ const r=await api("/api/logs",{method:"POST",body:JSON.stringify({...form,timestamp:form.timestamp||new Date().toISOString()})}); setResult(r); await refresh(); await loadStatus(form.userId);}catch(e){alert(`Submit failed: ${e.message}`);}finally{setLoading(false);} }
  async function submitBulk(){ setLoading(true); try{ const r=await api("/api/logs/bulk",{method:"POST",body:JSON.stringify({userId:bulk.userId,pcId:bulk.pcId,count:Number(bulk.count),scoreOnlyLast:true})}); setResult(r); await refresh(); await loadStatus(bulk.userId);}catch(e){alert(`Bulk failed: ${e.message}`);}finally{setLoading(false);} }
  async function loadStatus(userId){ try{ setStatus(await api(`/api/users/${encodeURIComponent(userId)}/status`)); }catch(e){console.error(e);} }
  async function loadHoldoutUsers(){ setLoading(true); try{ const data=await api("/api/demo/holdout-users"); const users=data.users||[]; setHoldout(users); if(users.length>0) setSelectedHoldout(users[0].userId); if(data.warning) alert(data.warning);}catch(e){alert(`Load holdout failed: ${e.message}`);}finally{setLoading(false);} }
  async function replay(days){ if(!selectedHoldout){alert("Select holdout user first.");return;} setLoading(true); try{ const r=await api("/api/demo/replay",{method:"POST",body:JSON.stringify({userId:selectedHoldout,count:days,unit:"days"})}); setResult(r); await refresh(); await loadStatus(selectedHoldout);}catch(e){alert(`Replay failed: ${e.message}`);}finally{setLoading(false);} }
  return <div className="page">
    <header className="hero"><div><h1>CERT R4.2 Multi-view Anomaly Framework</h1><p>Count view + raw token sequence encoder + Role → Department → Global calibration → delayed safe personal threshold.</p></div><button disabled={loading} onClick={trainGlobal} className="primary"><Database size={18}/> Train Global Role/Group Baseline</button></header>
    <section className="summary"><Metric label="Total logs" value={summary?.totalLogs??0}/><Metric label="Users" value={summary?.totalUsers??0}/><Metric label="Active days" value={summary?.activeDays??0}/><Metric label="Personalized users" value={summary?.personalizedReadyUsers??0}/><Metric label="Global routes" value={summary?.globalRoutes??0}/><Metric label="Personalized routes" value={summary?.personalizedRoutes??0}/></section>
    <main className="layout"><div className="left">
      <div className="card"><div className="row"><Activity/><h2>Manual log input</h2></div><form onSubmit={submitManual} className="form"><label>User ID<input value={form.userId} onChange={e=>setForm({...form,userId:e.target.value})}/></label><label>PC ID<input value={form.pcId} onChange={e=>setForm({...form,pcId:e.target.value})}/></label><label>Timestamp<input value={form.timestamp} onChange={e=>setForm({...form,timestamp:e.target.value})}/></label><label>Event type<select value={form.eventType} onChange={e=>setForm({...form,eventType:e.target.value})}><option value="logon">logon</option><option value="device">device</option><option value="file">file</option><option value="email">email</option><option value="http">http</option></select></label><label>Activity<input value={form.activity} onChange={e=>setForm({...form,activity:e.target.value})}/></label><label>Content<textarea value={form.content} onChange={e=>setForm({...form,content:e.target.value})}/></label><button disabled={loading} className="primary"><Play size={18}/> Submit 1 log</button></form></div>
      <div className="card"><div className="row"><Zap/><h2>Quick bulk input</h2></div><p className="muted">Synthetic bulk tạo mỗi log cách nhau 1 ngày. 30 active days chỉ là pre-check; personal threshold còn cần delayed safe history.</p><div className="form"><label>User ID<input value={bulk.userId} onChange={e=>setBulk({...bulk,userId:e.target.value})}/></label><label>PC ID<input value={bulk.pcId} onChange={e=>setBulk({...bulk,pcId:e.target.value})}/></label><label>Count / days<input type="number" value={bulk.count} onChange={e=>setBulk({...bulk,count:e.target.value})}/></label><button disabled={loading} onClick={submitBulk} className="secondary">Generate & submit bulk days</button></div></div>
      <div className="card"><div className="row"><UserRoundCog/><h2>Holdout replay demo</h2></div><p className="muted">Replay theo ngày hoạt động. Hệ thống chỉ fit/update threshold sau khi đủ safe days và qua update delay; không train model riêng.</p><button disabled={loading} onClick={loadHoldoutUsers} className="secondary">Load holdout users</button><select value={selectedHoldout} onChange={e=>setSelectedHoldout(e.target.value)}><option value="">Select holdout user</option>{holdout.map(u=><option key={u.userId} value={u.userId}>{u.userId} — {u.activeDays} days — {u.role}</option>)}</select><div className="row wrap"><button disabled={loading||!selectedHoldout} onClick={()=>replay(1)}>Replay 1 active day</button><button disabled={loading||!selectedHoldout} onClick={()=>replay(30)}>Replay 30 active days</button><button disabled={loading||!selectedHoldout} onClick={()=>replay(7)}>Replay 7 more days</button></div></div>
    </div><div className="right"><ResultCard result={result}/>{status&&<div className="card"><h2>User status</h2><div className="grid2"><Metric label="User ID" value={status.userId}/><Metric label="Role" value={status.role}/><Metric label="Department" value={status.department}/><Metric label="Log count" value={status.logCount}/><Metric label="Active days" value={status.activeDaysCount}/><Metric label="Safe personal threshold ready" value={String(status.personalizedReady)}/></div><p className="muted">{status.lastTrainingMessage}</p></div>}<div className="card"><div className="row"><ShieldAlert/><h2>Recent detections</h2></div><div className="table"><div className="thead"><span>User</span><span>Route</span><span>Role</span><span>Index</span><span>Days</span></div>{detections.map(d=><div className="tr" key={d.id}><span>{d.userId}</span><span className={`mini ${d.baselineRoute}`}>{d.baselineRoute}</span><span>{d.role||"-"}</span><span>{d.finalAnomalyIndex==null?"-":Number(d.finalAnomalyIndex).toFixed(2)}</span><span>{d.activeDaysAtPrediction}</span></div>)}</div></div></div></main>{loading&&<div className="loading">Working...</div>}</div>;
}
createRoot(document.getElementById("root")).render(<App/>);
