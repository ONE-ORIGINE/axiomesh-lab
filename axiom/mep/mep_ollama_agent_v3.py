"""
mep_ollama_agent.py  v3.0
════════════════════════════════════════════════════════════════════════════
MEP/2.0  —  Multi-Provider AI Agent in a Structured Causal Environment

Providers supported:
  --provider ollama     (default, local)
  --provider openai     (OpenAI API, needs OPENAI_API_KEY env var)
  --provider anthropic  (Claude API, needs ANTHROPIC_API_KEY env var)

Agent memory injection:
  --inject-memory       Injects last N decisions into prompt (large models only)
                        Default: OFF to preserve context for small models (phi, gemma:2b)
  --memory-size 4       How many past decisions to inject when ON (default 4)

Key design: memory + situation tone are OPTIONAL because small models
(phi-mini, gemma:2b) have tight context windows and the extra tokens
cause them to ignore instructions. Large models (gemma3:12b, llama3, gpt-4o,
claude-3) benefit from the added context.

MEP innovations in this file:
  • Multi-LLM same environment  — multiple agents, different providers, same env
  • Causal gravity ranking       — actions sorted by semantic affinity to context
  • Circumstance gating          — zero-cost Boolean algebra on action visibility
  • Circuit breaker (per agent)  — CLOSED/OPEN/HALF_OPEN, auto-recovery
  • Robust parser                — strips [CATEGORY] prefix, fuzzy-matches, retries
  • Error budget                 — N failures → graceful demo-mode degradation
  • WHY / WHY-NOT explainability — full causal ancestry on demand
  • Phenomenon engine            — emergent detection from reaction stream patterns

Usage:
  # Local Ollama (auto-detect, demo fallback):
  python mep_ollama_agent.py

  # Specific model with memory injection:
  python mep_ollama_agent.py --model gemma3:12b --inject-memory

  # OpenAI:
  python mep_ollama_agent.py --provider openai --model gpt-4o-mini

  # Anthropic:
  python mep_ollama_agent.py --provider anthropic --model claude-3-haiku-20240307

  # Demo mode (no LLM):
  python mep_ollama_agent.py --demo
"""

from __future__ import annotations

import asyncio, json, math, os, re, sys, textwrap, time, traceback, uuid
import argparse, urllib.request, urllib.error
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

# ─── ANSI ─────────────────────────────────────────────────────────────────

class C:
    R="\033[0m"; B="\033[1m"; D="\033[2m"
    CY="\033[96m"; G="\033[92m"; Y="\033[93m"; RD="\033[91m"; MG="\033[95m"

def hr(n=68,ch="═",c=C.D): print(f"{c}{ch*n}{C.R}")
def section(t): hr(); print(f"{C.B}{C.CY}  {t}{C.R}"); hr()
def ok(l,m):   print(f"  {C.G}[{l}]{C.R} ✓ {m}")
def err(l,m):  print(f"  {C.RD}[{l}]{C.R} ✗ {m}")
def warn(l,m): print(f"  {C.Y}[{l}]{C.R} ! {m}")
def info(l,m): print(f"  {C.CY}[{l}]{C.R} → {m}")
def dim(l,m):  print(f"  {C.D}[{l}]{C.R}   {m}")

# ─── ERRORS ───────────────────────────────────────────────────────────────

class MepError(Exception): pass
class LlmError(MepError): pass
class LlmTimeout(LlmError): pass
class LlmEmpty(LlmError): pass
class LlmParse(LlmError): pass
class LlmCircuitOpen(LlmError): pass

# ─── CIRCUIT BREAKER ──────────────────────────────────────────────────────

class CBState(Enum):
    CLOSED=auto(); OPEN=auto(); HALF_OPEN=auto()

class CircuitBreaker:
    def __init__(self, threshold:int=4, cooldown:float=30.0):
        self._t=threshold; self._cd=cooldown
        self._fails=0; self._last=0.0; self._state=CBState.CLOSED

    @property
    def state(self)->CBState:
        if self._state==CBState.OPEN and time.time()-self._last>=self._cd:
            self._state=CBState.HALF_OPEN
        return self._state

    def ok(self):  self._fails=0; self._state=CBState.CLOSED
    def fail(self):
        self._fails+=1; self._last=time.time()
        if self._fails>=self._t:
            self._state=CBState.OPEN
            warn("CIRCUIT", f"OPEN after {self._fails} failures ({self._cd:.0f}s cooldown)")

    def allow(self)->bool: return self.state in (CBState.CLOSED, CBState.HALF_OPEN)

    def line(self)->str:
        s=self.state
        col={CBState.CLOSED:C.G,CBState.OPEN:C.RD,CBState.HALF_OPEN:C.Y}[s]
        return f"{col}●{C.R} {s.name.lower()} ({self._fails} fails)"

# =============================================================================
# SENSE VECTOR — 8-dim semantic space, cosine affinity
# =============================================================================

DIMS=8

@dataclass(frozen=True)
class Sense:
    dim:str; meaning:str; mag:float; vec:Tuple[float,...]
    @classmethod
    def of(cls,d:str,m:str,axis:int,mag:float=1.0):
        v=[0.]*DIMS; v[axis]=mag; return cls(d,m,mag,tuple(v))
    def aff(self,o:"Sense")->float:
        dot=sum(a*b for a,b in zip(self.vec,o.vec))
        na=math.sqrt(sum(x*x for x in self.vec))
        nb=math.sqrt(sum(x*x for x in o.vec))
        return dot/(na*nb) if na>0 and nb>0 else 0.
    def dist(self,o:"Sense")->float:
        return math.acos(max(-1.,min(1.,self.aff(o))))/math.pi
    @classmethod
    def norm(cls,m,g=1.): return cls.of("normative",m,3,g)
    @classmethod
    def temp(cls,m,g=1.): return cls.of("temporal",m,1,g)
    @classmethod
    def tech(cls,m,g=1.): return cls.of("technical",m,6,g)
    @classmethod
    def soc(cls,m,g=1.):  return cls.of("social",m,4,g)

EMPTY=Sense("none","",0.,tuple([0.]*DIMS))

# =============================================================================
# CIRCUMSTANCE — Composable boolean gate
# =============================================================================

class Circ:
    def __init__(self,cid,desc,fn,role="enabler",w=1.0):
        self.id=cid; self.description=desc; self._fn=fn; self.role=role; self.w=w
    def eval(self,ctx,frame)->bool:
        try: return bool(self._fn(ctx,frame))
        except: return False
    def __and__(self,o): return Circ(f"{self.id}&{o.id}",f"({self.description}) AND ({o.description})",
        lambda c,f:self._fn(c,f) and o._fn(c,f))
    def __or__(self,o):  return Circ(f"{self.id}|{o.id}",f"({self.description}) OR ({o.description})",
        lambda c,f:self._fn(c,f) or o._fn(c,f))
    def __invert__(self): return Circ(f"!{self.id}",f"NOT {self.description}",
        lambda c,f:not self._fn(c,f),"blocker",-self.w)

def flag(cid,desc,key,val=True): return Circ(cid,desc,lambda ctx,_:ctx.data.get(key)==val)
def always(cid): return Circ(cid,cid,lambda *_:True)

# =============================================================================
# CONTEXT FRAME & ENVELOPE
# =============================================================================

@dataclass
class CtxFrame:
    ctx_id : str = field(default_factory=lambda:str(uuid.uuid4()))
    name   : str = ""
    kind   : str = "semantic"
    basis  : Sense = field(default_factory=lambda:EMPTY)
    depth  : int = 0
    data   : Dict[str,Any] = field(default_factory=dict)
    elements: List[Dict]   = field(default_factory=list)
    circs  : List[Circ]    = field(default_factory=list)
    _acts  : List[Dict]    = field(default_factory=list, repr=False)

    def add_circ(self,c): self.circs.append(c); return self
    def include(self,e):  self.elements.append(e); return self
    def set(self,k,v):    self.data[k]=v; return self

    def reg_action(self,a,filt=None):
        self._acts.append({"a":a,"f":filt}); return self

    def available(self,actor,frame)->List[Tuple]:
        out=[]
        for e in self._acts:
            a,f=e["a"],e["f"]
            if f and not f(actor): continue
            if not all(g.eval(self,frame) for g in a.guards): continue
            out.append((a, round(a.sense.aff(self.basis),4)))
        out.sort(key=lambda x:-x[1]); return out

    def eval_circs(self,frame)->List[Dict]:
        return [{"id":c.id,"desc":c.description,
                 "holds":c.eval(self,frame),"role":c.role} for c in self.circs]

    def envelope(self,actor,frame,events=(),situation=None,
                 memory:Optional[List[Dict]]=None,   # None = don't inject
                 inject_memory:bool=False,
                 phenomena=())->"Envelope":
        avail=self.available(actor,frame)
        circs=self.eval_circs(frame)
        attn=[]
        for c in circs:
            if not c["holds"] and c["role"]=="enabler":
                attn.append(f"'{c['desc']}' must become true to unlock more actions")
        for p in phenomena:
            if p.get("magnitude",0)>0.5:
                attn.append(f"Phenomenon '{p['name']}' at {p['magnitude']:.0%} — consider stabilizing")
        return Envelope(
            ctx_id=self.ctx_id, name=self.name, kind=self.kind, depth=self.depth,
            data=dict(self.data), circs=circs, situation=situation,
            events=list(events)[-5:],
            memory=(memory or [])[-4:] if inject_memory else [],
            attention=attn,
            actions=[{"type":a.type,"cat":a.cat,"desc":a.desc,"g":g,"dim":a.sense.dim}
                     for a,g in avail])

@dataclass
class Envelope:
    ctx_id:str=""; name:str=""; kind:str=""; depth:int=0
    data:Dict=field(default_factory=dict)
    circs:List[Dict]=field(default_factory=list)
    actions:List[Dict]=field(default_factory=list)
    situation:Optional[Dict]=None; events:List[Dict]=field(default_factory=list)
    memory:List[Dict]=field(default_factory=list)
    attention:List[str]=field(default_factory=list)

    @property
    def valid_types(self)->set: return {a["type"] for a in self.actions}

    def to_prompt(self,inject_sit_tone:bool=True)->str:
        holding=[c for c in self.circs if c["holds"]]
        missing=[c for c in self.circs if not c["holds"] and c["role"]=="enabler"]

        # ── Actions (type only — the crucial fix for LLM confusion) ───────
        acts="\n".join(
            f"  {i+1}. {a['type']}  g={a['g']:.3f}  [{a['cat']}] {a['desc']}"
            for i,a in enumerate(self.actions)
        ) or "  (none available)"

        hold="\n".join(f"  ✓ {c['desc']}" for c in holding) or "  (none)"
        miss="\n".join(f"  ✗ {c['desc']}" for c in missing) or "  (none)"
        dat="\n".join(f"  {k}: {v}" for k,v in self.data.items()
                      if not k.startswith("_")) or "  (empty)"

        sit_line=""
        tone=""
        if self.situation and inject_sit_tone:
            s=self.situation; k=s.get("kind","")
            sit_line=f"\nSITUATION: {k.upper()} sev={s.get('severity','?')} sat={s.get('saturation',0):.0%}"
            if k=="critical":  tone="\n⚠ CRITICAL — prioritize stabilization actions"
            elif k=="degraded":tone="\n! DEGRADED — environment needs attention"

        evts=""
        if self.events:
            evts="\nRECENT EVENTS:\n"+"\n".join(f"  • {e.get('type','')} — {e.get('s','')}"
                                                  for e in self.events)

        mem=""
        if self.memory:  # only present if inject_memory=True
            mem="\nYOUR LAST ACTIONS (don't repeat failures):\n"+"\n".join(
                f"  [{m['st']}] {m['a']} ← {m['g'][:50]}" for m in self.memory)

        attn=""
        if self.attention:
            attn="\nATTENTION:\n"+"\n".join(f"  ⚡ {a}" for a in self.attention)

        return textwrap.dedent(f"""
            CONTEXT: {self.name} [{self.kind}] depth={self.depth}{sit_line}{tone}

            ACTIVE NOW: {hold}
            MISSING:    {miss}
            DATA:
            {dat}
            {evts}{mem}{attn}

            AVAILABLE ACTIONS (copy type exactly — no brackets, no prefix):
            {acts}
        """).strip()

# =============================================================================
# ACTION & REACTION
# =============================================================================

@dataclass
class Reaction:
    atype:str; status:str; msg:Optional[str]=None; result:Any=None
    impact:str="actor"; target:Optional[str]=None
    chain:List[str]=field(default_factory=list)
    rid:str=field(default_factory=lambda:str(uuid.uuid4()))

    @property
    def ok(self)->bool: return self.status=="success"

    def line(self)->str:
        c=C.G if self.ok else (C.Y if self.status=="rejected" else C.RD)
        i="✓" if self.ok else "✗"
        r=f" → {json.dumps(self.result,default=str)[:80]}" if self.result else ""
        m=f" | {self.msg}" if self.msg else ""
        return f"{c}{i} {self.atype} → {self.status}{m}{r}{C.R}"

@dataclass
class Action:
    type:str; cat:str; desc:str; sense:Sense
    guards:List[Circ]=field(default_factory=list)
    can_chain:bool=True; _h:Any=field(default=None,repr=False)

    async def run(self,actor,payload,ctx,frame)->Reaction:
        for g in self.guards:
            if not g.eval(ctx,frame):
                return Reaction(self.type,"rejected",f"Circumstance not met: {g.description}")
        if not self._h: return Reaction(self.type,"success","OK (no handler)")
        try:   return await self._h(actor,payload,ctx,frame)
        except Exception as e: return Reaction(self.type,"error",str(e))

# =============================================================================
# CAUSAL GRAPH
# =============================================================================

@dataclass
class CNode:
    nid:str; ntype:str; atype:str; actor:str
    ctx:str; corr:str; cause:Optional[str]; depth:int
    ts:float; status:str; summary:str

class CausalGraph:
    def __init__(self): self._n:Dict[str,CNode]={}
    def add(self,n:CNode): self._n[n.nid]=n
    def ancestry(self,nid)->List[CNode]:
        ch,cur=[],nid
        while cur and (n:=self._n.get(cur)):
            ch.insert(0,n); cur=n.cause or ""
        return ch
    def explain(self,atype)->str:
        ms=[n for n in self._n.values() if n.atype==atype]
        if not ms: return f"No record for '{atype}'"
        ch=self.ancestry(ms[-1].nid)
        return "\n".join(f"  [d={n.depth}] {n.ntype}: {n.atype} ({n.status}) — {n.summary}"
                         for n in ch)
    @property
    def stats(self)->Dict:
        return {"nodes":len(self._n),
                "ok":sum(1 for n in self._n.values() if n.status=="success"),
                "rej":sum(1 for n in self._n.values() if n.status=="rejected"),
                "types":list({n.atype for n in self._n.values()})}

# =============================================================================
# PHENOMENON DETECTOR
# =============================================================================

class PhenomDetector:
    def __init__(self): self._h:List[Tuple[float,str]]=[]; self._det:List[Dict]=[]
    def rec(self,t): self._h.append((time.time(),t)); self._h=self._h[-500:]
    def detect(self,pt,thresh,ws)->Optional[Dict]:
        cut=time.time()-ws; m=[x for x,rt in self._h if rt==pt and x>=cut]
        if len(m)>=thresh:
            p={"name":f"{pt}.phenomenon","mag":min(1.,len(m)/thresh),"count":len(m)}
            if not any(d["name"]==p["name"] for d in self._det[-3:]): self._det.append(p)
            return p
        return None
    @property
    def all(self): return self._det

# =============================================================================
# LLM PROVIDERS
# =============================================================================

class LLMProvider:
    """Base. Subclasses implement _raw_chat()."""
    def __init__(self, model:str, timeout:float, retries:int):
        self.model=model; self.timeout=timeout; self.retries=retries
        self.cb=CircuitBreaker()
        self._avail=False

    async def chat(self, sys_p:str, user_p:str, temp:float=0.1)->str:
        if not self.cb.allow(): raise LlmCircuitOpen("Circuit OPEN")
        last=RuntimeError("no attempt")
        for i in range(1,self.retries+1):
            try:
                r=await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None,self._raw,sys_p,user_p,temp),
                    timeout=self.timeout)
                if not r or not r.strip(): raise LlmEmpty("Empty response")
                self.cb.ok(); return r
            except asyncio.TimeoutError:
                last=LlmTimeout(f"Timeout after {self.timeout:.0f}s (attempt {i})")
                warn("LLM",f"Timeout {i}/{self.retries}")
            except LlmEmpty as e:
                last=e; warn("LLM",f"Empty {i}/{self.retries}")
            except (LlmCircuitOpen,LlmTimeout): raise
            except Exception as e:
                last=LlmError(str(e)); warn("LLM",f"Error {i}: {e}")
            if i<self.retries: await asyncio.sleep(2**(i-1))
        self.cb.fail(); raise last

    def _raw(self,sys_p,user_p,temp)->str: raise NotImplementedError

    @property
    def available(self)->bool: return self._avail
    def provider_name(self)->str: return self.__class__.__name__

# ── Ollama ────────────────────────────────────────────────────────────────

class OllamaProvider(LLMProvider):
    def __init__(self,host,model,timeout,retries):
        super().__init__(model,timeout,retries); self.host=host
        self._lib=None
        try:
            import ollama as _l; self._lib=_l
            dim("OLLAMA","Library found")
        except ImportError:
            dim("OLLAMA","No library — using HTTP")
        self._avail=self._probe()

    def _probe(self)->bool:
        try:
            if self._lib:
                ms=self._lib.list(); ns=[m.get("name","") for m in (ms.get("models",[]) if isinstance(ms,dict) else [])]
                info("OLLAMA",f"Connected  models={ns[:4]}"); return True
            req=urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req,timeout=3) as r:
                d=json.loads(r.read()); ns=[m.get("name","") for m in d.get("models",[])]
                info("OLLAMA",f"Connected  models={ns[:4]}"); return True
        except Exception as e:
            warn("OLLAMA",f"Not reachable ({e})"); return False

    def _raw(self,sys_p,user_p,temp)->str:
        if self._lib:
            r=self._lib.chat(model=self.model,
                             messages=[{"role":"system","content":sys_p},
                                       {"role":"user","content":user_p}],
                             options={"temperature":temp})
            return (r.get("message",{}).get("content","") if isinstance(r,dict) else str(r))
        body=json.dumps({"model":self.model,
                         "messages":[{"role":"system","content":sys_p},
                                     {"role":"user","content":user_p}],
                         "options":{"temperature":temp},"stream":False}).encode()
        req=urllib.request.Request(f"{self.host}/api/chat",data=body,
            headers={"Content-Type":"application/json"},method="POST")
        with urllib.request.urlopen(req,timeout=self.timeout) as r:
            return json.loads(r.read())["message"]["content"]

# ── OpenAI ────────────────────────────────────────────────────────────────

class OpenAIProvider(LLMProvider):
    BASE="https://api.openai.com/v1/chat/completions"
    def __init__(self,model,timeout,retries):
        super().__init__(model,timeout,retries)
        self._key=os.environ.get("OPENAI_API_KEY","")
        if not self._key: warn("OPENAI","OPENAI_API_KEY not set")
        self._avail=bool(self._key)

    def _raw(self,sys_p,user_p,temp)->str:
        if not self._key: raise LlmError("OPENAI_API_KEY not set")
        body=json.dumps({"model":self.model,"temperature":temp,
            "messages":[{"role":"system","content":sys_p},
                        {"role":"user","content":user_p}]}).encode()
        req=urllib.request.Request(self.BASE,data=body,
            headers={"Content-Type":"application/json",
                     "Authorization":f"Bearer {self._key}"},method="POST")
        with urllib.request.urlopen(req,timeout=self.timeout) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]

    def provider_name(self)->str: return f"OpenAI({self.model})"

# ── Anthropic ─────────────────────────────────────────────────────────────

class AnthropicProvider(LLMProvider):
    BASE="https://api.anthropic.com/v1/messages"
    def __init__(self,model,timeout,retries):
        super().__init__(model,timeout,retries)
        self._key=os.environ.get("ANTHROPIC_API_KEY","")
        if not self._key: warn("ANTHROPIC","ANTHROPIC_API_KEY not set")
        self._avail=bool(self._key)

    def _raw(self,sys_p,user_p,temp)->str:
        if not self._key: raise LlmError("ANTHROPIC_API_KEY not set")
        body=json.dumps({"model":self.model,"max_tokens":512,"temperature":temp,
            "system":sys_p,
            "messages":[{"role":"user","content":user_p}]}).encode()
        req=urllib.request.Request(self.BASE,data=body,
            headers={"Content-Type":"application/json",
                     "x-api-key":self._key,
                     "anthropic-version":"2023-06-01"},method="POST")
        with urllib.request.urlopen(req,timeout=self.timeout) as r:
            d=json.loads(r.read())
            return d["content"][0]["text"]

    def provider_name(self)->str: return f"Anthropic({self.model})"

def build_provider(args)->LLMProvider:
    p=args.provider.lower()
    if p=="openai":    return OpenAIProvider(args.model,args.timeout,args.retries)
    if p=="anthropic": return AnthropicProvider(args.model,args.timeout,args.retries)
    return OllamaProvider(args.host,args.model,args.timeout,args.retries)

# =============================================================================
# RESPONSE PARSER — robust, strips LLM formatting artifacts
# =============================================================================

_CAT=re.compile(r"^\s*\[(?:LIFECYCLE|COMMAND|QUERY|SIGNAL|TRANSFORM)\]\s*",re.I)

class Parser:
    @classmethod
    def parse(cls,raw:str,valid:set)->Tuple[Optional[Dict],Optional[str]]:
        if not raw or not raw.strip(): return None,"Empty response"
        txt=cls._strip(raw.strip())
        d=cls._json(txt)
        if d is None: return None,f"No JSON in: {txt[:100]!r}"
        # Normalise action_type — strip category prefix
        at=_CAT.sub("",str(d.get("action_type",""))).strip()
        d["action_type"]=at
        dec=d.get("decision","").strip().lower()
        if dec not in ("execute","skip","query"):
            if at in valid: d["decision"]="execute"; dec="execute"
            else: return None,f"Unknown decision {dec!r}"
        if dec=="execute":
            if not at: return None,"action_type is empty"
            if at not in valid:
                # Fuzzy
                cands=[t for t in valid if at in t or t in at]
                if len(cands)==1: d["action_type"]=cands[0]; d["_fuzzy"]=True
                else: return None,f"'{at}' not in [{', '.join(sorted(valid))}]"
        return d,None

    @staticmethod
    def _strip(t:str)->str:
        m=re.search(r"```(?:json)?\s*(.*?)```",t,re.DOTALL)
        return m.group(1).strip() if m else t

    @staticmethod
    def _json(t:str)->Optional[Dict]:
        try: return json.loads(t)
        except: pass
        s=t.find("{")
        if s==-1: return None
        depth=end=-1
        for i,ch in enumerate(t[s:],s):
            if ch=="{": depth=depth+1 if depth>=0 else 1
            elif ch=="}":
                depth-=1
                if depth==0: end=i; break
        if end==-1: return None
        try: return json.loads(t[s:end+1])
        except: return None

# =============================================================================
# ESCHOOL ENVIRONMENT
# =============================================================================

class Env:
    def __init__(self):
        self.name="ESchool"
        self._stu:Dict[str,Dict]={}; self._tea:Dict[str,Dict]={}
        self._adm:Dict[str,Dict]={}; self._crs:Dict[str,Dict]={}
        self._sch:Dict[str,Dict]={}; self._grd:List[Dict]=[]
        self._enr:List[Dict]=[]; self._evts:List[Dict]=[]
        self._cg=CausalGraph(); self._pd=PhenomDetector()
        self._gd={"enrollmentOpen":False}

        # Circumstances
        self.C_SYS  =always("system.active")
        self.C_ENRL =flag("enrollment.open","Enrollment period is active","enrollmentOpen")

        # Contexts
        self.root =self._mk_root()
        self.adm  =self._mk_admin()
        self.acad =self._mk_acad()
        self.enrl =self._mk_enrl()

    # ── Context builders ──────────────────────────────────────────────────

    def _mk_root(self)->CtxFrame:
        c=CtxFrame(name="ESchool.Root",kind="global",basis=Sense.tech("global",0.3))
        c.add_circ(self.C_SYS); return c

    def _mk_admin(self)->CtxFrame:
        c=CtxFrame(name="ESchool.Admin",kind="semantic",
                   basis=Sense.norm("administrative ops",0.9),depth=1)
        c.add_circ(self.C_SYS)
        for a in [self._a_create_school(), self._a_create_course(),
                  self._a_assign_teacher(), self._a_open_enrl(),
                  self._a_close_enrl(), self._a_snapshot(), self._a_list()]:
            c.reg_action(a)
        return c

    def _mk_acad(self)->CtxFrame:
        c=CtxFrame(name="ESchool.Academic",kind="semantic",
                   basis=Sense.norm("academic grading",0.95),depth=1)
        c.add_circ(self.C_SYS)
        for a in [self._a_grade(), self._a_qgrades(), self._a_snapshot()]:
            c.reg_action(a)
        return c

    def _mk_enrl(self)->CtxFrame:
        c=CtxFrame(name="ESchool.Enrollment",kind="temporal",
                   basis=Sense.temp("enrollment period",0.9),depth=1)
        c.add_circ(self.C_SYS); c.add_circ(self.C_ENRL)
        for a in [self._a_enroll(), self._a_withdraw(), self._a_qenrl()]:
            c.reg_action(a)
        return c

    # ── Actions ───────────────────────────────────────────────────────────

    def _a_create_school(self)->Action:
        async def h(actor,p,ctx,frame):
            n=p.get("name","School"); code=p.get("code",uuid.uuid4().hex[:6].upper())
            sid=str(uuid.uuid4())
            self._sch[sid]={"id":sid,"name":n,"code":code,"country":p.get("countryCode","INT")}
            self._emit("school.created",f"'{n}' ({code})")
            return Reaction("school.create","success",f"School '{n}' created",
                {"id":sid,"name":n},"environment")
        return Action("school.create","lifecycle","Create a new school institution",
                      Sense.norm("school creation",0.9),_h=h)

    def _a_create_course(self)->Action:
        async def h(actor,p,ctx,frame):
            n=p.get("name"); cap=int(p.get("capacity",30))
            if not n: return Reaction("course.create","rejected","Missing 'name'")
            cid=str(uuid.uuid4())
            self._crs[cid]={"id":cid,"name":n,"code":p.get("code",uuid.uuid4().hex[:6].upper()),
                             "disc":p.get("discipline","General"),"maxCap":cap,
                             "credits":int(p.get("credits",3)),"enrolled":0,
                             "students":[],"grades":[]}
            self._emit("course.created",f"'{n}' cap={cap}")
            return Reaction("course.create","success",f"Course '{n}' created",
                {"id":cid,"name":n,"cap":cap},"environment")
        return Action("course.create","lifecycle","Create a new course",
                      Sense.tech("course provisioning",0.8),_h=h)

    def _a_assign_teacher(self)->Action:
        async def h(actor,p,ctx,frame):
            t=self._tea.get(p.get("teacherId","")); c=self._crs.get(p.get("courseId",""))
            if not t: return Reaction("course.assign-teacher","rejected","Teacher not found")
            if not c: return Reaction("course.assign-teacher","rejected","Course not found")
            c["teacher"]=t["name"]
            self._emit("teacher.assigned",f"{t['name']} → {c['name']}")
            return Reaction("course.assign-teacher","success",f"{t['name']} → {c['name']}",
                impact="specific",target=c["id"])
        return Action("course.assign-teacher","command","Assign teacher to course",
                      Sense.soc("teacher assignment",0.7),_h=h)

    def _a_open_enrl(self)->Action:
        async def h(actor,p,ctx,frame):
            self._gd["enrollmentOpen"]=True
            for cx in [self.root,self.adm,self.acad,self.enrl]: cx.set("enrollmentOpen",True)
            self._emit("enrollment.opened","Enrollment period OPEN")
            return Reaction("enrollment.open","success","Enrollment opened",impact="environment")
        return Action("enrollment.open","command","Open the enrollment period",
                      Sense.temp("enrollment activation",0.85),_h=h)

    def _a_close_enrl(self)->Action:
        async def h(actor,p,ctx,frame):
            self._gd["enrollmentOpen"]=False
            for cx in [self.root,self.adm,self.acad,self.enrl]: cx.set("enrollmentOpen",False)
            self._emit("enrollment.closed","Enrollment period CLOSED")
            return Reaction("enrollment.close","success","Enrollment closed",impact="environment")
        return Action("enrollment.close","command","Close the enrollment period",
                      Sense.temp("enrollment close",0.7),_h=h)

    def _a_enroll(self)->Action:
        async def h(actor,p,ctx,frame):
            s=self._stu.get(p.get("studentId","")); c=self._crs.get(p.get("courseId",""))
            if not s: return Reaction("student.enroll","rejected","Student not found")
            if not c: return Reaction("student.enroll","rejected","Course not found")
            if c["enrolled"]>=c["maxCap"]: return Reaction("student.enroll","rejected",f"At capacity ({c['maxCap']})")
            if s["id"] in c["students"]: return Reaction("student.enroll","rejected","Already enrolled")
            c["students"].append(s["id"]); c["enrolled"]+=1
            s.setdefault("courses",[]).append(c["id"])
            self._enr.append({"s":s["id"],"c":c["id"],"at":time.time()})
            self._emit("enrollment.confirmed",f"{s['name']} → {c['name']}")
            return Reaction("student.enroll","success",f"{s['name']} enrolled in {c['name']}",
                {"student":s["name"],"course":c["name"]},"specific",c["id"])
        return Action("student.enroll","command","Enroll a student in a course",
                      Sense.temp("enrollment",0.95),guards=[self.C_ENRL],_h=h)

    def _a_withdraw(self)->Action:
        async def h(actor,p,ctx,frame):
            s=self._stu.get(p.get("studentId","")); c=self._crs.get(p.get("courseId",""))
            if not s or not c: return Reaction("student.withdraw","rejected","Not found")
            if s["id"] not in c["students"]: return Reaction("student.withdraw","rejected","Not enrolled")
            c["students"].remove(s["id"]); c["enrolled"]-=1
            if c["id"] in s.get("courses",[]): s["courses"].remove(c["id"])
            self._emit("withdrawal.confirmed",f"{s['name']} withdrew from {c['name']}")
            return Reaction("student.withdraw","success",f"{s['name']} withdrew",
                impact="specific",target=c["id"])
        return Action("student.withdraw","command","Withdraw a student from a course",
                      Sense.temp("withdrawal",0.7),_h=h)

    def _a_grade(self)->Action:
        async def h(actor,p,ctx,frame):
            s=self._stu.get(p.get("studentId","")); c=self._crs.get(p.get("courseId",""))
            try: g=float(p.get("grade",0))
            except: return Reaction("grade.record","rejected","Invalid grade")
            if not s: return Reaction("grade.record","rejected","Student not found")
            if not c: return Reaction("grade.record","rejected","Course not found")
            if not (0<=g<=100): return Reaction("grade.record","rejected","Grade must be 0-100")
            lt="A"if g>=90 else"B"if g>=80 else"C"if g>=70 else"D"if g>=60 else"F"
            rec={"s":s["id"],"c":c["id"],"g":g,"l":lt,"at":time.time()}
            self._grd.append(rec); c["grades"].append(rec)
            gg=[r["g"] for r in self._grd if r["s"]==s["id"]]
            s["gpa"]=round(sum(gg)/len(gg),2)
            passing=g>=60
            self._emit("grade.recorded",f"{s['name']}: {g:.1f} ({lt}) in {c['name']}")
            self._pd.rec("fail" if not passing else "pass")
            ph=self._pd.detect("fail",3,600)
            if ph: warn("PHENOMENON",f"MassFailure mag={ph['mag']:.0%} n={ph['count']}")
            return Reaction("grade.record","success",f"{s['name']}: {g:.1f} ({lt})",
                {"grade":g,"letter":lt,"passing":passing,"gpa":s["gpa"]},
                "specific",c["id"],chain=[] if passing else ["notification.dispatch"])
        return Action("grade.record","command","Record a student grade",
                      Sense.norm("academic assessment",0.95),_h=h)

    def _a_qgrades(self)->Action:
        async def h(actor,p,ctx,frame):
            s=self._stu.get(p.get("studentId",""))
            if not s: return Reaction("student.query-grades","rejected","Not found")
            gs=[r for r in self._grd if r["s"]==s["id"]]
            return Reaction("student.query-grades","success",f"{len(gs)} grades for {s['name']}",
                {"name":s["name"],"gpa":s.get("gpa",0),"grades":gs})
        return Action("student.query-grades","query","Query grades for a student",
                      Sense.norm("grade query",0.6),_h=h)

    def _a_qenrl(self)->Action:
        async def h(actor,p,ctx,frame):
            c=self._crs.get(p.get("courseId",""))
            if not c: return Reaction("enrollment.query","rejected","Course not found")
            stus=[self._stu[sid]["name"] for sid in c["students"] if sid in self._stu]
            return Reaction("enrollment.query","success",f"{c['name']}: {c['enrolled']}/{c['maxCap']}",
                {"name":c["name"],"enrolled":c["enrolled"],"cap":c["maxCap"],"students":stus})
        return Action("enrollment.query","query","Query enrollment for a course",
                      Sense.temp("enrollment status",0.6),_h=h)

    def _a_snapshot(self)->Action:
        async def h(actor,p,ctx,frame):
            return Reaction("system.snapshot","success","Snapshot",
                {"schools":len(self._sch),"courses":len(self._crs),"students":len(self._stu),
                 "teachers":len(self._tea),"grades":len(self._grd),
                 "enrollmentOpen":self._gd.get("enrollmentOpen"),
                 "phenomena":len(self._pd.all)})
        return Action("system.snapshot","query","Snapshot the environment state",
                      Sense.tech("snapshot",0.3),_h=h)

    def _a_list(self)->Action:
        async def h(actor,p,ctx,frame):
            return Reaction("system.list","success","Element list",
                {"students":list(self._stu.values()),
                 "courses":list(self._crs.values()),
                 "teachers":list(self._tea.values())})
        return Action("system.list","query","List all environment elements",
                      Sense.tech("element listing",0.2),_h=h)

    # ── Public ────────────────────────────────────────────────────────────

    def add_student(self,n,code)->Dict:
        sid=str(uuid.uuid4()); e={"id":sid,"name":n,"code":code,"type":"Student","gpa":0,"courses":[]}
        self._stu[sid]=e; return e

    def add_teacher(self,n,code,disc)->Dict:
        tid=str(uuid.uuid4()); e={"id":tid,"name":n,"code":code,"type":"Teacher","disc":disc}
        self._tea[tid]=e; return e

    def add_admin(self,n,code,role="SystemAdmin")->Dict:
        aid=str(uuid.uuid4()); e={"id":aid,"name":n,"code":code,"type":"Admin","role":role}
        self._adm[aid]=e; return e

    def _emit(self,t,s): self._evts.append({"type":t,"s":s,"at":time.time()})

    async def dispatch(self,actor,atype,payload,ctx,corr)->Reaction:
        frame={"actor_id":actor["id"],"payload":payload}
        for e in ctx._acts:
            if e["a"].type==atype:
                r=await e["a"].run(actor,payload,ctx,frame)
                self._cg.add(CNode(str(uuid.uuid4()),"action",atype,actor["id"],
                    ctx.name,corr,None,0,time.time(),r.status,r.msg or ""))
                return r
        return Reaction(atype,"rejected",f"'{atype}' not registered in '{ctx.name}'")

    def envelope(self,actor,ctx,mem=None,inject_mem=False)->Envelope:
        frame={"actor_id":actor["id"]}
        ctx.data.update(self._gd)
        sit={"kind":"operational" if not self._pd.all else "degraded",
             "severity":"low" if not self._pd.all else "high",
             "saturation":min(1,len(self._stu)/max(1,len(self._stu)+50))}
        return ctx.envelope(actor,frame,events=self._evts[-6:],situation=sit,
                            memory=mem,inject_memory=inject_mem,phenomena=self._pd.all)

# =============================================================================
# AGENT
# =============================================================================

SYS=textwrap.dedent("""
You are an AI agent operating in a structured educational environment (ESchool) via MEP protocol.

You receive a CONTEXT that shows available actions, active circumstances, and environment state.

Respond ONLY with a JSON object, nothing else:
{"decision":"execute","action_type":"<EXACT type from list>","payload":{},"reasoning":"<brief>"}
OR:
{"decision":"skip","action_type":"","payload":{},"reasoning":"<why>"}

Rules:
- action_type must be copied EXACTLY from AVAILABLE ACTIONS list
- No brackets, no category prefix, no extra text before or after JSON
- Only choose actions listed under AVAILABLE ACTIONS
""").strip()

SYS_RETRY_SUFFIX="\n\nPREVIOUS ATTEMPT FAILED. Return ONLY the raw JSON object. No text outside {}."

class Agent:
    def __init__(self,env:Env,llm:LLMProvider,actor:Dict,
                 demo:bool=False,aid:str="",
                 inject_memory:bool=False,inject_sit_tone:bool=True):
        self.env=env; self.llm=llm; self.actor=actor
        self.demo=demo; self.aid=aid or str(uuid.uuid4())[:6]
        self.inject_memory=inject_memory
        self.inject_sit_tone=inject_sit_tone
        self._mem:List[Dict]=[]; self._budget=6

    async def decide(self,goal:str,ctx:CtxFrame)->Optional[Dict]:
        if self.demo: return self._demo(goal,ctx)
        if self._budget<=0:
            warn(self.aid,"Budget exhausted → demo mode"); self.demo=True
            return self._demo(goal,ctx)

        env=self.env.envelope(self.actor,ctx,
                              mem=self._mem,inject_mem=self.inject_memory)
        last_err=None

        for attempt in range(1,4):
            prompt=env.to_prompt(inject_sit_tone=self.inject_sit_tone)
            if last_err: prompt+=f"\n\nPREVIOUS ERROR: {last_err}{SYS_RETRY_SUFFIX}"
            user=f"GOAL: {goal}\n\n{prompt}\n\nDecide now."
            try:
                raw=await self.llm.chat(SYS,user,temp=0.05*attempt)
            except LlmCircuitOpen as e:
                err(self.aid,str(e)); self.demo=True; return self._demo(goal,ctx)
            except LlmError as e:
                err(self.aid,str(e)); self._budget-=1; return None

            d,pe=Parser.parse(raw,env.valid_types)
            if d:
                if d.get("_fuzzy"): warn(self.aid,f"Fuzzy match → {d['action_type']}")
                return d
            last_err=pe
            warn(self.aid,f"Parse attempt {attempt}/3: {pe}")

        err(self.aid,"3 parse attempts failed"); self._budget-=1; return None

    async def act(self,goal:str,ctx:CtxFrame,
                  extra:Optional[Dict]=None)->Optional[Reaction]:
        d=await self.decide(goal,ctx)
        if not d:
            err(self.aid,f"No decision for: {goal[:50]}"); return None
        if d.get("decision")=="skip":
            dim(self.aid,f"SKIP — {d.get('reasoning','')[:70]}"); return None

        atype=d["action_type"]; payload={**d.get("payload",{}),**(extra or {})}
        corr=str(uuid.uuid4())

        print()
        info(f"AGENT/{self.aid}",f"{C.B}{atype}{C.R}  "
             f"{C.D}g={self._grav(atype,ctx):.3f}{C.R}")
        dim("REASON",d.get("reasoning","")[:90])
        if payload: dim("PAYLOAD",json.dumps(payload,default=str)[:110])

        r=await self.env.dispatch(self.actor,atype,payload,ctx,corr)
        if r.ok: ok("REACTION",r.line())
        else:    err("REACTION",r.line())
        if r.chain: dim("CHAIN→",f"spawned: {', '.join(r.chain)}")

        self._mem.append({"g":goal[:50],"a":atype,"st":r.status,"at":time.time()})
        if len(self._mem)>20: self._mem=self._mem[-20:]
        return r

    def _demo(self,goal:str,ctx:CtxFrame)->Optional[Dict]:
        env=self.env.envelope(self.actor,ctx)
        if not env.actions:
            return {"decision":"skip","action_type":"","payload":{},"reasoning":"No actions"}
        gl=goal.lower()
        for kw,at,pl in [
            ("create school","school.create",{"name":"Demo Academy","code":"DEM","countryCode":"FR"}),
            ("create course","course.create",{}),
            ("open enrollment","enrollment.open",{}),
            ("close enrollment","enrollment.close",{}),
            ("enroll","student.enroll",{}),
            ("withdraw","student.withdraw",{}),
            ("grade","grade.record",{}),
            ("snapshot","system.snapshot",{}),
            ("list","system.list",{}),
            ("query grade","student.query-grades",{}),
            ("query enroll","enrollment.query",{}),
            ("assign teacher","course.assign-teacher",{}),
        ]:
            if kw in gl and at in env.valid_types:
                return {"decision":"execute","action_type":at,"payload":pl,
                        "reasoning":f"Keyword '{kw}' matched"}
        best=env.actions[0]
        return {"decision":"execute","action_type":best["type"],"payload":{},
                "reasoning":"Highest-gravity action"}

    def _grav(self,at:str,ctx:CtxFrame)->float:
        e=self.env.envelope(self.actor,ctx)
        for a in e.actions:
            if a["type"]==at: return a["g"]
        return 0.

# =============================================================================
# MAIN
# =============================================================================

async def main(args):
    hr(70,"╔"); hr(70,"║")
    print(f"  {C.B}{C.CY}MEP/2.0 v3.0  —  Multi-Provider Causal AI Agent{C.R}")
    hr(70,"║"); hr(70,"╚"); print()

    env=Env()
    llm=build_provider(args)

    demo=args.demo or not llm.available
    if not demo: info("MODE",f"Live: {llm.provider_name()}  timeout={args.timeout}s  retries={args.retries}")
    else:        warn("MODE","Demo mode (no LLM)")
    if args.inject_memory: info("MEMORY",f"Injection ON (size={args.memory_size})")
    else:                  dim("MEMORY","Injection OFF (small-model safe)")

    # Seed
    admin  =env.add_admin("Dr. Elena Vasquez","ADM001")
    teacher=env.add_teacher("Prof. Sophie Laurent","TCH001","Mathematics")
    env.add_teacher("Prof. James Kim","TCH002","Computer Science")
    students=[env.add_student(n,c) for n,c in [
        ("Alice Chen","STU001"),("Bob Martinez","STU002"),
        ("Clara Okonkwo","STU003"),("David Park","STU004")]]
    for a in [admin,teacher]:
        for ctx in [env.adm,env.acad,env.enrl]: ctx.include(a)

    ag_admin=Agent(env,llm,admin,demo,aid="ADMIN",
                   inject_memory=args.inject_memory,inject_sit_tone=True)
    ag_teach=Agent(env,llm,teacher,demo,aid="TEACH",
                   inject_memory=args.inject_memory,inject_sit_tone=True)

    info("BOOT",f"{len(env._stu)+len(env._tea)+len(env._adm)} elements seeded"); print()

    # ═══ PHASE 1: Envelope showcase ═══════════════════════════════════════
    section("PHASE 1 — Context Envelope: what the AI receives")

    e=env.envelope(admin,env.adm,inject_mem=args.inject_memory)
    print(f"\n{C.B}Available Actions (Causal Gravity):{C.R}")
    for a in e.actions:
        bar="█"*max(1,int(a["g"]*20))
        print(f"  [{bar:<20}] {a['g']:.3f}  {C.B}{a['type']}{C.R}  {C.D}{a['desc']}{C.R}")

    print(f"\n{C.B}Prompt excerpt sent to {llm.provider_name()}:{C.R}")
    print(C.D+textwrap.indent(e.to_prompt()[:500],"  ")+C.R+" [...]")

    # ═══ PHASE 2: Administrative ═══════════════════════════════════════════
    section("PHASE 2 — Administrative Agent: Setup")

    await ag_admin.act("Create a new school called Global Science Academy",env.adm,
        extra={"name":"Global Science Academy","code":"GSA","countryCode":"FR"})

    course_ids=[]
    for spec in [
        {"name":"Advanced Mathematics","code":"MATH201","discipline":"Mathematics","capacity":"25","credits":"4"},
        {"name":"Data Structures",     "code":"CS301",  "discipline":"Computer Science","capacity":"30","credits":"3"},
        {"name":"Physics Lab",         "code":"PHY101", "discipline":"Physics","capacity":"20","credits":"3"},
    ]:
        r=await ag_admin.act(f"Create course {spec['name']}",env.adm,extra=spec)
        if r and r.ok and r.result: course_ids.append(r.result.get("id",""))

    ok("STATUS",f"{len(env._crs)} courses registered")

    # ═══ PHASE 3: Circumstance gating ═════════════════════════════════════
    section("PHASE 3 — Circumstance Gating")

    print(f"\n{C.B}Enrollment context, enrollmentOpen=False:{C.R}")
    closed=env.envelope(admin,env.enrl)
    if not closed.actions: print(f"  {C.RD}✗ Zero actions — enrollment.open=False gates them{C.R}")
    else:
        for a in closed.actions: print(f"  {C.Y}~{C.R} {a['type']}  (not gated by enrollment.open)")

    await ag_admin.act("Open the enrollment period for students",env.adm)

    print(f"\n{C.B}Enrollment context, enrollmentOpen=True:{C.R}")
    for a in env.envelope(admin,env.enrl).actions:
        print(f"  {C.G}✓{C.R} {a['type']}  g={a['g']:.3f}")

    # ═══ PHASE 4: Enrollment ════════════════════════════════════════════════
    section("PHASE 4 — Enrollment Operations")

    if course_ids:
        mid,cid=course_ids[0], course_ids[1] if len(course_ids)>1 else course_ids[0]
        for s in students[:3]:
            await ag_admin.act(f"Enroll {s['name']} in Advanced Mathematics",env.enrl,
                extra={"studentId":s["id"],"courseId":mid})
        for s in students[:2]:
            await ag_admin.act(f"Enroll {s['name']} in Data Structures",env.enrl,
                extra={"studentId":s["id"],"courseId":cid})

        mc=env._crs.get(mid,{}); cc=env._crs.get(cid,{})
        ok("STATUS",f"Math: {mc.get('enrolled',0)}/{mc.get('maxCap',0)}  "
                    f"CS: {cc.get('enrolled',0)}/{cc.get('maxCap',0)}")

    # ═══ PHASE 5: Grading ═══════════════════════════════════════════════════
    section("PHASE 5 — Teacher Agent: Grading + Causal Chains")

    await ag_admin.act("Close the enrollment period",env.adm)
    dim("INFO","Failing grades (<60) spawn chain → notification.dispatch"); print()

    if course_ids:
        for sid,cid,g in [
            (students[0]["id"],course_ids[0],88.5),
            (students[1]["id"],course_ids[0],42.0),
            (students[2]["id"],course_ids[0],55.0),
            (students[0]["id"],course_ids[1] if len(course_ids)>1 else course_ids[0],93.0),
            (students[1]["id"],course_ids[1] if len(course_ids)>1 else course_ids[0],37.0),
            (students[2]["id"],course_ids[1] if len(course_ids)>1 else course_ids[0],61.0),
        ]:
            s=env._stu.get(sid,{})
            await ag_teach.act(f"Record grade {g} for {s.get('name','')}",env.acad,
                extra={"studentId":sid,"courseId":cid,"grade":str(g)})

    # ═══ PHASE 6: WHY / WHY-NOT ═════════════════════════════════════════════
    section("PHASE 6 — WHY / WHY-NOT Explainability")

    print(f"\n{C.B}Causal trace — 'grade.record':{C.R}")
    print(C.D+env._cg.explain("grade.record")+C.R)
    s=env._cg.stats
    print(f"\n{C.B}Causal graph:{C.R}  nodes={s['nodes']}  ok={s['ok']}  rej={s['rej']}")

    # ═══ PHASE 7: Topology ══════════════════════════════════════════════════
    section("PHASE 7 — Context Topology")

    pairs=[("Admin","Academic",env.adm.basis,env.acad.basis),
           ("Admin","Enrollment",env.adm.basis,env.enrl.basis),
           ("Academic","Enrollment",env.acad.basis,env.enrl.basis)]
    print(f"\n{'Pair':<32} {'Dist':>6}  Proximity")
    hr(55,"─")
    for na,nb,a,b in pairs:
        d=a.dist(b); bar="█"*int((1-d)*24)
        print(f"  {na} ↔ {nb:<20} {d:>6.3f}  {bar}")

    # ═══ PHASE 8: Multi-LLM power demo ════════════════════════════════════
    section("PHASE 8 — Multi-LLM Demo: Two Agents, One Environment")

    print(f"""
  {C.B}Key MEP Innovation:{C.R}
  Multiple AI agents (different models, different roles) coexist in
  ONE environment, naturally scoped by context + circumstances.

  {C.B}ADMIN agent{C.R} sees: school.create, course.create, enrollment controls
  {C.B}TEACHER agent{C.R} sees: grade.record, query-grades

  No isolation needed. The environment itself is the boundary.
  This is the difference between MEP and MCP.
""")

    # Second "AI" (demo simulating a second provider)
    ag2=Agent(env,llm,teacher,demo=True,aid="TEACH2",inject_memory=False)
    await ag2.act("Take a snapshot of the academic state",env.acad)

    # ═══ PHASE 9: Final ═════════════════════════════════════════════════════
    section("PHASE 9 — Final State")

    r=await ag_admin.act("Take a system snapshot",env.adm)
    if r and r.result:
        print(f"\n  {C.B}Environment:{C.R}")
        for k,v in r.result.items(): print(f"    {k:<22}: {v}")

    print(f"\n  {C.B}Students:{C.R}")
    for s in students:
        d=env._stu.get(s["id"],{}); g=d.get("gpa",0)
        bar="█"*max(0,int(g/10))
        print(f"    {s['name']:<20} GPA={g:>5.2f} [{bar:<10}]")

    if env._pd.all:
        print(f"\n  {C.B}Phenomena:{C.R}")
        for p in env._pd.all:
            print(f"    {C.MG}★ {p['name']}{C.R}  mag={p['mag']:.0%}  n={p['count']}")

    print(f"\n  {C.B}Circuit Breaker:{C.R}  {llm.cb.line()}")
    print(f"  Error budget: {ag_admin._budget}/6")

    print(f"\n  {C.B}Decision Log:{C.R}")
    for m in ag_admin._mem+ag_teach._mem:
        ic=f"{C.G}✓{C.R}" if m["st"]=="success" else f"{C.RD}✗{C.R}"
        print(f"    {ic} {m['a']:<36} ← {C.D}{m['g'][:45]}{C.R}")

    hr()
    print(f"\n  {C.B}{C.CY}MEP/2.0 v3.0 complete.{C.R}")
    hr()

if __name__=="__main__":
    ap=argparse.ArgumentParser(description="MEP/2.0 v3 Multi-Provider AI Agent")
    ap.add_argument("--demo",   action="store_true")
    ap.add_argument("--provider",default="ollama",choices=["ollama","openai","anthropic"])
    ap.add_argument("--model",  default="llama3")
    ap.add_argument("--host",   default="http://localhost:11434")
    ap.add_argument("--timeout",type=float,default=45.0)
    ap.add_argument("--retries",type=int,  default=3)
    ap.add_argument("--inject-memory",action="store_true",
                    help="Inject agent memory into prompts (large models only, default OFF)")
    ap.add_argument("--memory-size", type=int,default=4)
    args=ap.parse_args()

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print(f"\n{C.Y}Interrupted.{C.R}")
    except Exception as e:
        print(f"\n{C.RD}Fatal: {e}{C.R}"); traceback.print_exc(); sys.exit(1)
