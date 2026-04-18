"""
mep.py  —  Model Environment Protocol  v3.0
════════════════════════════════════════════════════════════════════════════
OneOrigine / ImperialSchool Research  —  I.S. License (Attributed Open)

MEP is the causal protocol between probabilistic AI and structured environments.
It implements the rational superstratum over statistical LLMs.

Key protocol objects:
  ContextEnvelope   — structured semantic frame delivered to AI
  MepParser         — robust JSON extractor with fuzzy matching
  MepGateway        — manages sessions, dispatch, explain
  MepSession        — stateful per-agent context + memory
  CircuitBreaker    — CLOSED/OPEN/HALF_OPEN resilience

Provider support (zero mandatory dependencies):
  OllamaProvider    — local LLM (lib or HTTP)
  OpenAIProvider    — OpenAI API (OPENAI_API_KEY)
  AnthropicProvider — Anthropic API (ANTHROPIC_API_KEY)

Mathematical grounding (from EDP axiom system):
  Harmony delivered to AI: H(A,C,S) per available action
  Causal ancestry: WHY/WHY-NOT queries on any reaction
  Context topology: distance matrix for multi-context navigation
"""

from __future__ import annotations

import asyncio, json, math, os, re, textwrap, time, uuid
import urllib.request, urllib.error
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

from edp import (
    Environment, Context, Action, Reaction, Element, Circumstance,
    SenseVector, HarmonyProfile, PhenomenonPattern,
    EnvironmentKind, ContextKind, ActionCategory,
    ReactionStatus, ImpactScope, Temporality,
    RawData, ContextualizedData, SENSE_NULL,
)

__version__ = "3.0.0"
__author__  = "OneOrigine"
__license__ = "I.S."

# ─── Circuit Breaker ──────────────────────────────────────────────────────────

class CBState(Enum):
    CLOSED=auto(); OPEN=auto(); HALF_OPEN=auto()

class CircuitBreaker:
    def __init__(self, threshold: int = 4, cooldown: float = 30.0):
        self._t=threshold; self._cd=cooldown
        self._fails=0; self._last=0.; self._state=CBState.CLOSED

    @property
    def state(self) -> CBState:
        if self._state==CBState.OPEN and time.time()-self._last>=self._cd:
            self._state=CBState.HALF_OPEN
        return self._state

    def ok(self):  self._fails=0; self._state=CBState.CLOSED
    def fail(self):
        self._fails+=1; self._last=time.time()
        if self._fails>=self._t: self._state=CBState.OPEN

    def allow(self) -> bool: return self.state != CBState.OPEN

    def status(self) -> str:
        s = self.state
        col = {"CLOSED":"✓","OPEN":"✗","HALF_OPEN":"◐"}[s.name]
        return f"{col} {s.name.lower()} ({self._fails} fails)"


# ─── MEP Errors ───────────────────────────────────────────────────────────────

class MepError(Exception): pass
class LlmError(MepError): pass
class LlmTimeout(LlmError): pass
class LlmEmpty(LlmError): pass
class LlmCircuitOpen(LlmError): pass
class ParseError(MepError): pass


# ─── ContextEnvelope ──────────────────────────────────────────────────────────

@dataclass
class ContextEnvelope:
    """
    The fundamental MEP unit: a structured semantic frame for the AI.
    Contains available actions ranked by H(A,C,S), active circumstances,
    situation snapshot, causal history, and reactive attention map.

    Format designed to answer: WHERE AM I? WHAT CAN I DO? WHY? WHAT JUST HAPPENED?
    """
    ctx_id     : str = ""
    name       : str = ""
    kind       : str = "semantic"
    depth      : int = 0
    data       : Dict[str,Any]  = field(default_factory=dict)
    circumstances: List[Dict]   = field(default_factory=list)
    actions    : List[Dict]     = field(default_factory=list)  # with harmony scores
    harmony_map: Dict[str,Dict] = field(default_factory=dict)  # full harmony per action
    situation  : Optional[Dict] = None
    events     : List[Dict]     = field(default_factory=list)
    memory     : List[Dict]     = field(default_factory=list)  # injected if enabled
    attention  : List[str]      = field(default_factory=list)  # ⚡ focus signals
    phenomena  : List[Dict]     = field(default_factory=list)

    @property
    def valid_types(self) -> set:
        return {a["type"] for a in self.actions}

    def to_prompt(self, inject_memory: bool = False,
                  inject_sit_tone: bool = True) -> str:
        """Convert envelope to LLM-ready structured prompt."""
        holding = [c for c in self.circumstances if c.get("holds")]
        missing = [c for c in self.circumstances
                   if not c.get("holds") and c.get("role")=="enabler"]

        # Actions with harmony scores (the key innovation over MCP)
        acts = "\n".join(
            f"  {i+1:2}. {a['type']:<38} H={a.get('score',a.get('gravity',0)):.3f}"
            f"  [{a.get('category','?')}] {a['description']}"
            for i, a in enumerate(self.actions)
        ) or "  (none — check MISSING circumstances)"

        hold_str = "\n".join(f"  ✓ {c['desc']}" for c in holding) or "  (none)"
        miss_str = "\n".join(f"  ✗ {c['desc']}" for c in missing) or "  (none)"
        dat_str  = "\n".join(f"  {k}: {v}" for k,v in self.data.items()
                             if not k.startswith("_")) or "  (empty)"

        sit_line = tone = ""
        if self.situation and inject_sit_tone:
            s=self.situation; k=s.get("kind","")
            sit_line = f"\nSITUATION: {k.upper()} sev={s.get('severity','?')} sat={s.get('saturation',0):.0%}"
            if k=="critical":   tone="\n⚠ CRITICAL — prioritize stabilization actions"
            elif k=="degraded": tone="\n! DEGRADED — emerging phenomena detected"

        evts = ""
        if self.events:
            evts = "\nRECENT EVENTS:\n" + "\n".join(
                f"  • {e.get('type','')} — {e.get('summary','')}"
                for e in self.events[-4:])

        mem = ""
        if inject_memory and self.memory:
            mem = "\nYOUR LAST ACTIONS:\n" + "\n".join(
                f"  [{m.get('st','?')}] {m.get('a','')} ← {m.get('g','')[:45]}"
                for m in self.memory[-4:])

        attn = ""
        if self.attention:
            attn = "\nATTENTION:\n" + "\n".join(f"  ⚡ {a}" for a in self.attention)

        phen = ""
        if self.phenomena:
            active = [p for p in self.phenomena if not p.get("dissolved")]
            if active:
                phen = "\nACTIVE PHENOMENA:\n" + "\n".join(
                    f"  ★ {p.get('name','')} mag={p.get('magnitude',0):.0%}"
                    for p in active)

        return textwrap.dedent(f"""
            CONTEXT: {self.name} [{self.kind}] depth={self.depth}{sit_line}{tone}

            HOLDING CIRCUMSTANCES (active right now):
            {hold_str}

            MISSING (these block some actions):
            {miss_str}

            CONTEXT DATA:
            {dat_str}
            {evts}{mem}{attn}{phen}

            AVAILABLE ACTIONS — H=harmony score (copy type exactly, no prefix):
            {acts}
        """).strip()

    def to_json(self) -> str: return json.dumps(self.__dict__, default=str)

    def derive(self, name: str, kind: str="semantic") -> "ContextEnvelope":
        return ContextEnvelope(name=name, kind=kind,
                               data=dict(self.data), depth=self.depth+1)


# ─── Response Parser ──────────────────────────────────────────────────────────

_CAT_PREFIX = re.compile(
    r"^\s*\[(?:LIFECYCLE|COMMAND|QUERY|SIGNAL|TRANSFORM)\]\s*", re.I)

class MepParser:
    """
    Robust LLM response parser.
    Handles: markdown fences, [CATEGORY] prefixes, partial JSON, fuzzy match.
    Returns (decision_dict, error_str).
    """

    @classmethod
    def parse(cls, raw: str, valid: set) -> Tuple[Optional[Dict], Optional[str]]:
        if not raw or not raw.strip(): return None, "Empty response"
        txt = cls._strip_fences(raw.strip())
        d   = cls._extract_json(txt)
        if d is None: return None, f"No JSON in: {txt[:120]!r}"

        # Normalize action_type — strip [CATEGORY] prefix
        at = _CAT_PREFIX.sub("", str(d.get("action_type",""))).strip()
        d["action_type"] = at

        dec = d.get("decision","").strip().lower()
        if dec not in ("execute","skip","query"):
            if at in valid: d["decision"]="execute"; dec="execute"
            else: return None, f"Unknown decision: {dec!r}"

        if dec == "execute":
            if not at: return None, "action_type is empty"
            if at not in valid:
                # Fuzzy match
                cands = [t for t in valid if at in t or t in at]
                if len(cands)==1:
                    d["action_type"]=cands[0]; d["_fuzzy"]=True
                else:
                    return None, (f"'{at}' not in valid types: "
                                  f"[{', '.join(sorted(valid))}]")
        return d, None

    @staticmethod
    def _strip_fences(t: str) -> str:
        m = re.search(r"```(?:json)?\s*(.*?)```", t, re.DOTALL)
        return m.group(1).strip() if m else t

    @staticmethod
    def _extract_json(t: str) -> Optional[Dict]:
        try: return json.loads(t)
        except: pass
        s = t.find("{")
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


# ─── LLM Providers ───────────────────────────────────────────────────────────

class LLMProvider:
    """Abstract base — chat() with timeout + retry + circuit breaker."""
    def __init__(self, model: str, timeout: float, retries: int):
        self.model=model; self.timeout=timeout; self.retries=retries
        self.cb=CircuitBreaker(); self._avail=False

    async def chat(self, sys_p: str, user_p: str, temp: float=0.1) -> str:
        if not self.cb.allow(): raise LlmCircuitOpen("Circuit OPEN")
        last: Exception = RuntimeError("no attempt")
        for i in range(1, self.retries+1):
            try:
                r = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, self._raw, sys_p, user_p, temp),
                    timeout=self.timeout)
                if not r or not r.strip(): raise LlmEmpty("Empty response")
                self.cb.ok(); return r
            except asyncio.TimeoutError:
                last=LlmTimeout(f"Timeout {self.timeout:.0f}s (attempt {i})")
            except LlmEmpty as e: last=e
            except (LlmCircuitOpen,): raise
            except Exception as e: last=LlmError(str(e))
            if i < self.retries: await asyncio.sleep(2**(i-1))
        self.cb.fail(); raise last

    def _raw(self, sys_p: str, user_p: str, temp: float) -> str:
        raise NotImplementedError

    @property
    def available(self) -> bool: return self._avail
    def provider_name(self) -> str: return self.__class__.__name__


class OllamaProvider(LLMProvider):
    def __init__(self, host: str="http://localhost:11434",
                 model: str="llama3", timeout: float=45., retries: int=3):
        super().__init__(model, timeout, retries)
        self.host=host; self._lib=None
        try:
            import ollama as _l; self._lib=_l
        except ImportError: pass
        self._avail=self._probe()

    def _probe(self) -> bool:
        try:
            if self._lib:
                self._lib.list(); return True
            req=urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req,timeout=3): return True
        except: return False

    def _raw(self, sys_p, user_p, temp) -> str:
        if self._lib:
            r=self._lib.chat(model=self.model,
                messages=[{"role":"system","content":sys_p},
                           {"role":"user","content":user_p}],
                options={"temperature":temp})
            return (r.get("message",{}).get("content","")
                    if isinstance(r,dict) else str(r))
        body=json.dumps({"model":self.model,"stream":False,"options":{"temperature":temp},
                "messages":[{"role":"system","content":sys_p},
                             {"role":"user","content":user_p}]}).encode()
        req=urllib.request.Request(f"{self.host}/api/chat",data=body,
            headers={"Content-Type":"application/json"},method="POST")
        with urllib.request.urlopen(req,timeout=self.timeout) as r:
            return json.loads(r.read())["message"]["content"]

    def provider_name(self): return f"Ollama({self.model})"


class OpenAIProvider(LLMProvider):
    BASE = "https://api.openai.com/v1/chat/completions"
    def __init__(self, model: str="gpt-4o-mini", timeout: float=30., retries: int=3):
        super().__init__(model, timeout, retries)
        self._key=os.environ.get("OPENAI_API_KEY","")
        self._avail=bool(self._key)

    def _raw(self, sys_p, user_p, temp) -> str:
        if not self._key: raise LlmError("OPENAI_API_KEY not set")
        body=json.dumps({"model":self.model,"temperature":temp,
            "messages":[{"role":"system","content":sys_p},
                         {"role":"user","content":user_p}]}).encode()
        req=urllib.request.Request(self.BASE,data=body,
            headers={"Content-Type":"application/json",
                     "Authorization":f"Bearer {self._key}"},method="POST")
        with urllib.request.urlopen(req,timeout=self.timeout) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]

    def provider_name(self): return f"OpenAI({self.model})"


class AnthropicProvider(LLMProvider):
    BASE = "https://api.anthropic.com/v1/messages"
    def __init__(self, model: str="claude-3-haiku-20240307",
                 timeout: float=30., retries: int=3):
        super().__init__(model, timeout, retries)
        self._key=os.environ.get("ANTHROPIC_API_KEY","")
        self._avail=bool(self._key)

    def _raw(self, sys_p, user_p, temp) -> str:
        if not self._key: raise LlmError("ANTHROPIC_API_KEY not set")
        body=json.dumps({"model":self.model,"max_tokens":512,"temperature":temp,
            "system":sys_p,
            "messages":[{"role":"user","content":user_p}]}).encode()
        req=urllib.request.Request(self.BASE,data=body,
            headers={"Content-Type":"application/json",
                     "x-api-key":self._key,
                     "anthropic-version":"2023-06-01"},method="POST")
        with urllib.request.urlopen(req,timeout=self.timeout) as r:
            return json.loads(r.read())["content"][0]["text"]

    def provider_name(self): return f"Anthropic({self.model})"


def make_provider(provider: str, model: str, host: str,
                  timeout: float, retries: int) -> LLMProvider:
    p = provider.lower()
    if p=="openai":    return OpenAIProvider(model, timeout, retries)
    if p=="anthropic": return AnthropicProvider(model, timeout, retries)
    return OllamaProvider(host, model, timeout, retries)


# ─── MepSession ───────────────────────────────────────────────────────────────

class MepSession:
    """
    Stateful per-agent session. Maintains:
      • Decision memory (for --inject-memory option)
      • Error budget (graceful degradation)
      • Subscriptions (proactive notifications)
    """
    def __init__(self, session_id: str, client_id: str):
        self.session_id = session_id
        self.client_id  = client_id
        self.created_at = time.time()
        self._memory: List[Dict] = []
        self._budget: int = 6
        self._subs: Dict[str, List[Callable]] = {}

    def record(self, goal: str, action: str, status: str):
        self._memory.append({"g":goal[:50],"a":action,"st":status,"at":time.time()})
        if len(self._memory)>50: self._memory=self._memory[-50:]

    def consume_budget(self, n: int=1): self._budget=max(0, self._budget-n)
    @property
    def budget_ok(self) -> bool: return self._budget > 0
    @property
    def recent_memory(self) -> List[Dict]: return self._memory[-5:]

    def subscribe(self, event: str, cb: Callable) -> Callable:
        self._subs.setdefault(event,[]).append(cb)
        def unsub(): self._subs[event].remove(cb)
        return unsub

    def notify(self, event: str, data: Any):
        for cb in self._subs.get(event,[])+self._subs.get("*",[]):
            try: cb(data)
            except: pass


# ─── MepGateway ───────────────────────────────────────────────────────────────

class MepGateway:
    """
    Connects AI agents (MEP clients) to an EDP environment (MEP server).

    Manages:
      • Session lifecycle
      • ContextEnvelope construction (with harmony map)
      • Action dispatch through environment
      • Causal explainability (WHY/WHY-NOT)
      • Proactive phenomenon notifications
    """

    def __init__(self, env: Environment):
        self._env  = env
        self._sessions: Dict[str, MepSession] = {}
        # Subscribe gateway to environment events
        self._env.on_phenomenon(self._on_phenomenon)
        self._env.on_event(self._on_event)

    def connect(self, client_id: str) -> MepSession:
        s = MepSession(str(uuid.uuid4()), client_id)
        self._sessions[s.session_id] = s
        return s

    def disconnect(self, session_id: str):
        self._sessions.pop(session_id, None)

    def build_envelope(self, session: MepSession, actor: Element,
                        ctx: Context,
                        inject_memory: bool = False,
                        current_sense: SenseVector = None) -> ContextEnvelope:
        """
        Build the ContextEnvelope — the structured semantic frame for the AI.
        Includes full harmony H(A,C,S) per action, not just gravity.
        """
        frame = {"actor_id": actor.element_id}
        ctx.data.update(self._env._elements.get(actor.element_id,actor).__dict__
                         if False else {})  # sync

        # Get actions with full harmony profile
        available = ctx.get_available_actions(actor.to_dict(), frame, current_sense)
        circ_evals = [c.evaluate_with_trace(ctx, frame) for c in ctx.circumstances]

        # Attention map: blocked enablers + high-magnitude phenomena
        attention = []
        for ce in circ_evals:
            if ce.blocker:
                attention.append(f"'{ce.circumstance.description}' must become true")

        # Phenomena
        phenomena = [p.to_dict() for p in self._env.phenomena if p.is_active]
        for p in phenomena:
            if p.get("magnitude",0) > 0.5:
                attention.append(f"Phenomenon '{p.get('name','')}' "
                                 f"at {p.get('magnitude',0):.0%}")

        # Situation
        snap = self._env.snapshot()
        sit  = {"kind": snap.get("situation","operational"),
                "severity": "high" if snap.get("phenomena",0)>0 else "low",
                "saturation": min(1., len(self._env.elements)/max(1,100))}

        return ContextEnvelope(
            ctx_id=ctx.ctx_id, name=ctx.name, kind=ctx.kind.value,
            depth=ctx.depth,
            data={k:v for k,v in ctx.data.items() if not k.startswith("_")},
            circumstances=[{"id":ce.circumstance.id,
                             "desc":ce.circumstance.description,
                             "holds":ce.holds,"role":ce.circumstance.role,
                             "blocker":ce.blocker} for ce in circ_evals],
            actions=[a.to_dict(ctx, current_sense) for a,_ in available],
            harmony_map={a.type: h.to_dict() for a,h in available},
            situation=sit,
            events=self._env.recent_events[-5:],
            memory=session.recent_memory if inject_memory else [],
            attention=attention,
            phenomena=phenomena,
        )

    async def dispatch(self, session: MepSession, actor: Element,
                        action_type: str, payload: Dict,
                        ctx: Context, chain_depth: int=0) -> Reaction:
        r = await self._env.dispatch(actor, action_type, payload,
                                      ctx, str(uuid.uuid4()), chain_depth)
        session.record(payload.get("_goal", action_type), action_type, r.status.value)
        session.notify("reaction", r)
        return r

    def explain_why(self, action_type: str) -> str:
        return self._env.explain_why(action_type)

    def explain_why_not(self, element: Element,
                         action_type: str, ctx: Context) -> List[str]:
        """WHY-NOT: which circumstances are blocking this action?"""
        frame = {"actor_id": element.element_id}
        for entry in ctx._actions:
            if entry["a"].type == action_type:
                a = entry["a"]
                return [ce.reason for ce in
                        [g.evaluate_with_trace(ctx, frame) for g in a.guards]
                        if not ce.holds]
        return [f"Action '{action_type}' not registered in '{ctx.name}'"]

    def context_topology(self) -> List[Dict]:
        """Return pairwise distances between all registered contexts."""
        ctxs = self._env._contexts
        result = []
        for i, a in enumerate(ctxs):
            for b in ctxs[i+1:]:
                result.append({
                    "ctx_a": a.name, "ctx_b": b.name,
                    "distance": round(a.distance_to(b), 4),
                    "dim_a": a.basis.dimension, "dim_b": b.basis.dimension
                })
        return result

    def _on_phenomenon(self, p) -> None:
        for s in self._sessions.values():
            s.notify("phenomenon", p)

    def _on_event(self, e: Dict) -> None:
        for s in self._sessions.values():
            s.notify("event", e)

    @property
    def env(self) -> Environment: return self._env


# ─── MepAgent — base AI agent ────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an AI agent operating inside a structured environment via MEP protocol.

You receive a CONTEXT ENVELOPE showing:
  • Available actions with harmony scores H (ranked highest first)
  • Active/missing circumstances
  • Environment state and phenomena

H = α·context_alignment + β·semantic_alignment (prefer higher H)

Respond ONLY with valid JSON:
{"decision":"execute","action_type":"<exact type>","payload":{},"reasoning":"<brief>"}
OR
{"decision":"skip","action_type":"","payload":{},"reasoning":"<why>"}

CRITICAL:
  - action_type must be copied EXACTLY from AVAILABLE ACTIONS list
  - No prefix, no brackets, no text outside the JSON object
  - Only choose actions listed under AVAILABLE ACTIONS
""").strip()

RETRY_SUFFIX = "\n\nPREVIOUS ERROR — Return ONLY the raw JSON object."


class MepAgent:
    """
    AI agent operating via MEP.
    Implements: decide → parse → validate → dispatch → record → return.
    """

    def __init__(self, gateway: MepGateway, provider: LLMProvider,
                 actor: Element, session: MepSession,
                 demo: bool = False, aid: str = "",
                 inject_memory: bool = False,
                 inject_sit_tone: bool = True):
        self.gw           = gateway
        self.provider     = provider
        self.actor        = actor
        self.session      = session
        self.demo         = demo
        self.aid          = aid or str(uuid.uuid4())[:6]
        self.inject_memory = inject_memory
        self.inject_sit_tone = inject_sit_tone

    async def decide(self, goal: str, ctx: Context,
                      sense: SenseVector = None) -> Optional[Dict]:
        if self.demo: return self._demo(goal, ctx)
        if not self.session.budget_ok:
            self.demo = True; return self._demo(goal, ctx)

        envelope = self.gw.build_envelope(
            self.session, self.actor, ctx,
            inject_memory=self.inject_memory, current_sense=sense)

        last_err: Optional[str] = None
        for attempt in range(1, 4):
            prompt = envelope.to_prompt(self.inject_memory, self.inject_sit_tone)
            if last_err: prompt += f"\n\nPREVIOUS ERROR: {last_err}{RETRY_SUFFIX}"
            user = f"GOAL: {goal}\n\n{prompt}\n\nDecide now."

            try:
                raw = await self.provider.chat(SYSTEM_PROMPT, user,
                                                temp=0.05*attempt)
            except LlmCircuitOpen:
                self.demo=True; return self._demo(goal, ctx)
            except LlmError as e:
                self.session.consume_budget(); return None

            d, pe = MepParser.parse(raw, envelope.valid_types)
            if d: return d
            last_err = pe

        self.session.consume_budget(); return None

    async def act(self, goal: str, ctx: Context,
                   extra: Optional[Dict] = None,
                   sense: SenseVector = None) -> Optional[Reaction]:
        d = await self.decide(goal, ctx, sense)
        if not d: return None
        if d.get("decision") == "skip": return None

        atype   = d["action_type"]
        payload = {**d.get("payload",{}), **(extra or {}), "_goal": goal}

        # Show harmony score for this action
        if ctx:
            frame = {"actor_id": self.actor.element_id}
            for entry in ctx._actions:
                if entry["a"].type == atype:
                    h = entry["a"].harmony_in(ctx, sense)
                    print(f"  [AGENT/{self.aid}] {atype}  H={h.score:+.3f}"
                          f"  (ctx={h.context_alignment:.2f} sem={h.semantic_alignment:.2f})")
                    break
            print(f"  [REASON]  {d.get('reasoning','')[:80]}")

        r = await self.gw.dispatch(self.session, self.actor, atype, payload, ctx)
        icon = "✓" if r.is_success else "✗"
        print(f"  [RXN]     {icon} {r.line()}")
        return r

    def _demo(self, goal: str, ctx: Context) -> Optional[Dict]:
        env = self.gw.build_envelope(self.session, self.actor, ctx)
        if not env.actions:
            return {"decision":"skip","action_type":"","payload":{},"reasoning":"No actions"}
        gl = goal.lower()
        kws = [
            ("create school","school.create",{"name":"Demo School","code":"DS1"}),
            ("create course","course.create",{}),
            ("open enrollment","enrollment.open",{}),
            ("close enrollment","enrollment.close",{}),
            ("enroll","student.enroll",{}),
            ("withdraw","student.withdraw",{}),
            ("grade","grade.record",{}),
            ("snapshot","system.snapshot",{}),
            ("query grade","student.query-grades",{}),
        ]
        for kw, at, pl in kws:
            if kw in gl and at in env.valid_types:
                return {"decision":"execute","action_type":at,"payload":pl,
                        "reasoning":f"Keyword '{kw}'"}
        best = env.actions[0]
        return {"decision":"execute","action_type":best["type"],"payload":{},
                "reasoning":"Highest harmony action"}
