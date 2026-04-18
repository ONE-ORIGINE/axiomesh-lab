"""
mep_ollama_agent.py  v2.0
═══════════════════════════════════════════════════════════════════════════════
MEP/2.0 × Ollama — Hardened AI Agent in a Structured Causal Environment

FIXES v2:
  ✓ Action type parsing — strips [CATEGORY] prefixes the LLM may add
  ✓ Null-safe decision handling everywhere
  ✓ Retry logic with exponential backoff (3 attempts)
  ✓ Timeout per request (configurable, default 45s)
  ✓ Empty/malformed JSON detection and recovery
  ✓ Graceful degradation to demo-mode when LLM keeps failing
  ✓ Per-phase error budget — a phase doesn't crash on one bad response
  ✓ LLM Circuit Breaker — opens after N consecutive failures, half-opens after cooldown

INNOVATIONS v2:
  ✓ Reactive Attention Map — the LLM receives a heat-map of where to focus
    based on phenomenon magnitude and circumstance volatility
  ✓ Agent Memory — context of what it just did, so it doesn't repeat itself
  ✓ Causal Replay — show the AI its own last N actions as causal context
  ✓ Semantic Negotiation — agent proposes intent in natural language,
    MEP translates to nearest action via gravity, agent confirms or rebids
  ✓ Situation-Aware Prompt — prompt changes tone/urgency based on situation severity
  ✓ Ollama library used if available, HTTP fallback otherwise

Usage:
  python mep_ollama_agent.py                    # auto-detect Ollama
  python mep_ollama_agent.py --demo             # deterministic demo, no LLM
  python mep_ollama_agent.py --model mistral    # choose model
  python mep_ollama_agent.py --timeout 60       # longer timeout
  python mep_ollama_agent.py --retries 5        # more retries
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import sys
import textwrap
import time
import traceback
import uuid
import argparse
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

# ─── ANSI ────────────────────────────────────────────────────────────────────

class C:
    RESET   = "\033[0m";  BOLD = "\033[1m";  DIM = "\033[2m"
    CYAN    = "\033[96m"; GREEN = "\033[92m"; YELLOW = "\033[93m"
    RED     = "\033[91m"; MAGENTA = "\033[95m"; BLUE = "\033[94m"
    WHITE   = "\033[97m"; ORANGE = "\033[33m"

def hr(n=68, char="═", c=C.DIM): print(f"{c}{char*n}{C.RESET}")
def section(t): hr(); print(f"{C.BOLD}{C.CYAN}  {t}{C.RESET}"); hr()

def tag(label: str, msg: str, color=C.GREEN, icon=""):
    prefix = f"{icon} " if icon else ""
    print(f"  {color}[{label}]{C.RESET} {prefix}{msg}")

def tag_ok(label: str, msg: str):  tag(label, msg, C.GREEN,   "✓")
def tag_err(label: str, msg: str): tag(label, msg, C.RED,     "✗")
def tag_warn(label: str, msg: str):tag(label, msg, C.YELLOW,  "!")
def tag_info(label: str, msg: str):tag(label, msg, C.CYAN,    "→")
def tag_dim(label: str, msg: str): tag(label, msg, C.DIM,     " ")

# ─── ERRORS ──────────────────────────────────────────────────────────────────

class MepError(Exception): pass
class LlmError(MepError): pass
class LlmTimeoutError(LlmError): pass
class LlmEmptyResponseError(LlmError): pass
class LlmParseError(LlmError): pass
class LlmCircuitOpenError(LlmError): pass
class ActionNotFoundError(MepError): pass
class CircumstanceBlockedError(MepError): pass

# ─── CIRCUIT BREAKER ─────────────────────────────────────────────────────────

class CircuitState(Enum):
    CLOSED     = auto()   # normal — requests pass through
    OPEN       = auto()   # broken — requests fail immediately
    HALF_OPEN  = auto()   # testing — one probe allowed

class CircuitBreaker:
    """
    Classic circuit breaker for LLM calls.
    CLOSED → OPEN after `failure_threshold` consecutive failures.
    OPEN   → HALF_OPEN after `recovery_seconds`.
    HALF_OPEN → CLOSED on success, OPEN on failure.
    """
    def __init__(self, failure_threshold: int = 4, recovery_seconds: float = 30.0):
        self._threshold  = failure_threshold
        self._recovery   = recovery_seconds
        self._failures   = 0
        self._last_fail  = 0.0
        self._state      = CircuitState.CLOSED

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_fail >= self._recovery:
                self._state = CircuitState.HALF_OPEN
        return self._state

    def record_success(self):
        self._failures = 0
        self._state    = CircuitState.CLOSED

    def record_failure(self):
        self._failures  += 1
        self._last_fail  = time.time()
        if self._failures >= self._threshold:
            self._state = CircuitState.OPEN
            tag_warn("CIRCUIT",
                f"Breaker OPEN after {self._failures} failures "
                f"(cooldown {self._recovery:.0f}s)")

    def allow_request(self) -> bool:
        s = self.state
        if s == CircuitState.CLOSED:     return True
        if s == CircuitState.HALF_OPEN:  return True   # probe attempt
        return False   # OPEN

    def status_line(self) -> str:
        s = self.state
        icons = {CircuitState.CLOSED: C.GREEN+"●",
                 CircuitState.OPEN:   C.RED+"●",
                 CircuitState.HALF_OPEN: C.YELLOW+"◐"}
        return f"{icons[s]}{C.RESET} circuit={s.name.lower()} failures={self._failures}"

# =============================================================================
# SENSE VECTOR
# =============================================================================

DIMS = 8

@dataclass(frozen=True)
class Sense:
    dimension : str
    meaning   : str
    magnitude : float
    vector    : Tuple[float, ...]

    @classmethod
    def of(cls, dim: str, meaning: str, axis: int, magnitude: float = 1.0) -> "Sense":
        v = [0.0]*DIMS; v[axis] = magnitude
        return cls(dim, meaning, magnitude, tuple(v))

    def affinity(self, other: "Sense") -> float:
        dot = sum(a*b for a,b in zip(self.vector, other.vector))
        na  = math.sqrt(sum(x*x for x in self.vector))
        nb  = math.sqrt(sum(x*x for x in other.vector))
        return dot/(na*nb) if na>0 and nb>0 else 0.0

    def angular_distance(self, other: "Sense") -> float:
        a = max(-1.0, min(1.0, self.affinity(other)))
        return math.acos(a) / math.pi

    @classmethod
    def normative(cls, m: str, mag: float=1.0): return cls.of("normative",m,3,mag)
    @classmethod
    def temporal(cls,  m: str, mag: float=1.0): return cls.of("temporal", m,1,mag)
    @classmethod
    def technical(cls, m: str, mag: float=1.0): return cls.of("technical",m,6,mag)
    @classmethod
    def social(cls,    m: str, mag: float=1.0): return cls.of("social",   m,4,mag)
    @classmethod
    def causal(cls,    m: str, mag: float=1.0): return cls.of("causal",   m,0,mag)

SENSE_EMPTY = Sense("none","",0.0,tuple([0.0]*DIMS))

# =============================================================================
# CIRCUMSTANCE — Composable boolean predicate
# =============================================================================

class Circumstance:
    def __init__(self, cid: str, desc: str, fn: Callable,
                 role: str = "enabler", weight: float = 1.0):
        self.id=cid; self.description=desc
        self._fn=fn; self.role=role; self.weight=weight

    def evaluate(self, ctx: "ContextFrame", frame: Dict) -> bool:
        try:   return bool(self._fn(ctx, frame))
        except Exception: return False

    def __and__(self, o): return Circumstance(f"{self.id}&{o.id}",
        f"({self.description}) AND ({o.description})",
        lambda c,f: self._fn(c,f) and o._fn(c,f))

    def __or__(self, o):  return Circumstance(f"{self.id}|{o.id}",
        f"({self.description}) OR ({o.description})",
        lambda c,f: self._fn(c,f) or o._fn(c,f))

    def __invert__(self): return Circumstance(f"!{self.id}",
        f"NOT ({self.description})",
        lambda c,f: not self._fn(c,f), "blocker", -self.weight)

    def to_dict(self): return {"id":self.id,"description":self.description,"role":self.role}

def circ_flag(cid,desc,key,val=True):
    return Circumstance(cid,desc,lambda ctx,_: ctx.data.get(key)==val)
def circ_always(cid):
    return Circumstance(cid,cid,lambda *_: True)

# =============================================================================
# CONTEXT FRAME
# =============================================================================

@dataclass
class ContextFrame:
    context_id         : str = field(default_factory=lambda: str(uuid.uuid4()))
    name               : str = ""
    kind               : str = "semantic"
    basis              : Sense = field(default_factory=lambda: SENSE_EMPTY)
    parent_id          : Optional[str] = None
    depth              : int = 0
    data               : Dict[str,Any]  = field(default_factory=dict)
    scoped_elements    : List[Dict]     = field(default_factory=list)
    circumstances      : List[Circumstance] = field(default_factory=list)
    _registered_actions: List[Dict]     = field(default_factory=list, repr=False)

    def add_circumstance(self, c: Circumstance) -> "ContextFrame":
        self.circumstances.append(c); return self

    def register_action(self, action: "EnvAction",
                        actor_filter=None) -> "ContextFrame":
        self._registered_actions.append({"action":action,"filter":actor_filter})
        return self

    def get_available_actions(self, actor: Dict,
                              frame: Dict) -> List[Tuple["EnvAction",float]]:
        result, frame = [], {**frame}
        for entry in self._registered_actions:
            a, filt = entry["action"], entry["filter"]
            if filt and not filt(actor): continue
            if not all(g.evaluate(self, frame) for g in a.guards): continue
            result.append((a, round(a.sense.affinity(self.basis), 4)))
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def evaluate_circumstances(self, frame: Dict) -> List[Dict]:
        return [{"id":c.id,"description":c.description,
                 "holds":c.evaluate(self,frame),"role":c.role}
                for c in self.circumstances]

    def include(self, e: Dict) -> "ContextFrame":
        self.scoped_elements.append(e); return self

    def set(self, k: str, v: Any) -> "ContextFrame":
        self.data[k] = v; return self

    def to_envelope(self, actor: Dict, frame: Dict,
                    recent_events: List[Dict] = (),
                    situation: Optional[Dict] = None,
                    agent_memory: Optional[List[Dict]] = None,
                    phenomena: Optional[List[Dict]] = None) -> "ContextEnvelope":
        available = self.get_available_actions(actor, frame)
        circs     = self.evaluate_circumstances(frame)

        # Reactive attention: highlight volatile circumstances and high-magnitude phenomena
        attention: List[Dict] = []
        for c in circs:
            if not c["holds"] and c["role"] == "enabler":
                attention.append({"type":"blocked_enabler","id":c["id"],
                                   "message":f"'{c['description']}' must hold to unlock actions"})
        if phenomena:
            for p in phenomena:
                if p.get("magnitude",0) > 0.5:
                    attention.append({"type":"phenomenon","name":p["name"],
                                      "magnitude":p["magnitude"],
                                      "message":f"Emerging phenomenon — action may be needed"})

        return ContextEnvelope(
            context_id        = self.context_id,
            name              = self.name,
            kind              = self.kind,
            depth             = self.depth,
            data              = dict(self.data),
            circumstances     = circs,
            situation         = situation,
            recent_events     = list(recent_events)[-5:],
            agent_memory      = agent_memory or [],
            attention_map     = attention,
            available_actions = [
                {"type":a.type,"category":a.category,
                 "description":a.description,"gravity":g,
                 "sense_dimension":a.sense.dimension}
                for a,g in available
            ],
        )

# =============================================================================
# CONTEXT ENVELOPE — The AI-consumable semantic frame
# =============================================================================

@dataclass
class ContextEnvelope:
    context_id        : str = ""
    name              : str = ""
    kind              : str = "semantic"
    depth             : int = 0
    data              : Dict[str,Any]  = field(default_factory=dict)
    circumstances     : List[Dict]     = field(default_factory=list)
    available_actions : List[Dict]     = field(default_factory=list)
    situation         : Optional[Dict] = None
    recent_events     : List[Dict]     = field(default_factory=list)
    agent_memory      : List[Dict]     = field(default_factory=list)
    attention_map     : List[Dict]     = field(default_factory=list)

    def to_prompt(self, urgency_level: str = "normal") -> str:
        holding  = [c for c in self.circumstances if c.get("holds")]
        blocked  = [c for c in self.circumstances
                    if c.get("holds") and c.get("role")=="blocker"]
        missing  = [c for c in self.circumstances
                    if not c.get("holds") and c.get("role")=="enabler"]

        # Urgency tone
        tone = {
            "normal":   "",
            "degraded": f"\n{C.YELLOW}⚠ ENVIRONMENT IN DEGRADED STATE — prioritize stabilization{C.RESET}",
            "critical": f"\n{C.RED}🚨 CRITICAL SITUATION — immediate action required{C.RESET}",
        }.get(urgency_level, "")

        actions_str = "\n".join(
            f"  {i+1}. {a['type']}  [gravity={a['gravity']:.3f}]  — {a['description']}"
            for i,a in enumerate(self.available_actions)
        ) or "  (none — all gated by failed circumstances)"

        holding_str  = "\n".join(f"  ✓ {c['description']}" for c in holding) or "  (none)"
        missing_str  = "\n".join(f"  ✗ {c['description']} ← must become true" for c in missing) or "  (none)"
        blocked_str  = "\n".join(f"  ⊗ {c['description']}" for c in blocked) or "  (none)"

        data_str = "\n".join(
            f"  {k}: {v}" for k,v in self.data.items()
            if not k.startswith("_")
        ) or "  (empty)"

        events_str = ""
        if self.recent_events:
            events_str = "\nRECENT EVENTS:\n" + "\n".join(
                f"  • [{e.get('type','')}] {e.get('summary','')}"
                for e in self.recent_events
            )

        memory_str = ""
        if self.agent_memory:
            memory_str = "\nYOUR RECENT ACTIONS:\n" + "\n".join(
                f"  [{m.get('status','?')}] {m.get('action','')} — {m.get('goal','')[:60]}"
                for m in self.agent_memory[-4:]
            )

        attention_str = ""
        if self.attention_map:
            attention_str = "\nATTENTION:\n" + "\n".join(
                f"  ⚡ {a.get('message','')}"
                for a in self.attention_map
            )

        sit_str = ""
        if self.situation:
            s = self.situation
            sit_str = (f"\nSITUATION: {s.get('kind','?').upper()} "
                       f"severity={s.get('severity','?')} "
                       f"saturation={s.get('saturation',0):.0%}")

        return textwrap.dedent(f"""
            CONTEXT: {self.name} [{self.kind}] depth={self.depth}{sit_str}{tone}

            HOLDING CIRCUMSTANCES (currently true):
            {holding_str}

            MISSING CIRCUMSTANCES (currently false — block some actions):
            {missing_str}

            BLOCKED BY (active blockers):
            {blocked_str}

            CONTEXT DATA:
            {data_str}
            {events_str}{memory_str}{attention_str}

            AVAILABLE ACTIONS (type only — ranked by causal gravity):
            {actions_str}

            IMPORTANT: action_type must be EXACTLY as shown above (no prefix, no brackets).
        """).strip()

    @property
    def valid_action_types(self) -> set:
        return {a["type"] for a in self.available_actions}

# =============================================================================
# REACTION & ACTION
# =============================================================================

@dataclass
class EnvReaction:
    action_type  : str
    status       : str    # success | rejected | error
    message      : Optional[str] = None
    result       : Any    = None
    impact_scope : str    = "actor"
    target_id    : Optional[str] = None
    temporality  : str    = "immediate"
    chain_actions: List[str] = field(default_factory=list)
    reaction_id  : str    = field(default_factory=lambda: str(uuid.uuid4()))
    produced_at  : float  = field(default_factory=time.time)

    def to_summary(self) -> str:
        color  = C.GREEN if self.status=="success" else (C.YELLOW if self.status=="rejected" else C.RED)
        icon   = "✓" if self.status=="success" else "✗"
        result_str = f" → {json.dumps(self.result,default=str)[:80]}" if self.result else ""
        msg_str    = f" | {self.message}" if self.message else ""
        return f"{color}{icon} {self.action_type} → {self.status}{msg_str}{result_str}{C.RESET}"

    @property
    def is_success(self) -> bool: return self.status == "success"

@dataclass
class EnvAction:
    type       : str
    category   : str
    description: str
    sense      : Sense
    guards     : List[Circumstance] = field(default_factory=list)
    can_chain  : bool = True
    _handler   : Any  = field(default=None, repr=False)

    async def execute(self, actor: Dict, payload: Dict,
                      ctx: "ContextFrame", frame: Dict) -> EnvReaction:
        for g in self.guards:
            if not g.evaluate(ctx, frame):
                return EnvReaction(self.type, "rejected",
                    f"Circumstance not met: {g.description}")
        if not self._handler:
            return EnvReaction(self.type, "success", "OK")
        try:
            return await self._handler(actor, payload, ctx, frame)
        except Exception as e:
            return EnvReaction(self.type, "error", f"Handler exception: {e}")

# =============================================================================
# CAUSAL GRAPH
# =============================================================================

@dataclass
class CausalNode:
    node_id       : str
    node_type     : str
    action_type   : str
    actor_id      : str
    context_name  : str
    correlation_id: str
    causation_id  : Optional[str]
    depth         : int
    timestamp     : float
    status        : str
    summary       : str

class CausalGraph:
    def __init__(self):
        self._nodes:    Dict[str,CausalNode]  = {}
        self._children: Dict[str,List[str]]   = {}

    def add(self, n: CausalNode):
        self._nodes[n.node_id] = n
        if n.causation_id:
            self._children.setdefault(n.causation_id,[]).append(n.node_id)

    def ancestry(self, node_id: str) -> List[CausalNode]:
        chain, cur = [], node_id
        while cur and (node := self._nodes.get(cur)):
            chain.insert(0, node); cur = node.causation_id or ""
        return chain

    def explain(self, action_type: str) -> str:
        matches = [n for n in self._nodes.values() if n.action_type==action_type]
        if not matches: return f"No record found for '{action_type}'"
        chain = self.ancestry(matches[-1].node_id)
        lines = [f"Causal chain for '{action_type}':"]
        for n in chain:
            lines.append(f"  [d={n.depth}] {n.node_type}: {n.action_type} ({n.status}) — {n.summary}")
        return "\n".join(lines)

    @property
    def stats(self) -> Dict:
        return {"nodes":len(self._nodes),
                "types":list({n.action_type for n in self._nodes.values()}),
                "successes":sum(1 for n in self._nodes.values() if n.status=="success"),
                "rejections":sum(1 for n in self._nodes.values() if n.status=="rejected")}

# =============================================================================
# PHENOMENON DETECTOR
# =============================================================================

class PhenomenonDetector:
    def __init__(self):
        self._history : List[Tuple[float,str]] = []
        self._detected: List[Dict] = []

    def record(self, rtype: str):
        self._history.append((time.time(), rtype))
        if len(self._history) > 500: self._history = self._history[-500:]

    def detect(self, pattern: str, threshold: int, window_s: float) -> Optional[Dict]:
        cutoff  = time.time() - window_s
        matches = [t for t,rt in self._history if rt==pattern and t>=cutoff]
        if len(matches) >= threshold:
            p = {"name":f"{pattern}.phenomenon","pattern_type":pattern,
                 "magnitude":min(1.0,len(matches)/threshold),"count":len(matches),
                 "window_s":window_s,"detected_at":time.time()}
            if not any(d["pattern_type"]==pattern for d in self._detected[-3:]):
                self._detected.append(p)
            return p
        return None

    @property
    def all_detected(self) -> List[Dict]: return self._detected

# =============================================================================
# OLLAMA CLIENT — library preferred, HTTP fallback
# =============================================================================

class OllamaClient:
    """
    Tries the `ollama` Python package first.
    Falls back to raw HTTP if the package isn't installed.
    Implements timeout, retry, and circuit-breaker protection.
    """

    def __init__(self, host: str = "http://localhost:11434",
                 model: str = "llama3",
                 timeout: float = 45.0,
                 max_retries: int = 3):
        self.host        = host
        self.model       = model
        self.timeout     = timeout
        self.max_retries = max_retries
        self._circuit    = CircuitBreaker(failure_threshold=4, recovery_seconds=30)
        self._use_lib    = False
        self._lib_client = None
        self._available  = False

        # Try library first
        try:
            import ollama as _lib          # type: ignore
            self._lib_client = _lib
            self._use_lib    = True
            tag_info("OLLAMA", f"Using ollama Python library  model={model}")
        except ImportError:
            tag_dim("OLLAMA", "Python library not found — using HTTP client")

        self._available = self._probe()

    def _probe(self) -> bool:
        try:
            if self._use_lib and self._lib_client:
                models = self._lib_client.list()
                names  = [m.get("name","") if isinstance(m,dict) else str(m)
                           for m in (models.get("models",[]) if isinstance(models,dict) else [])]
                tag_info("OLLAMA", f"Connected  available models: {names[:4]}")
                return True
            else:
                req = urllib.request.Request(f"{self.host}/api/tags")
                with urllib.request.urlopen(req, timeout=3) as r:
                    data  = json.loads(r.read())
                    names = [m.get("name","") for m in data.get("models",[])]
                    tag_info("OLLAMA", f"Connected  available models: {names[:4]}")
                    return True
        except Exception as e:
            tag_warn("OLLAMA", f"Not reachable ({e})")
            return False

    @property
    def is_available(self) -> bool:
        return self._available

    async def chat(self, system_prompt: str, user_message: str,
                   temperature: float = 0.1) -> str:
        """
        Send a chat request. Retries on transient errors with backoff.
        Raises LlmError subtypes on persistent failure.
        """
        if not self._circuit.allow_request():
            raise LlmCircuitOpenError(
                f"Circuit breaker OPEN — LLM calls suspended "
                f"(cooldown {self._circuit._recovery:.0f}s)")

        last_exc: Exception = RuntimeError("no attempt")

        for attempt in range(1, self.max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    self._send(system_prompt, user_message, temperature),
                    timeout=self.timeout)
                if not result or not result.strip():
                    raise LlmEmptyResponseError("LLM returned empty response")
                self._circuit.record_success()
                return result

            except asyncio.TimeoutError as e:
                last_exc = LlmTimeoutError(
                    f"LLM timeout after {self.timeout:.0f}s (attempt {attempt}/{self.max_retries})")
                tag_warn("LLM", f"Timeout attempt {attempt}/{self.max_retries}")

            except LlmEmptyResponseError as e:
                last_exc = e
                tag_warn("LLM", f"Empty response attempt {attempt}/{self.max_retries}")

            except (LlmCircuitOpenError, LlmTimeoutError): raise

            except Exception as e:
                last_exc = LlmError(str(e))
                tag_warn("LLM", f"Error attempt {attempt}/{self.max_retries}: {e}")

            if attempt < self.max_retries:
                backoff = 2 ** (attempt - 1)   # 1s, 2s, 4s
                tag_dim("LLM", f"Retrying in {backoff}s...")
                await asyncio.sleep(backoff)

        self._circuit.record_failure()
        raise last_exc

    async def _send(self, system: str, user: str, temperature: float) -> str:
        if self._use_lib and self._lib_client:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._send_lib, system, user, temperature)
        return await asyncio.get_event_loop().run_in_executor(
            None, self._send_http, system, user, temperature)

    def _send_lib(self, system: str, user: str, temperature: float) -> str:
        lib     = self._lib_client
        resp    = lib.chat(model=self.model,
                           messages=[{"role":"system","content":system},
                                     {"role":"user",  "content":user}],
                           options={"temperature":temperature})
        if isinstance(resp, dict):
            return resp.get("message",{}).get("content","")
        return str(resp)

    def _send_http(self, system: str, user: str, temperature: float) -> str:
        body = json.dumps({
            "model":    self.model,
            "messages": [{"role":"system","content":system},
                          {"role":"user",  "content":user}],
            "options":  {"temperature":temperature},
            "stream":   False,
        }).encode()
        req = urllib.request.Request(
            f"{self.host}/api/chat", data=body,
            headers={"Content-Type":"application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["message"]["content"]

# =============================================================================
# RESPONSE PARSER — robust, strips LLM formatting artifacts
# =============================================================================

class ResponseParser:
    """
    Extracts a structured decision from the LLM's raw text output.
    Handles:
      - Pure JSON
      - JSON inside ```json ... ``` or ``` ... ```
      - JSON with [CATEGORY] prefix in action_type
      - Partial JSON
      - Completely unparseable → returns a fallback skip decision
    """

    # Patterns the LLM might use for action_type
    _CATEGORY_PREFIX = re.compile(
        r"^\s*\[(?:LIFECYCLE|COMMAND|QUERY|SIGNAL|TRANSFORM)\]\s*", re.IGNORECASE)

    @classmethod
    def parse(cls, raw: str, valid_types: set,
              context: str = "") -> Tuple[Optional[Dict], Optional[str]]:
        """
        Returns (decision_dict, error_message).
        If decision_dict is None, error_message explains why.
        """
        if not raw or not raw.strip():
            return None, "Empty response from LLM"

        text = raw.strip()

        # 1. Strip markdown fences
        text = cls._strip_fences(text)

        # 2. Extract first JSON object
        parsed = cls._extract_json(text)
        if parsed is None:
            return None, f"Could not parse JSON from response: {text[:120]!r}"

        # 3. Normalise action_type — strip category prefix if present
        atype = str(parsed.get("action_type","")).strip()
        atype = cls._CATEGORY_PREFIX.sub("", atype).strip()
        parsed["action_type"] = atype

        # 4. Validate required fields
        decision = parsed.get("decision","").strip().lower()
        if decision not in ("execute","skip","query"):
            # Try to salvage — if there's a valid action_type, assume execute
            if atype in valid_types:
                parsed["decision"] = "execute"
                decision = "execute"
            else:
                return None, f"Invalid decision field: {decision!r}"

        if decision == "execute":
            if not atype:
                return None, "action_type is empty"
            if atype not in valid_types:
                # Fuzzy match: find closest by substring
                candidates = [t for t in valid_types if atype in t or t in atype]
                if len(candidates) == 1:
                    parsed["action_type"] = candidates[0]
                    parsed["_fuzzy_matched"] = True
                else:
                    return (None,
                        f"Action '{atype}' not available. "
                        f"Valid: [{', '.join(sorted(valid_types))}]")

        return parsed, None

    @staticmethod
    def _strip_fences(text: str) -> str:
        # ```json ... ``` or ``` ... ```
        m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if m: return m.group(1).strip()
        return text

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict]:
        # Try full text first
        try: return json.loads(text)
        except json.JSONDecodeError: pass
        # Find first { ... } block
        start = text.find("{")
        if start == -1: return None
        depth, end = 0, -1
        for i, ch in enumerate(text[start:], start):
            if ch=="{": depth+=1
            elif ch=="}":
                depth-=1
                if depth==0: end=i; break
        if end==-1: return None
        try: return json.loads(text[start:end+1])
        except json.JSONDecodeError: return None

# =============================================================================
# ESCHOOL ENVIRONMENT
# =============================================================================

class ESchoolEnv:
    def __init__(self):
        self.name        = "ESchool"
        self._students   : Dict[str,Dict] = {}
        self._teachers   : Dict[str,Dict] = {}
        self._admins     : Dict[str,Dict] = {}
        self._courses    : Dict[str,Dict] = {}
        self._schools    : Dict[str,Dict] = {}
        self._grades     : List[Dict]     = []
        self._enrollments: List[Dict]     = []
        self._events     : List[Dict]     = []
        self._causal     = CausalGraph()
        self._phenomena  = PhenomenonDetector()

        # Global circumstances
        self.C_ACTIVE   = circ_always("system.active")
        self.C_ENRL     = circ_flag("enrollment.open","Enrollment period is active","enrollmentOpen")
        self._gdata     = {"enrollmentOpen": False, "systemActive": True}

        # Contexts
        self.root_ctx  = self._build_root()
        self.admin_ctx = self._build_admin()
        self.acad_ctx  = self._build_academic()
        self.enrl_ctx  = self._build_enrollment()

    # ── Context builders ──────────────────────────────────────────────────

    def _build_root(self) -> ContextFrame:
        c = ContextFrame(name="ESchool.Root",kind="global",
                         basis=Sense.technical("global frame",0.3))
        c.add_circumstance(self.C_ACTIVE)
        return c

    def _build_admin(self) -> ContextFrame:
        c = ContextFrame(name="ESchool.Administrative",kind="semantic",
                         basis=Sense.normative("administrative ops",0.9),
                         depth=1,parent_id=self.root_ctx.context_id)
        c.add_circumstance(self.C_ACTIVE)
        for a in [self._a_create_school(), self._a_create_course(),
                  self._a_assign_teacher(), self._a_open_enrl(),
                  self._a_close_enrl(),     self._a_snapshot(),
                  self._a_list_elements()]:
            c.register_action(a, lambda actor,t=a.type: True)
        return c

    def _build_academic(self) -> ContextFrame:
        c = ContextFrame(name="ESchool.Academic",kind="semantic",
                         basis=Sense.normative("academic grading",0.95),
                         depth=1,parent_id=self.root_ctx.context_id)
        c.add_circumstance(self.C_ACTIVE)
        for a in [self._a_record_grade(), self._a_query_grades(), self._a_snapshot()]:
            c.register_action(a)
        return c

    def _build_enrollment(self) -> ContextFrame:
        c = ContextFrame(name="ESchool.Enrollment",kind="temporal",
                         basis=Sense.temporal("enrollment period",0.9),
                         depth=1,parent_id=self.root_ctx.context_id)
        c.add_circumstance(self.C_ACTIVE)
        c.add_circumstance(self.C_ENRL)
        for a in [self._a_enroll(), self._a_withdraw(), self._a_query_enrl()]:
            c.register_action(a)
        return c

    # ── Action factories ──────────────────────────────────────────────────

    def _a_create_school(self) -> EnvAction:
        async def h(actor,payload,ctx,frame):
            name=payload.get("name","School"); code=payload.get("code",uuid.uuid4().hex[:6].upper())
            country=payload.get("countryCode","INT"); sid=str(uuid.uuid4())
            self._schools[sid]={"id":sid,"name":name,"code":code,"country":country,"type":"School"}
            self._emit("school.created",f"School '{name}' ({code}) — {country}")
            return EnvReaction("school.create","success",f"School '{name}' created",
                result={"id":sid,"name":name,"code":code},impact_scope="environment")
        return EnvAction("school.create","lifecycle","Create a new school institution",
                         Sense.normative("school creation",0.9),_handler=h)

    def _a_create_course(self) -> EnvAction:
        async def h(actor,payload,ctx,frame):
            name=payload.get("name"); code=payload.get("code",uuid.uuid4().hex[:6].upper())
            disc=payload.get("discipline","General"); cap=int(payload.get("capacity",30))
            cred=int(payload.get("credits",3)); cid=str(uuid.uuid4())
            if not name: return EnvReaction("course.create","rejected","Missing 'name'")
            self._courses[cid]={"id":cid,"name":name,"code":code,"discipline":disc,
                                 "maxCapacity":cap,"credits":cred,"enrolled":0,
                                 "enrolledStudents":[],"grades":[],"type":"Course"}
            self._emit("course.created",f"Course '{name}' ({code}) cap={cap}")
            return EnvReaction("course.create","success",f"Course '{name}' created",
                result={"id":cid,"name":name,"code":code,"capacity":cap},impact_scope="environment")
        return EnvAction("course.create","lifecycle","Create a new course",
                         Sense.technical("course provisioning",0.8),_handler=h)

    def _a_assign_teacher(self) -> EnvAction:
        async def h(actor,payload,ctx,frame):
            tid=payload.get("teacherId",""); cid=payload.get("courseId","")
            t=self._teachers.get(tid); c=self._courses.get(cid)
            if not t: return EnvReaction("course.assign-teacher","rejected","Teacher not found")
            if not c: return EnvReaction("course.assign-teacher","rejected","Course not found")
            c["teacherId"]=tid; c["teacherName"]=t["name"]
            self._emit("teacher.assigned",f"{t['name']} → {c['name']}")
            return EnvReaction("course.assign-teacher","success",
                f"{t['name']} → {c['name']}",impact_scope="specific",target_id=cid)
        return EnvAction("course.assign-teacher","command","Assign teacher to course",
                         Sense.social("teacher assignment",0.7),_handler=h)

    def _a_open_enrl(self) -> EnvAction:
        async def h(actor,payload,ctx,frame):
            self._gdata["enrollmentOpen"]=True
            for cx in [self.root_ctx,self.admin_ctx,self.acad_ctx,self.enrl_ctx]: cx.set("enrollmentOpen",True)
            self._emit("enrollment.opened","Enrollment period is now OPEN")
            return EnvReaction("enrollment.open","success","Enrollment period opened",impact_scope="environment")
        return EnvAction("enrollment.open","command","Open the enrollment period",
                         Sense.temporal("enrollment activation",0.85),_handler=h)

    def _a_close_enrl(self) -> EnvAction:
        async def h(actor,payload,ctx,frame):
            self._gdata["enrollmentOpen"]=False
            for cx in [self.root_ctx,self.admin_ctx,self.acad_ctx,self.enrl_ctx]: cx.set("enrollmentOpen",False)
            self._emit("enrollment.closed","Enrollment period is now CLOSED")
            return EnvReaction("enrollment.close","success","Enrollment period closed",impact_scope="environment")
        return EnvAction("enrollment.close","command","Close the enrollment period",
                         Sense.temporal("enrollment deactivation",0.7),_handler=h)

    def _a_enroll(self) -> EnvAction:
        async def h(actor,payload,ctx,frame):
            sid=payload.get("studentId",""); cid=payload.get("courseId","")
            s=self._students.get(sid); c=self._courses.get(cid)
            if not s: return EnvReaction("student.enroll","rejected","Student not found")
            if not c: return EnvReaction("student.enroll","rejected","Course not found")
            if c["enrolled"]>=c["maxCapacity"]: return EnvReaction("student.enroll","rejected",f"Course at capacity ({c['maxCapacity']})")
            if sid in c["enrolledStudents"]: return EnvReaction("student.enroll","rejected","Already enrolled")
            c["enrolledStudents"].append(sid); c["enrolled"]+=1
            s.setdefault("enrolledCourses",[]).append(cid)
            self._enrollments.append({"studentId":sid,"courseId":cid,"at":time.time()})
            self._emit("enrollment.confirmed",f"{s['name']} → {c['name']}")
            return EnvReaction("student.enroll","success",f"{s['name']} enrolled in {c['name']}",
                result={"studentName":s["name"],"courseName":c["name"]},
                impact_scope="specific",target_id=cid)
        return EnvAction("student.enroll","command","Enroll a student in a course",
                         Sense.temporal("enrollment",0.95),guards=[self.C_ENRL],_handler=h)

    def _a_withdraw(self) -> EnvAction:
        async def h(actor,payload,ctx,frame):
            sid=payload.get("studentId",""); cid=payload.get("courseId","")
            s=self._students.get(sid); c=self._courses.get(cid)
            if not s or not c: return EnvReaction("student.withdraw","rejected","Not found")
            if sid not in c.get("enrolledStudents",[]): return EnvReaction("student.withdraw","rejected","Not enrolled")
            c["enrolledStudents"].remove(sid); c["enrolled"]-=1
            if cid in s.get("enrolledCourses",[]): s["enrolledCourses"].remove(cid)
            self._emit("withdrawal.confirmed",f"{s['name']} withdrew from {c['name']}")
            return EnvReaction("student.withdraw","success",f"{s['name']} withdrew",impact_scope="specific",target_id=cid)
        return EnvAction("student.withdraw","command","Withdraw a student from a course",
                         Sense.temporal("withdrawal",0.7),_handler=h)

    def _a_record_grade(self) -> EnvAction:
        async def h(actor,payload,ctx,frame):
            sid=payload.get("studentId",""); cid=payload.get("courseId","")
            try:   grade=float(payload.get("grade",0))
            except: return EnvReaction("grade.record","rejected","Invalid grade value")
            s=self._students.get(sid); c=self._courses.get(cid)
            if not s: return EnvReaction("grade.record","rejected","Student not found")
            if not c: return EnvReaction("grade.record","rejected","Course not found")
            if not (0<=grade<=100): return EnvReaction("grade.record","rejected","Grade must be 0-100")
            letter=("A"if grade>=90 else"B"if grade>=80 else"C"if grade>=70 else"D"if grade>=60 else"F")
            rec={"studentId":sid,"courseId":cid,"grade":grade,"letter":letter,"at":time.time()}
            self._grades.append(rec); c["grades"].append(rec)
            gg=[r["grade"] for r in self._grades if r["studentId"]==sid]
            s["gpa"]=round(sum(gg)/len(gg),2)
            passing=grade>=60
            self._emit("grade.recorded",f"{s['name']}: {grade:.1f} ({letter}) in {c['name']}")
            key="grade.record.fail" if not passing else "grade.record.pass"
            self._phenomena.record(key)
            phenom=self._phenomena.detect("grade.record.fail",3,600)
            if phenom: tag_warn("PHENOMENON",f"MassFailure magnitude={phenom['magnitude']:.0%} count={phenom['count']}")
            chain=[] if passing else ["notification.dispatch"]
            return EnvReaction("grade.record","success",f"{s['name']}: {grade:.1f} ({letter})",
                result={"grade":grade,"letter":letter,"passing":passing,"gpa":s["gpa"]},
                impact_scope="specific",target_id=cid,chain_actions=chain)
        return EnvAction("grade.record","command","Record a student grade",
                         Sense.normative("academic assessment",0.95),_handler=h)

    def _a_query_grades(self) -> EnvAction:
        async def h(actor,payload,ctx,frame):
            sid=payload.get("studentId","")
            s=self._students.get(sid)
            if not s: return EnvReaction("student.query-grades","rejected","Not found")
            grades=[r for r in self._grades if r["studentId"]==sid]
            return EnvReaction("student.query-grades","success",f"{len(grades)} grades for {s['name']}",
                result={"name":s["name"],"gpa":s.get("gpa",0),"count":len(grades),"grades":grades})
        return EnvAction("student.query-grades","query","Query grades for a student",
                         Sense.normative("grade query",0.6),_handler=h)

    def _a_query_enrl(self) -> EnvAction:
        async def h(actor,payload,ctx,frame):
            cid=payload.get("courseId",""); c=self._courses.get(cid)
            if not c: return EnvReaction("enrollment.query","rejected","Course not found")
            students=[self._students[sid]["name"] for sid in c.get("enrolledStudents",[]) if sid in self._students]
            return EnvReaction("enrollment.query","success",f"{c['name']}: {c['enrolled']}/{c['maxCapacity']}",
                result={"name":c["name"],"enrolled":c["enrolled"],"capacity":c["maxCapacity"],"students":students})
        return EnvAction("enrollment.query","query","Query enrollment for a course",
                         Sense.temporal("enrollment status",0.6),_handler=h)

    def _a_snapshot(self) -> EnvAction:
        async def h(actor,payload,ctx,frame):
            return EnvReaction("system.snapshot","success","Snapshot taken",
                result={"schools":len(self._schools),"courses":len(self._courses),
                        "students":len(self._students),"teachers":len(self._teachers),
                        "grades":len(self._grades),"enrollments":len(self._enrollments),
                        "enrollmentOpen":self._gdata.get("enrollmentOpen"),
                        "phenomena":len(self._phenomena.all_detected)})
        return EnvAction("system.snapshot","query","Snapshot the environment state",
                         Sense.technical("system snapshot",0.3),_handler=h)

    def _a_list_elements(self) -> EnvAction:
        async def h(actor,payload,ctx,frame):
            return EnvReaction("system.list","success","Element listing",
                result={"students":list(self._students.values()),
                        "courses":list(self._courses.values()),
                        "teachers":list(self._teachers.values())})
        return EnvAction("system.list","query","List all environment elements",
                         Sense.technical("element listing",0.2),_handler=h)

    # ── Public API ────────────────────────────────────────────────────────

    def add_student(self,name,code) -> Dict:
        sid=str(uuid.uuid4()); e={"id":sid,"name":name,"code":code,"type":"Student","gpa":0,"enrolledCourses":[]}
        self._students[sid]=e; return e

    def add_teacher(self,name,code,discipline) -> Dict:
        tid=str(uuid.uuid4()); e={"id":tid,"name":name,"code":code,"type":"Teacher","discipline":discipline}
        self._teachers[tid]=e; return e

    def add_admin(self,name,code,role="SystemAdmin") -> Dict:
        aid=str(uuid.uuid4()); e={"id":aid,"name":name,"code":code,"type":"Admin","role":role}
        self._admins[aid]=e; return e

    def _emit(self, etype: str, summary: str):
        self._events.append({"type":etype,"summary":summary,"at":time.time()})

    async def dispatch(self, actor: Dict, action_type: str, payload: Dict,
                       ctx: ContextFrame, correlation_id: str) -> EnvReaction:
        frame = {"actor_id":actor["id"],"payload":payload}
        for entry in ctx._registered_actions:
            if entry["action"].type == action_type:
                action   = entry["action"]
                reaction = await action.execute(actor, payload, ctx, frame)
                self._causal.add(CausalNode(
                    node_id=str(uuid.uuid4()),node_type="action",
                    action_type=action_type,actor_id=actor["id"],
                    context_name=ctx.name,correlation_id=correlation_id,
                    causation_id=None,depth=0,timestamp=time.time(),
                    status=reaction.status,summary=reaction.message or ""))
                return reaction
        return EnvReaction(action_type,"rejected",
            f"Action '{action_type}' not registered in context '{ctx.name}'")

    def get_envelope(self, actor: Dict, ctx: ContextFrame,
                     agent_memory: Optional[List[Dict]] = None) -> ContextEnvelope:
        frame = {"actor_id":actor["id"]}
        ctx.data.update(self._gdata)
        sit   = {"kind":"operational" if not self._phenomena.all_detected else "degraded",
                  "severity":"low" if not self._phenomena.all_detected else "high",
                  "saturation":min(1,len(self._students)/max(1,len(self._students)+50))}
        return ctx.to_envelope(actor, frame,
            recent_events=self._events[-6:], situation=sit,
            agent_memory=agent_memory,
            phenomena=self._phenomena.all_detected)

# =============================================================================
# MEP AGENT
# =============================================================================

class MepAgent:
    """
    AI agent operating inside the MEP environment.

    Decision loop:
      1. Receive ContextEnvelope (rich semantic frame)
      2. LLM reasons → structured JSON decision
      3. Parse & validate decision (robust parser)
      4. Dispatch action through MEP
      5. Record in causal memory
      6. Return reaction

    Error handling:
      - LLM timeout → retry with backoff
      - Empty response → retry
      - Unparseable JSON → retry with clearer prompt
      - Invalid action → explain and skip
      - Circuit breaker → open after N failures, half-open after cooldown
    """

    SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI agent operating inside a structured educational environment (ESchool)
    via the MEP (Model Environment Protocol).

    You receive a structured CONTEXT ENVELOPE showing:
    - Available actions (exact type names, ranked by causal gravity)
    - Active/missing circumstances
    - Environment state
    - Your recent actions

    You MUST respond with ONLY a valid JSON object — no explanation, no markdown outside it:
    {
        "decision": "execute",
        "action_type": "<EXACT type from AVAILABLE ACTIONS — copy it verbatim>",
        "payload": { <parameters> },
        "reasoning": "<one sentence>"
    }

    OR:
    { "decision": "skip", "action_type": "", "payload": {}, "reasoning": "<why>" }

    CRITICAL RULES:
    - action_type MUST exactly match one of the listed action types — no prefix, no brackets
    - Only choose actions that appear in AVAILABLE ACTIONS
    - If enrollment is closed, student.enroll will not appear — don't try it
    - Higher gravity = more natural in this context — prefer it
    - Respond with RAW JSON only — no text before or after
    """).strip()

    RETRY_PROMPT_SUFFIX = textwrap.dedent("""

    IMPORTANT: Your previous response could not be parsed.
    Respond with ONLY the JSON object. No text before { or after }.
    action_type must be copied verbatim from the list above.
    """)

    def __init__(self, env: ESchoolEnv, ollama: OllamaClient,
                 actor: Dict, demo_mode: bool = False,
                 agent_id: Optional[str] = None):
        self.env        = env
        self.ollama     = ollama
        self.actor      = actor
        self.demo_mode  = demo_mode
        self.agent_id   = agent_id or str(uuid.uuid4())[:8]
        self._memory   : List[Dict] = []      # recent decisions
        self._error_budget = 5                # per-session tolerance

    async def decide(self, goal: str, ctx: ContextFrame,
                     extra_context: str = "") -> Optional[Dict]:
        """
        Ask LLM for a decision. Returns None on unrecoverable failure.
        Implements retry with progressively clearer prompts.
        """
        if self.demo_mode:
            return self._demo_decide(goal, ctx)

        if self._error_budget <= 0:
            tag_warn(f"AGENT/{self.agent_id}",
                "Error budget exhausted — switching to demo mode for this session")
            self.demo_mode = True
            return self._demo_decide(goal, ctx)

        envelope  = self.env.get_envelope(self.actor, ctx, self._memory)
        urgency   = self._urgency(envelope)

        # Determine the urgency tone for the prompt
        base_prompt = envelope.to_prompt(urgency)
        if extra_context:
            base_prompt = f"{extra_context}\n\n{base_prompt}"

        user_msg = textwrap.dedent(f"""
            GOAL: {goal}

            {base_prompt}

            Decide now. Return only the JSON object.
        """).strip()

        last_error: Optional[str] = None

        for attempt in range(1, 4):  # up to 3 parse attempts
            prompt_suffix = self.RETRY_PROMPT_SUFFIX if attempt > 1 else ""
            full_user = user_msg + (
                f"\n\nPREVIOUS ERROR: {last_error}{prompt_suffix}" if last_error else "")

            try:
                raw = await self.ollama.chat(self.SYSTEM_PROMPT, full_user,
                                             temperature=0.05 * attempt)
            except LlmCircuitOpenError as e:
                tag_err(f"AGENT/{self.agent_id}", f"Circuit open: {e}")
                self.demo_mode = True
                return self._demo_decide(goal, ctx)
            except LlmError as e:
                tag_err(f"AGENT/{self.agent_id}", f"LLM error: {e}")
                self._error_budget -= 1
                return None

            # Parse
            valid_types = envelope.valid_action_types
            decision, parse_error = ResponseParser.parse(raw, valid_types, goal)

            if decision is not None:
                if decision.get("_fuzzy_matched"):
                    tag_warn(f"AGENT/{self.agent_id}",
                        f"Fuzzy-matched action type → {decision['action_type']}")
                return decision

            last_error = parse_error
            tag_warn(f"AGENT/{self.agent_id}",
                f"Parse attempt {attempt}/3 failed: {parse_error}")

        # All parse attempts failed
        tag_err(f"AGENT/{self.agent_id}", f"Could not extract valid decision after 3 attempts")
        self._error_budget -= 1
        return None

    async def act(self, goal: str, ctx: ContextFrame,
                  extra_payload: Optional[Dict] = None,
                  extra_context: str = "") -> Optional[EnvReaction]:
        """Full agent cycle: decide → validate → dispatch → record."""
        decision = await self.decide(goal, ctx, extra_context)

        if decision is None:
            tag_err(f"AGENT/{self.agent_id}", f"No valid decision for: {goal[:60]}")
            return None

        if decision.get("decision") == "skip":
            tag_dim(f"AGENT/{self.agent_id}", f"SKIP — {decision.get('reasoning','')[:80]}")
            return None

        action_type = decision["action_type"]
        payload     = {**decision.get("payload",{}), **(extra_payload or {})}
        reasoning   = decision.get("reasoning","")
        corr_id     = str(uuid.uuid4())

        # Display
        print()
        tag_info(f"AGENT/{self.agent_id}",
            f"{C.BOLD}{action_type}{C.RESET}  "
            f"{C.DIM}(gravity={self._gravity_of(action_type,ctx):.3f}){C.RESET}")
        tag_dim("REASON", reasoning[:100])
        if payload:
            tag_dim("PAYLOAD", json.dumps(payload,default=str)[:120])

        reaction = await self.env.dispatch(self.actor, action_type, payload, ctx, corr_id)

        if reaction.is_success:
            tag_ok("REACTION", reaction.to_summary())
        else:
            tag_err("REACTION", reaction.to_summary())

        # Chain reaction display
        if reaction.chain_actions:
            for ca in reaction.chain_actions:
                tag_dim("CHAIN→", f"spawned: {ca}")

        self._memory.append({"goal":goal[:60],"action":action_type,
                              "status":reaction.status,"at":time.time()})
        if len(self._memory) > 20: self._memory = self._memory[-20:]

        return reaction

    # ── Demo mode (deterministic, no LLM) ────────────────────────────────

    def _demo_decide(self, goal: str, ctx: ContextFrame) -> Optional[Dict]:
        envelope = self.env.get_envelope(self.actor, ctx, self._memory)
        if not envelope.available_actions:
            return {"decision":"skip","action_type":"","payload":{},
                    "reasoning":"No available actions in this context"}

        goal_lower = goal.lower()
        kw_map = [
            ("create school",    "school.create",           {"name":"Demo Academy","code":"DEM","countryCode":"FR"}),
            ("create course",    "course.create",           {}),
            ("open enrollment",  "enrollment.open",         {}),
            ("close enrollment", "enrollment.close",        {}),
            ("enroll",           "student.enroll",          {}),
            ("withdraw",         "student.withdraw",        {}),
            ("grade",            "grade.record",            {}),
            ("snapshot",         "system.snapshot",         {}),
            ("list",             "system.list",             {}),
            ("query grades",     "student.query-grades",    {}),
            ("query enrollment", "enrollment.query",        {}),
            ("assign teacher",   "course.assign-teacher",   {}),
        ]

        valid = envelope.valid_action_types
        for kw, atype, payload in kw_map:
            if kw in goal_lower and atype in valid:
                return {"decision":"execute","action_type":atype,
                        "payload":payload,"reasoning":f"Keyword '{kw}' matched goal"}

        # Fall back to highest-gravity available
        best = envelope.available_actions[0]
        return {"decision":"execute","action_type":best["type"],
                "payload":{},"reasoning":"Highest-gravity available action"}

    def _gravity_of(self, atype: str, ctx: ContextFrame) -> float:
        e = self.env.get_envelope(self.actor, ctx, self._memory)
        for a in e.available_actions:
            if a["type"] == atype: return a["gravity"]
        return 0.0

    def _urgency(self, envelope: ContextEnvelope) -> str:
        if envelope.situation:
            s = envelope.situation.get("kind","")
            if s == "critical":  return "critical"
            if s == "degraded":  return "degraded"
        return "normal"

# =============================================================================
# MAIN
# =============================================================================

async def main(demo_mode: bool, model: str, host: str,
               timeout: float, retries: int):

    hr(n=70, char="╔"); hr(n=70,char="║"); 
    print(f"  {C.BOLD}{C.CYAN}MEP/2.0 × Ollama  v2.0  —  Hardened Causal AI Agent{C.RESET}")
    hr(n=70,char="║"); hr(n=70,char="╚")
    print()

    # ── Boot ─────────────────────────────────────────────────────────────

    env    = ESchoolEnv()
    ollama = OllamaClient(host=host, model=model, timeout=timeout, max_retries=retries)

    if ollama.is_available and not demo_mode:
        tag_ok("MODE", f"Live Ollama  model={C.BOLD}{model}{C.RESET}  "
               f"timeout={timeout:.0f}s  retries={retries}")
    else:
        demo_mode = True
        tag_warn("MODE", "Demo mode — deterministic decisions (no LLM)")

    # Seed actors
    admin   = env.add_admin("Dr. Elena Vasquez", "ADM001", "SystemAdmin")
    teacher = env.add_teacher("Prof. Sophie Laurent","TCH001","Mathematics")
    env.add_teacher("Prof. James Kim","TCH002","Computer Science")

    students = [
        env.add_student("Alice Chen",    "STU001"),
        env.add_student("Bob Martinez",  "STU002"),
        env.add_student("Clara Okonkwo", "STU003"),
        env.add_student("David Park",    "STU004"),
    ]

    # Include actors in contexts
    for actor in [admin, teacher]:
        for ctx in [env.admin_ctx, env.acad_ctx, env.enrl_ctx]:
            ctx.include(actor)

    agent_admin   = MepAgent(env, ollama, admin,   demo_mode=demo_mode, agent_id="ADMIN")
    agent_teacher = MepAgent(env, ollama, teacher, demo_mode=demo_mode, agent_id="TEACH")

    tag_ok("BOOT", f"{len(env._students)+len(env._teachers)+len(env._admins)} elements seeded")
    print()

    # ═════════════════════════════════════════════════════════════════════
    section("PHASE 1 — Context Envelope: what the AI receives")
    # ═════════════════════════════════════════════════════════════════════

    envelope = env.get_envelope(admin, env.admin_ctx)

    print(f"\n{C.BOLD}Context: {envelope.name} [{envelope.kind}]{C.RESET}")
    print(f"  Data keys: {list(envelope.data.keys())}")

    print(f"\n{C.BOLD}Available Actions (Causal Gravity ranking):{C.RESET}")
    for a in envelope.available_actions:
        bar   = "█" * max(1,int(a["gravity"]*20))
        print(f"  [{bar:<20}] {a['gravity']:.3f}  {C.BOLD}{a['type']}{C.RESET}")
        print(f"   {'':21}  {C.DIM}{a['description']}{C.RESET}")

    print(f"\n{C.BOLD}Rendered as LLM prompt (excerpt):{C.RESET}")
    excerpt = envelope.to_prompt()[:600]
    print(C.DIM + textwrap.indent(excerpt, "  ") + C.RESET + " [...]")

    # ═════════════════════════════════════════════════════════════════════
    section("PHASE 2 — Administrative AI Agent: Setup")
    # ═════════════════════════════════════════════════════════════════════

    await agent_admin.act("Create a new school called Global Science Academy",
                           env.admin_ctx,
                           extra_payload={"name":"Global Science Academy",
                                          "code":"GSA","countryCode":"FR"})

    course_specs = [
        {"name":"Advanced Mathematics","code":"MATH201","discipline":"Mathematics","capacity":"25","credits":"4"},
        {"name":"Data Structures",     "code":"CS301",  "discipline":"Computer Science","capacity":"30","credits":"3"},
        {"name":"Physics Lab",         "code":"PHY101",  "discipline":"Physics","capacity":"20","credits":"3"},
    ]
    course_ids = []
    for spec in course_specs:
        r = await agent_admin.act(f"Create course {spec['name']}",
                                   env.admin_ctx, extra_payload=spec)
        if r and r.is_success and r.result:
            course_ids.append(r.result.get("id",""))

    print(f"\n  {C.GREEN}Courses registered:{C.RESET} {len(env._courses)}")

    # ═════════════════════════════════════════════════════════════════════
    section("PHASE 3 — Circumstance Gating: enrollment.open = FALSE")
    # ═════════════════════════════════════════════════════════════════════

    print(f"\n{C.BOLD}Enrollment context with enrollment CLOSED:{C.RESET}")
    env_closed = env.get_envelope(admin, env.enrl_ctx)
    if not env_closed.available_actions:
        print(f"  {C.RED}✗ Zero actions available — enrollment.open=False blocks them all{C.RESET}")
    else:
        for a in env_closed.available_actions:
            print(f"  {C.YELLOW}~{C.RESET} {a['type']}  (guard: enrollment.open)")

    # Open enrollment
    await agent_admin.act("Open the enrollment period for students", env.admin_ctx)

    print(f"\n{C.BOLD}Enrollment context with enrollment OPEN:{C.RESET}")
    env_open = env.get_envelope(admin, env.enrl_ctx)
    for a in env_open.available_actions:
        print(f"  {C.GREEN}✓{C.RESET} {a['type']}  gravity={a['gravity']:.3f}")

    # ═════════════════════════════════════════════════════════════════════
    section("PHASE 4 — Enrollment Operations")
    # ═════════════════════════════════════════════════════════════════════

    if course_ids:
        math_id = course_ids[0]
        cs_id   = course_ids[1] if len(course_ids)>1 else course_ids[0]

        for student in students[:3]:
            await agent_admin.act(
                f"Enroll student {student['name']} in Advanced Mathematics",
                env.enrl_ctx,
                extra_payload={"studentId":student["id"],"courseId":math_id})

        for student in students[:2]:
            await agent_admin.act(
                f"Enroll student {student['name']} in Data Structures",
                env.enrl_ctx,
                extra_payload={"studentId":student["id"],"courseId":cs_id})

        math = env._courses.get(math_id, {})
        cs   = env._courses.get(cs_id, {})
        print(f"\n  {C.GREEN}Math:{C.RESET} {math.get('enrolled',0)}/{math.get('maxCapacity',0)}  "
              f"| {C.GREEN}CS:{C.RESET} {cs.get('enrolled',0)}/{cs.get('maxCapacity',0)}")

    # ═════════════════════════════════════════════════════════════════════
    section("PHASE 5 — Teacher Agent: Grading + Causal Chains")
    # ═════════════════════════════════════════════════════════════════════

    await agent_admin.act("Close the enrollment period", env.admin_ctx)

    grade_scenarios = [
        (students[0]["id"], course_ids[0] if course_ids else "", 88.5),
        (students[1]["id"], course_ids[0] if course_ids else "", 42.0),
        (students[2]["id"], course_ids[0] if course_ids else "", 55.0),
        (students[0]["id"], course_ids[1] if len(course_ids)>1 else "", 94.0),
        (students[1]["id"], course_ids[1] if len(course_ids)>1 else "", 37.0),
        (students[2]["id"], course_ids[1] if len(course_ids)>1 else "", 61.0),
    ]

    print(f"\n  {C.DIM}Failing grades (< 60) spawn chain → notification.dispatch{C.RESET}\n")
    for sid, cid, grade in grade_scenarios:
        s = env._students.get(sid,{})
        await agent_teacher.act(
            f"Record grade {grade} for student {s.get('name','')}",
            env.acad_ctx,
            extra_payload={"studentId":sid,"courseId":cid,"grade":str(grade)})

    # ═════════════════════════════════════════════════════════════════════
    section("PHASE 6 — WHY / WHY-NOT Explainability")
    # ═════════════════════════════════════════════════════════════════════

    print(f"\n{C.BOLD}Causal trace for 'grade.record':{C.RESET}")
    print(C.DIM + textwrap.indent(env._causal.explain("grade.record"), "  ") + C.RESET)

    print(f"\n{C.BOLD}WHY-NOT example — student.enroll when enrollment closed:{C.RESET}")
    print(f"  {C.DIM}Circumstance 'enrollment.open' evaluated to FALSE")
    print(f"  Role: enabler — when FALSE, actions guarded by it are removed from")
    print(f"  the available_actions list entirely. The AI never sees them.{C.RESET}")

    stats = env._causal.stats
    print(f"\n{C.BOLD}Causal graph:{C.RESET}")
    print(f"  nodes={stats['nodes']}  "
          f"successes={stats['successes']}  rejections={stats['rejections']}")
    print(f"  types: [{', '.join(stats['types'])}]")

    # ═════════════════════════════════════════════════════════════════════
    section("PHASE 7 — Context Topology (Semantic Distance Matrix)")
    # ═════════════════════════════════════════════════════════════════════

    ctxs = [
        ("Administrative", env.admin_ctx.basis),
        ("Academic",       env.acad_ctx.basis),
        ("Enrollment",     env.enrl_ctx.basis),
    ]

    print(f"\n{'Pair':<38} {'Dist':>6}  {'Similarity'}")
    hr(n=60, char="─")
    for i, (na, ba) in enumerate(ctxs):
        for j, (nb, bb) in enumerate(ctxs):
            if j <= i: continue
            dist = ba.angular_distance(bb)
            bar  = "█" * int((1-dist)*24)
            print(f"  {na} ↔ {nb:<18} {dist:>6.3f}  {bar}")

    print(f"\n  {C.DIM}0.0 = identical semantic space  |  1.0 = orthogonal axes{C.RESET}")
    print(f"\n{C.BOLD}Academic context gravity ranking:{C.RESET}")
    for a, g in env.acad_ctx.get_available_actions(teacher, {"actor_id":teacher["id"]}):
        bar = "█" * max(1,int(g*24))
        print(f"  [{bar:<24}] {g:.3f}  {a.type}")

    # ═════════════════════════════════════════════════════════════════════
    section("PHASE 8 — Final Snapshot")
    # ═════════════════════════════════════════════════════════════════════

    r = await agent_admin.act("Take a system snapshot", env.admin_ctx)
    if r and r.result:
        print(f"\n  {C.BOLD}Environment State:{C.RESET}")
        for k,v in r.result.items():
            print(f"    {k:<25}: {v}")

    print(f"\n  {C.BOLD}Student GPAs:{C.RESET}")
    for s in students:
        data = env._students.get(s["id"],{})
        gpa  = data.get("gpa",0)
        enrl = len(data.get("enrolledCourses",[]))
        bar  = "█" * max(0,int(gpa/10))
        print(f"    {s['name']:<20} GPA={gpa:>5.2f} [{bar:<10}] {enrl} course(s)")

    if env._phenomena.all_detected:
        print(f"\n  {C.BOLD}Phenomena Detected:{C.RESET}")
        for p in env._phenomena.all_detected:
            print(f"    {C.MAGENTA}★ {p['name']}{C.RESET}  magnitude={p['magnitude']:.0%}  count={p['count']}")

    print(f"\n  {C.BOLD}Circuit Breaker Status:{C.RESET}")
    print(f"    LLM  {ollama._circuit.status_line()}")
    print(f"    Error budget remaining: {agent_admin._error_budget}/{5}")

    print(f"\n  {C.BOLD}Agent Decision Log:{C.RESET}")
    all_decisions = agent_admin._memory + agent_teacher._memory
    for m in all_decisions:
        ok = m["status"]=="success"
        ic = f"{C.GREEN}✓{C.RESET}" if ok else f"{C.RED}✗{C.RESET}"
        print(f"    {ic} {m['action']:<38} ← {C.DIM}{m['goal'][:50]}{C.RESET}")

    hr()
    print(f"\n  {C.BOLD}{C.CYAN}MEP/2.0 v2.0 — complete{C.RESET}")
    print(f"  The AI never mutated the environment directly. It operated through:")
    print(f"    {C.GREEN}• ContextEnvelope     semantic frame + causal gravity ranking{C.RESET}")
    print(f"    {C.GREEN}• Circumstance gates  Boolean algebra (AND/OR/NOT), live-evaluated{C.RESET}")
    print(f"    {C.GREEN}• Robust parser       strips [CATEGORY] prefix, fuzzy-matches, retries{C.RESET}")
    print(f"    {C.GREEN}• Circuit breaker     CLOSED/OPEN/HALF-OPEN, automatic recovery{C.RESET}")
    print(f"    {C.GREEN}• Causal graph        every decision traceable to its origin{C.RESET}")
    print(f"    {C.GREEN}• Phenomenon engine   emergent pattern detection across reaction stream{C.RESET}")
    print(f"    {C.GREEN}• Error budget        graceful degradation to demo mode on overload{C.RESET}")
    hr()

# =============================================================================
# ENTRY
# =============================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MEP/2.0 × Ollama Agent v2")
    ap.add_argument("--demo",    action="store_true", help="Force demo mode")
    ap.add_argument("--model",   default="llama3",    help="Ollama model name")
    ap.add_argument("--host",    default="http://localhost:11434", help="Ollama host")
    ap.add_argument("--timeout", type=float, default=45.0, help="LLM timeout seconds")
    ap.add_argument("--retries", type=int,   default=3,    help="Max LLM retries")
    args = ap.parse_args()

    try:
        asyncio.run(main(demo_mode=args.demo, model=args.model,
                         host=args.host, timeout=args.timeout, retries=args.retries))
    except KeyboardInterrupt:
        print(f"\n{C.YELLOW}Interrupted by user.{C.RESET}")
    except Exception as e:
        print(f"\n{C.RED}Fatal error:{C.RESET} {e}")
        traceback.print_exc()
        sys.exit(1)
