"""
edp.py  —  Environment Design Pattern  v4.1
════════════════════════════════════════════════════════════════════════════
OneOrigine / ImperialSchool Research  —  I.S. License (Attributed Open)

MATHEMATICAL INNOVATIONS (from the axiom system):
  ─────────────────────────────────────────────────
  • Central Equation:
      E_{t+1} = 𝔘( E_t, 𝔯( x_t, c_t, Σ_t, 𝔄(x_t, c_t, Σ_t, Γ_t, Ψ(D_t,c_t,Σ_t)) ) )

  • Sense Vector ∈ ℝ⁸  (causal/temporal/spatial/normative/social/financial/technical/emergent)
  • Causal Gravity:  g(a,c) = cos(φ_A(a), φ_C(c)) = A·C / (|A|·|C|)
  • Harmony Function:  H = α·cos(A,C) + β·cos(A,S) + γ·cos(R̂,R) − δ·D_t
  • Context as Operator:  M_c : ℝⁿ → ℝⁿ  (context deforms action space)
  • Causal Delta:  Δ_t = φ_Σ(Σ_{t+1}) − φ_Σ(Σ_t)
  • Phenomenon as Attractor:  k* = argmin_k d(φ(π_t), μ_k)
  • Circumstance Algebra:  (Γ, ∧, ∨, ¬, ⊤, ⊥)
  • Admissibility:  Adm(a) = ∏_{γ∈Γ_a} γ(E,c,Σ,x)  (Boolean product)
  • Action Optimum:  a* = argmax_{a∈Avail} H(a,c,s,r)

  These are implemented as concrete operations — not metadata.
"""

from __future__ import annotations

import asyncio, math, time, uuid, json
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Deque

__version__ = "4.1.0"
__author__  = "OneOrigine"
__license__ = "I.S."

# ═════════════════════════════════════════════════════════════════════════════
# MATHEMATICAL LAYER — φ_A, φ_C, φ_S, φ_R ∈ ℝ⁸
# ═════════════════════════════════════════════════════════════════════════════

DIMS = 8  # semantic space dimensionality

class Dim(int, Enum):
    """Semantic axis index within ℝ⁸."""
    CAUSAL    = 0
    TEMPORAL  = 1
    SPATIAL   = 2
    NORMATIVE = 3
    SOCIAL    = 4
    FINANCIAL = 5
    TECHNICAL = 6
    EMERGENT  = 7


@dataclass(frozen=True)
class SenseVector:
    """
    φ ∈ ℝ⁸ — A positioned meaning in 8-dimensional semantic space.
    Used for: Actions, Reactions, Contexts, Elements, Phenomena.

    All semantic comparisons in EDP derive from this vector:
      - Causal Gravity    = cos(φ_A, φ_C)
      - Context Distance  = acos(cos(φ_C1, φ_C2)) / π
      - Harmony Component = cos(φ_A, φ_S)
    """
    dimension: str
    meaning:   str
    magnitude: float
    v:         Tuple[float, ...]  # the actual ℝ⁸ vector

    @classmethod
    def of(cls, dim: str, meaning: str, axis: Dim, mag: float = 1.0) -> "SenseVector":
        vec = [0.0] * DIMS
        vec[axis.value] = mag
        return cls(dim, meaning, mag, tuple(vec))

    # ── Cosine similarity: cos(θ) = A·B / (|A||B|) ────────────────────────

    def dot(self, other: "SenseVector") -> float:
        return sum(a * b for a, b in zip(self.v, other.v))

    def norm(self) -> float:
        return math.sqrt(sum(x * x for x in self.v))

    def cosine(self, other: "SenseVector") -> float:
        na, nb = self.norm(), other.norm()
        return self.dot(other) / (na * nb) if na > 0 and nb > 0 else 0.0

    def angular_distance(self, other: "SenseVector") -> float:
        """ω = acos(cos(θ)) / π ∈ [0,1]. 0=identical, 1=opposite."""
        return math.acos(max(-1.0, min(1.0, self.cosine(other)))) / math.pi

    # ── Context-as-operator: M_c transforms this vector ────────────────────

    def apply_context_operator(self, ctx_vector: "SenseVector",
                                alpha: float = 0.7) -> "SenseVector":
        """
        R = α·A + (1-α)·(A ⊙ C)  — Hadamard blend with context.
        Context acts as a deforming operator on the sense space.
        """
        hadamard = tuple(a * b for a, b in zip(self.v, ctx_vector.v))
        blended  = tuple(alpha * a + (1-alpha) * h
                         for a, h in zip(self.v, hadamard))
        # Renormalize
        n = math.sqrt(sum(x*x for x in blended)) or 1.0
        normed = tuple(x/n for x in blended)
        return SenseVector(self.dimension, f"{self.meaning}@{ctx_vector.meaning}",
                           self.magnitude, normed)

    # ── Causal delta: Δ_t = φ(Σ_{t+1}) - φ(Σ_t) ──────────────────────────

    def delta(self, other: "SenseVector") -> "SenseVector":
        """Directional change between two semantic states."""
        d = tuple(b - a for a, b in zip(self.v, other.v))
        mag = math.sqrt(sum(x*x for x in d)) or 0.0
        return SenseVector("delta", f"Δ({self.meaning}→{other.meaning})", mag, d)

    # ── Factories ─────────────────────────────────────────────────────────

    @classmethod
    def causal(cls,    m: str, g: float=1.0): return cls.of("causal",    m, Dim.CAUSAL,    g)
    @classmethod
    def temporal(cls,  m: str, g: float=1.0): return cls.of("temporal",  m, Dim.TEMPORAL,  g)
    @classmethod
    def spatial(cls,   m: str, g: float=1.0): return cls.of("spatial",   m, Dim.SPATIAL,   g)
    @classmethod
    def normative(cls, m: str, g: float=1.0): return cls.of("normative", m, Dim.NORMATIVE, g)
    @classmethod
    def social(cls,    m: str, g: float=1.0): return cls.of("social",    m, Dim.SOCIAL,    g)
    @classmethod
    def financial(cls, m: str, g: float=1.0): return cls.of("financial", m, Dim.FINANCIAL, g)
    @classmethod
    def technical(cls, m: str, g: float=1.0): return cls.of("technical", m, Dim.TECHNICAL, g)
    @classmethod
    def emergent(cls,  m: str, g: float=1.0): return cls.of("emergent",  m, Dim.EMERGENT,  g)

    def __repr__(self):
        return f"φ({self.dimension}:{self.meaning} |{self.magnitude:.2f}|)"

SENSE_NULL = SenseVector("none", "", 0.0, tuple([0.0]*DIMS))


# ═════════════════════════════════════════════════════════════════════════════
# HARMONY FUNCTION  H(A,C,S,R,Σ)
# H = α·cos(A,C) + β·cos(A,S) + γ·cos(R̂,R) − δ·D
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class HarmonyProfile:
    """
    Multi-dimensional harmony score for an action in a context.
    H = α·context_alignment + β·semantic_alignment + γ·reaction_coherence − δ·dissonance

    Replaces single gravity score with a rich semantic profile.
    """
    context_alignment  : float  # cos(φ_A, φ_C)     — how natural in this context
    semantic_alignment : float  # cos(φ_A, φ_S)     — how aligned with current sense
    reaction_coherence : float  # cos(R̂, R_observed) — predicted vs actual match
    dissonance         : float  # ||R_expected - R_observed|| — divergence penalty
    action_type        : str    = ""

    # Weights (tunable per domain)
    ALPHA: float = 0.35   # context alignment weight
    BETA:  float = 0.30   # semantic alignment weight
    GAMMA: float = 0.25   # reaction coherence weight
    DELTA: float = 0.10   # dissonance penalty weight

    @property
    def score(self) -> float:
        """H ∈ [-1, 1] — the unified harmony measure."""
        return (self.ALPHA * self.context_alignment
              + self.BETA  * self.semantic_alignment
              + self.GAMMA * self.reaction_coherence
              - self.DELTA * self.dissonance)

    @property
    def gravity(self) -> float:
        """Backward-compatible gravity = context_alignment."""
        return self.context_alignment

    def to_dict(self) -> Dict:
        return {"action_type": self.action_type, "score": round(self.score, 4),
                "gravity": round(self.gravity, 4),
                "ctx_align": round(self.context_alignment, 3),
                "sem_align": round(self.semantic_alignment, 3),
                "rxn_coher": round(self.reaction_coherence, 3),
                "dissonance": round(self.dissonance, 3)}


def compute_harmony(action_vec: SenseVector, ctx_vec: SenseVector,
                    sense_vec: SenseVector,
                    expected_rxn: Optional[SenseVector] = None,
                    observed_rxn: Optional[SenseVector] = None,
                    action_type: str = "") -> HarmonyProfile:
    """
    Compute H(A,C,S,R̂,R) for an action candidate.
    Used by context.get_available_actions to rank and score actions.
    """
    ctx_align = action_vec.cosine(ctx_vec)
    sem_align = action_vec.cosine(sense_vec) if sense_vec != SENSE_NULL else ctx_align

    rxn_coher = 0.0
    dissonance = 0.0
    if expected_rxn and observed_rxn:
        rxn_coher  = expected_rxn.cosine(observed_rxn)
        # Euclidean distance as dissonance measure
        delta_v    = expected_rxn.delta(observed_rxn)
        dissonance = min(1.0, delta_v.magnitude)

    return HarmonyProfile(ctx_align, sem_align, rxn_coher, dissonance, action_type)


# ═════════════════════════════════════════════════════════════════════════════
# CIRCUMSTANCE ALGEBRA  (Γ, ∧, ∨, ¬, ⊤, ⊥)
# Adm(a) = ∏_{γ∈Γ_a} γ(E,c,Σ,x)  — Boolean product
# ═════════════════════════════════════════════════════════════════════════════

class Circumstance:
    """
    A live Boolean predicate γ : E × C × Σ × X → {0,1}.

    Algebraic composition:
      c1 & c2  → AND   (∧)
      c1 | c2  → OR    (∨)
      ~c       → NOT   (¬)

    Role semantics:
      enabler  — when True, actions become available
      blocker  — when True, actions are gated off
      modifier — when True, action behavior changes
    """

    def __init__(self, cid: str, desc: str, fn: Callable,
                 role: str = "enabler", weight: float = 1.0,
                 kind: str = "logical"):
        self.id = cid; self.description = desc
        self._fn = fn; self.role = role
        self.weight = weight; self.kind = kind

    def evaluate(self, ctx: "Context", frame: Dict) -> bool:
        """γ(E,c,Σ,x) → {0,1}"""
        try:   return bool(self._fn(ctx, frame))
        except: return False

    def evaluate_with_trace(self, ctx: "Context", frame: Dict) -> "CircumstanceEval":
        """Full evaluation with WHY/WHY-NOT traceability."""
        holds = self.evaluate(ctx, frame)
        return CircumstanceEval(
            circumstance=self, holds=holds,
            reason=f"{'✓' if holds else '✗'} {self.description}",
            blocker=not holds and self.role == "enabler")

    # ── Algebraic operators ────────────────────────────────────────────────

    def __and__(self, o: "Circumstance") -> "Circumstance":
        return Circumstance(
            f"({self.id}∧{o.id})", f"({self.description}) AND ({o.description})",
            lambda c, f: self._fn(c, f) and o._fn(c, f),
            "enabler", min(self.weight, o.weight))

    def __or__(self, o: "Circumstance") -> "Circumstance":
        return Circumstance(
            f"({self.id}∨{o.id})", f"({self.description}) OR ({o.description})",
            lambda c, f: self._fn(c, f) or o._fn(c, f),
            "enabler", max(self.weight, o.weight))

    def __invert__(self) -> "Circumstance":
        return Circumstance(
            f"¬{self.id}", f"NOT ({self.description})",
            lambda c, f: not self._fn(c, f), "blocker", -self.weight)

    def __xor__(self, o: "Circumstance") -> "Circumstance":
        return Circumstance(
            f"({self.id}⊕{o.id})", f"({self.description}) XOR ({o.description})",
            lambda c, f: bool(self._fn(c, f)) != bool(o._fn(c, f)),
            "enabler", abs(self.weight - o.weight))

    # ── Factories ─────────────────────────────────────────────────────────

    @classmethod
    def when(cls, cid: str, desc: str, fn: Callable,
             role: str = "enabler", weight: float = 1.0) -> "Circumstance":
        return cls(cid, desc, fn, role, weight)

    @classmethod
    def always(cls, cid: str = "⊤") -> "Circumstance":
        return cls(cid, f"Always ({cid})", lambda *_: True, "enabler", 1.0)

    @classmethod
    def never(cls, cid: str = "⊥") -> "Circumstance":
        return cls(cid, f"Never ({cid})", lambda *_: False, "blocker", -1.0)

    @classmethod
    def flag(cls, cid: str, desc: str, key: str, val: Any = True) -> "Circumstance":
        return cls(cid, desc, lambda ctx, _: ctx.data.get(key) == val)

    @classmethod
    def threshold(cls, cid: str, desc: str, fn: Callable[[Any], bool],
                  key: str) -> "Circumstance":
        return cls(cid, desc,
            lambda ctx, _: fn(ctx.data[key]) if key in ctx.data else False,
            kind="threshold")

    @classmethod
    def role_check(cls, cid: str, element_type: str) -> "Circumstance":
        return cls(cid, f"Actor is {element_type}",
            lambda ctx, frame: any(
                e.get("element_type") == element_type
                for e in ctx.elements
                if e.get("element_id") == frame.get("actor_id")))

    def to_dict(self) -> Dict:
        return {"id": self.id, "description": self.description,
                "role": self.role, "weight": self.weight, "kind": self.kind}


@dataclass
class CircumstanceEval:
    """Full evaluation result — enables WHY/WHY-NOT queries."""
    circumstance : Circumstance
    holds        : bool
    reason       : str
    blocker      : bool = False  # True = this circumstance is preventing action


# ═════════════════════════════════════════════════════════════════════════════
# DATA PIPELINE:  IData → Ψ(D,c,Σ) → IContextualizedData → ISense
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class RawData:
    """IData — a raw measurement before contextual interpretation."""
    tag       : str
    value     : Any
    captured  : float = field(default_factory=time.time)
    source    : str   = ""

@dataclass
class ContextualizedData:
    """
    Ψ(D,c,Σ) → result of contextualizing raw data.
    Data does not have meaning until contextualized.
    """
    tag          : str
    value        : Any
    sense        : SenseVector
    frame        : str          # context name that produced this
    relevance    : float        # ∈ [0,1] — how relevant in this context
    is_actionable: bool         # can this information gate an action?
    at           : float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {"tag": self.tag, "value": self.value,
                "sense": self.sense.meaning, "dimension": self.sense.dimension,
                "relevance": round(self.relevance, 3),
                "is_actionable": self.is_actionable, "frame": self.frame}


# ═════════════════════════════════════════════════════════════════════════════
# REACTION  —  r : E → E'  (structured state transformation)
# ═════════════════════════════════════════════════════════════════════════════

class ReactionStatus(str, Enum):
    SUCCESS  = "success";  REJECTED = "rejected"
    DEFERRED = "deferred"; PARTIAL  = "partial"
    ERROR    = "error";    CHAIN_MAX = "chain_max"

class ImpactTarget(str, Enum):
    ACTOR="actor"; SPECIFIC="specific"; ENVIRONMENT="environment"
    ALL="all"; NONE="none"; BROADCAST="broadcast"

class TemporalMode(str, Enum):
    IMMEDIATE="immediate"; DEFERRED="deferred"
    RECURRING="recurring"; TEMPORARY="temporary"

@dataclass
class ImpactScope:
    target:    ImpactTarget = ImpactTarget.ACTOR
    element_id: Optional[str] = None
    magnitude:  float = 1.0
    @classmethod
    def on_actor(cls, m=1.0):        return cls(ImpactTarget.ACTOR,       None, m)
    @classmethod
    def on_element(cls, eid, m=1.0): return cls(ImpactTarget.SPECIFIC,    eid,  m)
    @classmethod
    def on_env(cls, m=0.5):          return cls(ImpactTarget.ENVIRONMENT, None, m)
    @classmethod
    def broadcast(cls, m=0.3):       return cls(ImpactTarget.ALL,         None, m)
    @classmethod
    def none(cls):                   return cls(ImpactTarget.NONE,        None, 0.)

@dataclass
class Temporality:
    mode:        TemporalMode = TemporalMode.IMMEDIATE
    delay_ms:    int = 0
    interval_ms: int = 0
    duration_ms: int = 0
    max_repeat:  int = -1
    @classmethod
    def immediate(cls):            return cls(TemporalMode.IMMEDIATE)
    @classmethod
    def deferred(cls, ms: int):    return cls(TemporalMode.DEFERRED, delay_ms=ms)
    @classmethod
    def recurring(cls, ms: int):   return cls(TemporalMode.RECURRING, interval_ms=ms)
    @classmethod
    def temporary(cls, ms: int):   return cls(TemporalMode.TEMPORARY, duration_ms=ms)

@dataclass
class Reaction:
    """
    The mandatory consequence of every action.
    r : E_t → E_{t+1}  (environment state transformation).

    Carries:
      • Semantic vector φ_R — enables harmony computation
      • Causal ancestry (correlation_id, causation_id, depth)
      • Impact scope (who is affected, how much)
      • Temporality (when the effect manifests)
      • Causal chain continuation (spawned actions)
    """
    action_type    : str
    status         : ReactionStatus
    message        : Optional[str] = None
    result         : Any           = None
    sense          : SenseVector   = field(default_factory=lambda: SENSE_NULL)
    impact         : ImpactScope   = field(default_factory=ImpactScope.on_actor)
    temporality    : Temporality   = field(default_factory=Temporality.immediate)
    chain_actions  : List[str]     = field(default_factory=list)
    reaction_id    : str           = field(default_factory=lambda: str(uuid.uuid4()))
    produced_at    : float         = field(default_factory=time.time)
    correlation_id : str           = ""
    causation_id   : str           = ""
    chain_depth    : int           = 0
    causal_delta   : Optional["SenseVector"] = None  # Δ_t = φ(Σ_{t+1}) - φ(Σ_t)

    @property
    def is_success(self) -> bool: return self.status == ReactionStatus.SUCCESS

    @classmethod
    def ok(cls, action_type: str, message: str = None, result: Any = None,
           sense: SenseVector = None, impact: ImpactScope = None,
           temporality: Temporality = None, chain: List[str] = None) -> "Reaction":
        return cls(action_type, ReactionStatus.SUCCESS, message, result,
                   sense or SENSE_NULL, impact or ImpactScope.on_actor(),
                   temporality or Temporality.immediate(), chain or [])

    @classmethod
    def reject(cls, action_type: str, reason: str) -> "Reaction":
        return cls(action_type, ReactionStatus.REJECTED, reason,
                   impact=ImpactScope.none())

    @classmethod
    def deferred(cls, action_type: str, delay_ms: int, msg: str = None) -> "Reaction":
        return cls(action_type, ReactionStatus.DEFERRED, msg,
                   temporality=Temporality.deferred(delay_ms))

    def line(self) -> str:
        i = "✓" if self.is_success else "✗"
        r = f" → {json.dumps(self.result, default=str)[:60]}" if self.result else ""
        m = f" | {self.message}" if self.message else ""
        return f"{i} {self.action_type}{m}{r}"


# ═════════════════════════════════════════════════════════════════════════════
# ACTION  —  𝔄 : X × C × Σ × Γ × S → A
# a* = argmax_{a ∈ Avail} H(a,c,s,r)
# ═════════════════════════════════════════════════════════════════════════════

class ActionCategory(str, Enum):
    COMMAND   = "command";  QUERY     = "query"
    SIGNAL    = "signal";   TRANSFORM = "transform"
    LIFECYCLE = "lifecycle"

class Action:
    """
    A discoverable possibility in the environment.
    NOT a method owned by an element. Elements discover actions via context.

    Carries a SenseVector φ_A for:
      • Causal gravity:   g(a,c) = cos(φ_A, φ_C)
      • Harmony score:    H(A,C,S,R)
      • Context-operator: M_c(φ_A)
    """
    def __init__(self, type: str, category: ActionCategory, description: str,
                 sense: SenseVector, guards: List[Circumstance] = None,
                 handler: Callable = None,
                 expected_reaction_sense: SenseVector = None):
        self.type     = type
        self.category = category
        self.description = description
        self.sense    = sense
        self.guards   = guards or []
        self._handler = handler
        self.expected_rxn_sense = expected_reaction_sense  # for harmony prediction
        self.action_id = str(uuid.uuid4())

    def admissible(self, actor, ctx: "Context", frame: Dict) -> bool:
        """Adm(a) = ∏_{γ∈Γ_a} γ(E,c,Σ,x)  — product of Boolean guards."""
        return all(g.evaluate(ctx, frame) for g in self.guards)

    def why_not_admissible(self, actor, ctx: "Context",
                            frame: Dict) -> List[CircumstanceEval]:
        """WHY-NOT: which circumstances are blocking this action?"""
        return [g.evaluate_with_trace(ctx, frame)
                for g in self.guards if not g.evaluate(ctx, frame)]

    async def execute(self, actor: Any, payload: Dict,
                      ctx: "Context", frame: Dict) -> Reaction:
        for g in self.guards:
            if not g.evaluate(ctx, frame):
                return Reaction.reject(self.type,
                    f"Circumstance not met: {g.description}")
        if not self._handler:
            return Reaction.ok(self.type, "OK (no handler)")
        try:
            r = await self._handler(actor, payload, ctx, frame)
            return r
        except Exception as e:
            return Reaction(self.type, ReactionStatus.ERROR, str(e),
                           impact=ImpactScope.none())

    def harmony_in(self, ctx: "Context",
                    current_sense: SenseVector = None) -> HarmonyProfile:
        """Compute H(A,C,S) for this action in this context."""
        s = current_sense or ctx.basis
        return compute_harmony(self.sense, ctx.basis, s,
                               self.expected_rxn_sense, None, self.type)

    def to_dict(self, ctx: "Context" = None,
                sense: SenseVector = None) -> Dict:
        h = self.harmony_in(ctx, sense) if ctx else None
        return {"type": self.type, "category": self.category.value,
                "description": self.description,
                "sense_dim": self.sense.dimension,
                **(h.to_dict() if h else {"gravity": 0.0, "score": 0.0})}


# ═════════════════════════════════════════════════════════════════════════════
# CONTEXT  c : E_t → E_t^(c)
# Multidimensional, fractal, gravitational — actions ranked by harmony
# ═════════════════════════════════════════════════════════════════════════════

class ContextKind(str, Enum):
    GLOBAL="global"; TEMPORAL="temporal"; SPATIAL="spatial"
    SEMANTIC="semantic"; RELATIONAL="relational"
    TRANSACTIONAL="transactional"; COMPOSITE="composite"
    CAUSAL="causal"; OBSERVATION="observation"; GOVERNANCE="governance"

class Context:
    """
    c : E_t → E_t^(c)  — a logical frame on the environment.

    Properties:
      • Gravitational: basis vector φ_C ∈ ℝ⁸, actions ranked by cos(φ_A, φ_C)
      • Fractal:       parent chain, depth, COW narrowing
      • Algebraic:     circumstances compose (∧, ∨, ¬)
      • Contextualizer: transforms IRawData → IContextualizedData
      • Transportable: serializable to dict/JSON for MEP
    """

    def __init__(self, name: str, kind: ContextKind = ContextKind.SEMANTIC,
                 basis: SenseVector = None, parent: "Context" = None,
                 depth: int = 0):
        self.ctx_id   = str(uuid.uuid4())
        self.name     = name
        self.kind     = kind
        self.basis    = basis or SENSE_NULL
        self.parent   = parent
        self.depth    = depth
        self.opened_at = time.time()
        self.closed_at: Optional[float] = None
        self.data     : Dict[str, Any]  = {}
        self.elements : List[Dict]      = []
        self.circumstances: List[Circumstance] = []
        self._actions : List[Dict]      = []
        self._info_cache: Dict[str, ContextualizedData] = {}

    @property
    def is_closed(self) -> bool: return self.closed_at is not None

    def close(self): self.closed_at = time.time()

    def add_circ(self, c: Circumstance) -> "Context":
        self.circumstances.append(c); return self

    def include(self, e: Dict) -> "Context":
        self.elements.append(e); return self

    def set(self, k: str, v: Any) -> "Context":
        self.data[k] = v; return self

    def reg(self, action: Action, actor_filter: Callable = None) -> "Context":
        self._actions.append({"a": action, "f": actor_filter}); return self

    def resolve(self, key: str, default: Any = None) -> Any:
        """Walk parent chain — child overrides parent."""
        ctx = self
        while ctx:
            if key in ctx.data: return ctx.data[key]
            ctx = ctx.parent
        return default

    # ── Ψ(D,c,Σ) : contextualizer ─────────────────────────────────────────

    def contextualize(self, raw: RawData,
                      sense: SenseVector) -> ContextualizedData:
        """
        Ψ : IData × IContext × IΣ → IContextualizedData
        Raw data acquires meaning through the context frame.
        """
        relevance  = float(sense.magnitude) * sense.cosine(self.basis)
        relevance  = max(0.0, min(1.0, relevance))
        actionable = relevance > 0.5
        cdata = ContextualizedData(
            tag=raw.tag, value=raw.value,
            sense=sense, frame=self.name,
            relevance=relevance, is_actionable=actionable)
        self._info_cache[raw.tag] = cdata
        # Mirror in data for resolution
        self.data[f"_cdata.{raw.tag}"] = cdata.to_dict()
        return cdata

    # ── Action discovery with harmony ranking ───────────────────────────────

    def get_available_actions(self, actor: Dict,
                               frame: Dict,
                               current_sense: SenseVector = None
                               ) -> List[Tuple[Action, HarmonyProfile]]:
        """
        Returns actions available to actor, filtered by:
          1. Actor filter (role-based)
          2. Guard evaluation: Adm(a) = ∏_{γ∈Γ_a} γ(E,c,Σ,x)
          3. Ranked by full harmony H(A,C,S) — not just gravity

        a* = argmax H(a,c,s,r)
        """
        result, seen = [], set()
        s = current_sense or self.basis

        ctx_cursor = self
        while ctx_cursor:
            for entry in ctx_cursor._actions:
                a, filt = entry["a"], entry["f"]
                if a.type in seen: continue
                if filt and not filt(actor): continue
                if not a.admissible(actor, self, frame): continue
                seen.add(a.type)
                h = compute_harmony(a.sense, self.basis, s,
                                    a.expected_rxn_sense, None, a.type)
                result.append((a, h))
            ctx_cursor = ctx_cursor.parent

        # a* = argmax H (sort by harmony score, descending)
        result.sort(key=lambda x: -x[1].score)
        return result

    def evaluate_circumstances(self, frame: Dict) -> List[Dict]:
        return [c.evaluate_with_trace(self, frame).to_dict()
                for c in self.circumstances]

    # ── Context topology: ω(c1,c2) = acos(cos(φ_C1,φ_C2)) / π ────────────

    def distance_to(self, other: "Context") -> float:
        return self.basis.angular_distance(other.basis)

    # ── Narrow: COW child context ──────────────────────────────────────────

    def narrow(self, name: str, kind: ContextKind = None,
               basis: SenseVector = None) -> "Context":
        child = Context(name, kind or self.kind,
                        basis or self.basis, self, self.depth + 1)
        child.data.update(self.data)
        return child

    # ── Serialization for MEP transport ───────────────────────────────────

    def to_envelope_dict(self, actor: Dict, frame: Dict,
                          sense: SenseVector = None) -> Dict:
        available = self.get_available_actions(actor, frame, sense)
        circs     = self.evaluate_circumstances(frame)
        return {
            "ctx_id": self.ctx_id, "name": self.name, "kind": self.kind.value,
            "depth": self.depth, "data": {k: v for k, v in self.data.items()
                                           if not k.startswith("_")},
            "circumstances": [{"id":c.circumstance.id,"desc":c.circumstance.description,
                                "holds":c.holds,"role":c.circumstance.role,
                                "blocker":c.blocker}
                               for c in [c.evaluate_with_trace(self, frame)
                                         for c in self.circumstances]],
            "available_actions": [a.to_dict(self, sense) for a, _ in available],
            "harmony_map": {a.type: h.to_dict() for a, h in available},
        }

    @property
    def valid_action_types(self) -> set:
        return {e["a"].type for e in self._actions}


# ═════════════════════════════════════════════════════════════════════════════
# ELEMENT  —  x ∈ X, operates in environment
# ═════════════════════════════════════════════════════════════════════════════

class ElementState(str, Enum):
    PENDING="pending"; ACTIVE="active"; SUSPENDED="suspended"; EJECTED="ejected"

class Element:
    """
    x ∈ X — an entity that discovers and executes actions via context.
    Elements do NOT own actions. Actions exist in the environment.

    Carries:
      • SenseVector φ_X (semantic fingerprint)
      • Property bag: identity (stable) / dynamic (mutable)
      • on_impacted(): receives reaction consequences
    """
    def __init__(self, name: str, element_type: str,
                 sense: SenseVector = None, impact_strength: float = 0.5):
        self.element_id   = str(uuid.uuid4())
        self.name         = name
        self.element_type = element_type
        self.sense        = sense or SenseVector.social(name)
        self.impact_strength = impact_strength
        self.state        = ElementState.PENDING
        self.admitted_at  = 0.0
        self._env: Optional["Environment"] = None
        # Dual property representation: strong-typed + bag for pattern surface
        self._stable:  Dict[str, Any] = {}  # immutable after construction
        self._dynamic: Dict[str, Any] = {}  # mutable state

    # ── Property discipline (P0 pattern rule) ─────────────────────────────

    def set_stable(self, k: str, v: Any):
        """Structural data — set once at construction, never mutated."""
        self._stable[k] = v

    def set_dynamic(self, k: str, v: Any):
        """Dynamic state — can change via reactions."""
        self._dynamic[k] = v

    def get(self, k: str, default: Any = None) -> Any:
        return self._dynamic.get(k, self._stable.get(k, default))

    @property
    def properties(self) -> Dict:
        return {**self._stable, **self._dynamic}

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def on_admitted(self, env: "Environment") -> None:
        self._env        = env
        self.admitted_at = time.time()
        self.state       = ElementState.ACTIVE

    async def on_impacted(self, reaction: Reaction, frame: Dict) -> None:
        pass  # override in domain subclasses

    async def on_evolved(self, tick: int) -> None:
        pass

    def to_dict(self) -> Dict:
        return {"element_id": self.element_id, "name": self.name,
                "element_type": self.element_type, "state": self.state.value,
                "sense_dim": self.sense.dimension,
                **{k: v for k, v in self._stable.items()}}


# ═════════════════════════════════════════════════════════════════════════════
# PHENOMENON ENGINE  —  p_t = DetectPhenomenon(V*, I*)
# Phenomena as attractors: k* = argmin_k d(φ(π_t), μ_k)
# ═════════════════════════════════════════════════════════════════════════════

class PhenomenonKind(str, Enum):
    RESOURCE_STARVATION="resource_starvation"; CASCADE_FAILURE="cascade_failure"
    OVERLOAD="overload"; DEADLOCK="deadlock"; OSCILLATION="oscillation"
    CONVERGENCE="convergence"; EMERGENCE="emergence"; CUSTOM="custom"

@dataclass
class Phenomenon:
    """Stable emergent pattern — an attractor in causal-semantic space."""
    name         : str
    kind         : PhenomenonKind
    magnitude    : float          # ∈ [0,1]
    count        : int
    window_s     : float
    context_name : str            = ""
    centroid     : Optional[SenseVector] = None  # μ_k — the attractor center
    detected_at  : float         = field(default_factory=time.time)
    dissolved_at : Optional[float] = None
    phenomenon_id: str           = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def is_active(self) -> bool: return self.dissolved_at is None

    def intensify(self, delta: float = 0.1):
        self.magnitude = min(1.0, self.magnitude + delta)

    def dissolve(self): self.dissolved_at = time.time()

    def to_dict(self) -> Dict:
        return {"name": self.name, "kind": self.kind.value, "magnitude": self.magnitude,
                "count": self.count, "active": self.is_active,
                "context": self.context_name}


class PhenomenonPattern:
    """
    Pattern matcher for emerging phenomena.
    φ(π_t) matched against attractors μ_k via:
      k* = argmin_k d(φ(π_t), μ_k)
    """
    def __init__(self, name: str, watch_type: str, threshold: int,
                 window_s: float, kind: PhenomenonKind = PhenomenonKind.CUSTOM,
                 attractor: SenseVector = None):
        self.name       = name
        self.watch_type = watch_type
        self.threshold  = threshold
        self.window_s   = window_s
        self.kind       = kind
        self.attractor  = attractor or SenseVector.emergent(name)  # μ_k

    def detect(self, history: List[Tuple[float, str, str]],
               context_name: str = "") -> Optional[Phenomenon]:
        """Detect if pattern crosses threshold in sliding window."""
        cutoff  = time.time() - self.window_s
        matches = [(t, s) for t, rt, s in history
                   if rt == self.watch_type and t >= cutoff]
        if len(matches) < self.threshold: return None
        magnitude = min(1.0, len(matches) / self.threshold)
        return Phenomenon(self.name, self.kind, magnitude, len(matches),
                          self.window_s, context_name, self.attractor)

    def distance_to_pattern(self, observation: SenseVector) -> float:
        """d(φ(π_t), μ_k) — angular distance to this pattern's attractor."""
        return observation.angular_distance(self.attractor)


# ═════════════════════════════════════════════════════════════════════════════
# CAUSAL GRAPH  —  G_t = G_{t-1} ⊕ (x_t → a_t → r_t → x'_t)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class CausalNode:
    """v_t = (t, x_t, c_t, a_t, r_t, Σ_t, Σ_{t+1})"""
    node_id       : str
    action_type   : str
    actor_id      : str
    context_name  : str
    status        : str
    summary       : str
    timestamp     : float
    depth         : int = 0
    causation_id  : str = ""
    reaction_sense: Optional[SenseVector] = None  # φ_R for harmony analysis


class CausalGraph:
    """
    M_{t+1} = M_t ∪ {v_t, r_t, i_t, p_t}
    Persistent genealogy with WHY/WHY-NOT explainability.
    """
    def __init__(self):
        self._nodes: Dict[str, CausalNode] = {}
        self._causal_deltas: List[SenseVector] = []  # Δ_t sequence

    def record(self, action_type: str, actor_id: str, context_name: str,
               status: str, summary: str, depth: int = 0,
               causation_id: str = "", reaction_sense: SenseVector = None) -> str:
        nid = str(uuid.uuid4())
        self._nodes[nid] = CausalNode(
            nid, action_type, actor_id, context_name,
            status, summary, time.time(), depth, causation_id, reaction_sense)
        return nid

    def record_delta(self, delta: SenseVector):
        """Record Δ_t = φ(Σ_{t+1}) - φ(Σ_t) for causal analysis."""
        self._causal_deltas.append(delta)

    def ancestry(self, node_id: str) -> List[CausalNode]:
        chain, cur = [], node_id
        while cur and (n := self._nodes.get(cur)):
            chain.insert(0, n); cur = n.causation_id or ""
        return chain

    def explain_why(self, action_type: str) -> str:
        ms = [n for n in self._nodes.values() if n.action_type == action_type]
        if not ms: return f"No record for '{action_type}'"
        chain = self.ancestry(ms[-1].node_id)
        return "\n".join(f"  [d={n.depth}] {n.action_type} ({n.status}): {n.summary}"
                         for n in chain)

    def average_dissonance(self) -> float:
        """Mean magnitude of causal deltas — system stability indicator."""
        if not self._causal_deltas: return 0.0
        return sum(d.magnitude for d in self._causal_deltas) / len(self._causal_deltas)

    @property
    def stats(self) -> Dict:
        return {"nodes": len(self._nodes),
                "ok": sum(1 for n in self._nodes.values() if n.status=="success"),
                "rej": sum(1 for n in self._nodes.values() if n.status=="rejected"),
                "avg_dissonance": round(self.average_dissonance(), 3),
                "types": list({n.action_type for n in self._nodes.values()})}


# ═════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT  E_t = (Σ_t, M_t, G_t, C_t)
# Central equation: E_{t+1} = 𝔘(E_t, 𝔯(x_t,c_t,Σ_t, 𝔄(x_t,c_t,Σ_t,Γ_t,Ψ(D_t,c_t,Σ_t))))
# ═════════════════════════════════════════════════════════════════════════════

class EnvironmentKind(str, Enum):
    STATIC="static"; REACTIVE="reactive"; DYNAMIC="dynamic"; LIVING="living"

class Environment:
    """
    The sovereign, passive container.
    E_t = (Σ_t, M_t, G_t, C_t)

    NOT an actor. A space where elements exist, circumstances govern,
    actions are discovered, and reactions transform state.

    Implements the central equation:
      E_{t+1} = 𝔘( E_t, 𝔯( x_t, c_t, Σ_t, 𝔄(x_t,c_t,Σ_t,Γ_t, Ψ(D_t,c_t,Σ_t)) ) )
    """

    def __init__(self, name: str, kind: EnvironmentKind = EnvironmentKind.REACTIVE):
        self.env_id   = str(uuid.uuid4())
        self.name     = name
        self.kind     = kind
        self.root_ctx = Context(name, ContextKind.GLOBAL,
                                SenseVector.technical("global frame", 0.3))
        self._elements   : Dict[str, Element] = {}
        self._contexts   : List[Context]      = []
        self._phenomena  : List[Phenomenon]   = []
        self._patterns   : List[PhenomenonPattern] = []
        self._causal     = CausalGraph()
        self._history    : List[Tuple[float, str, str]] = []  # (t, type, ctx)
        self._events     : List[Dict]         = []
        self._tick       : int = 0
        self._cbs: Dict[str, List[Callable]] = {
            "reaction":[], "event":[], "phenomenon":[], "interaction":[]}

    # ── 𝔄: action discovery ───────────────────────────────────────────────

    def discover_actions(self, element: Element, ctx: Context,
                          sense: SenseVector = None
                          ) -> List[Tuple[Action, HarmonyProfile]]:
        """
        𝔄(x,c,Σ,Γ,s) — return admissible actions ranked by H(A,C,S).
        a* = argmax_{a∈Avail} H(a,c,s,r)
        """
        frame = {"actor_id": element.element_id}
        return ctx.get_available_actions(element.to_dict(), frame, sense)

    # ── 𝔯: reaction generation  ───────────────────────────────────────────

    async def dispatch(self, actor: Element, action_type: str,
                       payload: Dict, ctx: Context,
                       correlation_id: str = "",
                       chain_depth: int = 0) -> Reaction:
        """
        𝔯(x,c,Σ,a,s) → R*
        Core dispatch implementing the central equation's reaction step.
        """
        if chain_depth >= 16:
            return Reaction(action_type, ReactionStatus.CHAIN_MAX,
                           "Maximum chain depth reached", impact=ImpactScope.none())

        corr = correlation_id or str(uuid.uuid4())
        frame = {"actor_id": actor.element_id, "payload": payload}

        for entry in ctx._actions:
            if entry["a"].type != action_type: continue
            action   = entry["a"]
            reaction = await action.execute(actor, payload, ctx, frame)
            reaction.correlation_id = corr
            reaction.chain_depth    = chain_depth

            # Record in causal graph G_t ⊕ (x→a→r→x')
            self._causal.record(
                action_type, actor.element_id, ctx.name,
                reaction.status.value, reaction.message or "",
                chain_depth, "", reaction.sense)

            # Record causal delta Δ_t
            if reaction.causal_delta:
                self._causal.record_delta(reaction.causal_delta)

            # History for phenomenon detection
            self._history.append((time.time(), reaction.status.value, ctx.name))
            self._emit("reaction.produced",
                       f"{action_type}→{reaction.status.value}")

            # 𝔘: route impact (Apply: E_t × R → E_{t+1})
            await self._apply(reaction, frame)

            # Detect phenomena p_t = DetectPhenomenon(V*, I*)
            self._detect_phenomena(ctx)

            # Notify callbacks
            for cb in self._cbs.get("reaction", []):
                try: cb(reaction)
                except: pass

            # Spawn chain
            if reaction.chain_actions and chain_depth < 15:
                for ct in reaction.chain_actions:
                    asyncio.get_event_loop().call_soon(
                        lambda t=ct: asyncio.create_task(
                            self.dispatch(actor, t, payload, ctx, corr, chain_depth+1)))

            return reaction

        return Reaction.reject(action_type,
            f"'{action_type}' not registered in '{ctx.name}'")

    # ── 𝔘: state update (Apply: E_t × R → E_{t+1}) ──────────────────────

    async def _apply(self, reaction: Reaction, frame: Dict):
        t = reaction.impact.target
        if t == ImpactTarget.ACTOR:
            eid = frame.get("actor_id")
            if el := self._elements.get(eid or ""):
                await el.on_impacted(reaction, frame)
        elif t == ImpactTarget.SPECIFIC and reaction.impact.element_id:
            if el := self._elements.get(reaction.impact.element_id):
                await el.on_impacted(reaction, frame)
        elif t == ImpactTarget.ALL:
            for el in self._elements.values():
                await el.on_impacted(reaction, frame)
        elif t == ImpactTarget.BROADCAST:
            for el in self._elements.values():
                if el.element_id != frame.get("actor_id"):
                    await el.on_impacted(reaction, frame)

    # ── Ψ: contextualize raw data ─────────────────────────────────────────

    def contextualize(self, raw: RawData, ctx: Context,
                       sense: SenseVector) -> ContextualizedData:
        """Ψ(D,c,Σ) → contextualized information."""
        return ctx.contextualize(raw, sense)

    # ── Phenomenon detection: argmin_k d(φ(π_t), μ_k) ──────────────────

    def _detect_phenomena(self, ctx: Context):
        for pattern in self._patterns:
            p = pattern.detect(self._history, ctx.name)
            if p and not any(ph.name==p.name for ph in self._phenomena[-5:] if ph.is_active):
                self._phenomena.append(p)
                self._emit("phenomenon.detected",
                            f"'{p.name}' mag={p.magnitude:.0%}")
                for cb in self._cbs.get("phenomenon", []):
                    try: cb(p)
                    except: pass

    def nearest_phenomenon_pattern(self,
                                    observation: SenseVector) -> Optional[PhenomenonPattern]:
        """k* = argmin_k d(φ(π_t), μ_k) — find nearest attractor."""
        if not self._patterns: return None
        return min(self._patterns, key=lambda p: p.distance_to_pattern(observation))

    # ── Element lifecycle ─────────────────────────────────────────────────

    async def admit(self, element: Element):
        if element.element_id in self._elements:
            raise ValueError(f"'{element.name}' already admitted")
        self._elements[element.element_id] = element
        await element.on_admitted(self)
        self._emit("element.admitted", f"'{element.name}'")

    async def eject(self, element_id: str):
        if el := self._elements.pop(element_id, None):
            el.state = ElementState.EJECTED
            self._emit("element.ejected", f"'{el.name}'")

    def get(self, eid: str) -> Optional[Element]:
        return self._elements.get(eid)

    def query(self, etype: str = None, pred: Callable = None) -> List[Element]:
        els = list(self._elements.values())
        if etype: els = [e for e in els if e.element_type == etype]
        if pred:  els = [e for e in els if pred(e)]
        return els

    # ── Context management ────────────────────────────────────────────────

    def create_context(self, name: str, kind: ContextKind = ContextKind.SEMANTIC,
                        basis: SenseVector = None, parent: Context = None,
                        circumstances: List[Circumstance] = None,
                        actions: List[Action] = None) -> Context:
        ctx = Context(name, kind, basis or SENSE_NULL,
                      parent or self.root_ctx, (parent.depth+1) if parent else 1)
        for c in (circumstances or []): ctx.add_circ(c)
        for a in (actions or []):       ctx.reg(a)
        self._contexts.append(ctx)
        return ctx

    def register_pattern(self, p: PhenomenonPattern): self._patterns.append(p)

    # ── Observable streams ────────────────────────────────────────────────

    def on_reaction(self, cb: Callable):   self._cbs["reaction"].append(cb)
    def on_phenomenon(self, cb: Callable): self._cbs["phenomenon"].append(cb)
    def on_event(self, cb: Callable):      self._cbs["event"].append(cb)

    def _emit(self, t: str, s: str):
        e = {"type": t, "summary": s, "at": time.time()}
        self._events.append(e)
        for cb in self._cbs.get("event", []):
            try: cb(e)
            except: pass

    # ── Evolution: E_t → E_{t+1} ─────────────────────────────────────────

    async def evolve(self) -> int:
        """Advance environment by one tick. Fires on_evolved on all elements."""
        self._tick += 1
        for el in self._elements.values():
            await el.on_evolved(self._tick)
        self._emit("environment.evolved", f"tick={self._tick}")
        return self._tick

    # ── WHY / WHY-NOT explainability ─────────────────────────────────────

    def explain_why(self, action_type: str) -> str:
        return self._causal.explain_why(action_type)

    def explain_why_not(self, element: Element, action: Action,
                         ctx: Context) -> List[str]:
        """Return WHY-NOT for a specific action: which guards block it?"""
        frame = {"actor_id": element.element_id}
        return [e.reason for e in action.why_not_admissible(element, ctx, frame)]

    # ── Snapshot: ISituation ─────────────────────────────────────────────

    def snapshot(self) -> Dict:
        active_p = [p for p in self._phenomena if p.is_active]
        kind = ("critical" if any(p.magnitude > 0.8 for p in active_p) else
                "degraded"  if active_p else "operational")
        return {"env_id": self.env_id, "name": self.name, "tick": self._tick,
                "elements": len(self._elements), "contexts": len(self._contexts),
                "phenomena": len(active_p), "situation": kind,
                "causal": self._causal.stats, "at": time.time()}

    @property
    def elements(self): return self._elements
    @property
    def phenomena(self): return self._phenomena
    @property
    def recent_events(self): return self._events[-20:]
    @property
    def causal(self): return self._causal


# ═════════════════════════════════════════════════════════════════════════════
# USAGE EXAMPLE — eSchool domain
# ═════════════════════════════════════════════════════════════════════════════

async def demo():
    """Quick demonstration of all EDP innovations."""

    # Build environment
    env = Environment("ESchool", EnvironmentKind.REACTIVE)

    # Circumstances (algebraic)
    C_SYS  = Circumstance.always("system.active")
    C_ENRL = Circumstance.flag("enrollment.open","Enrollment active","enrollmentOpen")
    C_ROLE_ADMIN = Circumstance.when("actor.is.admin","Actor is Admin",
        lambda ctx, frame: any(e.get("element_type")=="Admin"
                               for e in ctx.elements
                               if e.get("element_id")==frame.get("actor_id")))
    # Compose: admin OR system AND enrollment
    C_COMPOUND = C_ROLE_ADMIN | (C_SYS & C_ENRL)

    # Actions with semantic vectors
    async def enroll_handler(actor, payload, ctx, frame):
        sid = payload.get("studentId","")
        cid = payload.get("courseId","")
        delta = SenseVector.temporal("enrollment state change", 0.3)
        return Reaction.ok("student.enroll",
            f"Enrolled student {sid} in {cid}",
            result={"studentId":sid,"courseId":cid},
            sense=SenseVector.temporal("enrollment confirmed"),
            impact=ImpactScope.on_element(cid, 0.8),
            chain=["notification.dispatch"] if not cid else [])

    enroll_action = Action(
        "student.enroll", ActionCategory.COMMAND, "Enroll student in course",
        sense=SenseVector.temporal("enrollment operation", 0.95),
        guards=[C_ENRL],
        handler=enroll_handler,
        expected_reaction_sense=SenseVector.temporal("enrollment confirmed"))

    snapshot_action = Action(
        "system.snapshot", ActionCategory.QUERY, "Snapshot environment",
        sense=SenseVector.technical("system state", 0.3),
        handler=lambda a,p,c,f: asyncio.coroutine(
            lambda: Reaction.ok("system.snapshot","OK",env.snapshot()))())

    # Build context with basis in normative space
    ctx = env.create_context("Academic", ContextKind.SEMANTIC,
        basis=SenseVector.normative("academic operations", 0.9),
        circumstances=[C_SYS, C_ENRL],
        actions=[enroll_action, snapshot_action])
    ctx.set("enrollmentOpen", True)

    # Phenomenon patterns (attractors)
    env.register_pattern(PhenomenonPattern(
        "EnrollmentSurge", "success", threshold=3, window_s=60,
        kind=PhenomenonKind.OVERLOAD,
        attractor=SenseVector.emergent("enrollment overload")))

    # Admit elements
    admin = Element("Dr. Vasquez", "Admin", SenseVector.normative("admin role", 0.9))
    student = Element("Alice", "Student", SenseVector.social("student", 0.5))
    await env.admit(admin)
    await env.admit(student)
    ctx.include(admin.to_dict())

    # Subscribe
    env.on_reaction(lambda r: print(f"  [RXN] {r.line()}"))
    env.on_phenomenon(lambda p: print(f"  [PHN] {p.name} mag={p.magnitude:.0%}"))

    print("\n─ EDP v4.1 Demo ─────────────────────────────────────────────")

    # 1. Discover actions with HARMONY ranking
    available = env.discover_actions(admin, ctx, SenseVector.temporal("now"))
    print(f"\n  Available actions (ranked by H(A,C,S)):")
    for a, h in available:
        print(f"    {h.score:+.3f}  {a.type:30s} "
              f"ctx={h.context_alignment:.2f} sem={h.semantic_alignment:.2f}")

    # 2. Context topology
    ctx2 = env.create_context("Financial", ContextKind.SEMANTIC,
        basis=SenseVector.financial("financial ops", 0.9))
    dist = ctx.distance_to(ctx2)
    print(f"\n  Context distance Academic↔Financial: {dist:.3f} "
          f"(0=identical, 1=orthogonal)")

    # 3. WHY-NOT query (enrollment closed)
    ctx.set("enrollmentOpen", False)
    blocked = env.explain_why_not(student, enroll_action, ctx)
    print(f"\n  WHY-NOT student.enroll: {blocked}")

    # 4. Execute (enrollment open)
    ctx.set("enrollmentOpen", True)
    reaction = await env.dispatch(admin, "student.enroll",
        {"studentId": student.element_id, "courseId": "math-101"}, ctx)

    # 5. Nearest phenomenon attractor
    obs = SenseVector.emergent("overload signal", 0.8)
    nearest = env.nearest_phenomenon_pattern(obs)
    if nearest:
        print(f"\n  Nearest phenomenon attractor: {nearest.name} "
              f"dist={nearest.distance_to_pattern(obs):.3f}")

    # 6. Snapshot
    snap = env.snapshot()
    print(f"\n  Snapshot: {snap}")
    print(f"  Causal stats: {env.causal.stats}")
    print("─────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    asyncio.run(demo())
