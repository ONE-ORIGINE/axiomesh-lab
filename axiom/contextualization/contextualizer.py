"""
contextualizer.py  —  Standalone Contextualizer Component
════════════════════════════════════════════════════════════════════════════
OneOrigine / ImperialSchool Research  —  I.S. License

THE CONTEXTUALIZER
──────────────────
The Contextualizer is the component that transforms raw data into
actionable, contextualized information. It is the implementation of:

  Ψ : IData × IContext × IΣ → IContextualizedData

In natural language:
  • A temperature of 38°C means nothing alone.
  • In a medical context:  → "fever — action required" (relevance=0.9, actionable)
  • In a baking context:   → "normal oven temp" (relevance=0.4, informational)
  • In a drone context:    → "motor temp warning" (relevance=0.7, actionable)

The same datum. Different frames. Different meaning. Different actions.

This is why the Contextualizer exists as a standalone component:
  → It can be reused across ANY EDP environment
  → It can be extended with domain-specific rules
  → It converts the raw world into the structured world

COMPONENTS:
  • DataSignal:       typed raw input datum
  • SignalProfile:    semantic fingerprint of a signal class
  • ContextualRule:   mapping (signal_type, context_kind) → interpretation
  • Contextualizer:   the engine that applies rules and computes Ψ
  • ContextMatrix:    M_C ∈ ℝ^(C×D) — how contexts weight semantic dimensions
"""

from __future__ import annotations

import math, time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from edp import SenseVector, Context, ContextKind, SENSE_NULL

__version__ = "1.0.0"
__author__  = "OneOrigine"
__license__ = "I.S."


# ─── Data Signal ──────────────────────────────────────────────────────────────

@dataclass
class DataSignal:
    """
    A typed raw input datum before contextual interpretation.
    Carries semantic hints but no meaning until contextualized.
    """
    tag       : str              # e.g. "temperature", "battery_pct", "velocity"
    value     : Any              # the raw measurement
    unit      : str = ""         # e.g. "°C", "%", "m/s"
    source    : str = "sensor"   # origin of measurement
    captured  : float = field(default_factory=time.time)
    signal_type: str = "scalar"  # scalar | vector | categorical | boolean

    def numeric(self, default: float = 0.0) -> float:
        try: return float(self.value)
        except: return default

    def __repr__(self):
        return f"Signal({self.tag}={self.value}{self.unit} @{self.source})"


# ─── Signal Profile ───────────────────────────────────────────────────────────

@dataclass
class SignalProfile:
    """
    Semantic fingerprint of a signal class.
    Defines how a type of signal maps to the semantic space.

    Example:
      "temperature" → primarily normative+causal (it matters, it causes things)
      "position"    → primarily spatial
      "battery"     → primarily temporal (time-to-empty) + causal
    """
    signal_tag  : str
    base_sense  : SenseVector
    unit        : str         = ""
    min_val     : float       = 0.0
    max_val     : float       = 100.0
    thresholds  : Dict[str, float] = field(default_factory=dict)
    # e.g. {"warning": 0.25, "critical": 0.10, "optimal": 0.80}

    def normalize(self, value: float) -> float:
        """Normalize raw value to [0,1] within this signal's range."""
        rng = self.max_val - self.min_val
        return max(0., min(1., (value - self.min_val) / rng)) if rng > 0 else 0.


# ─── Contextualization Rule ───────────────────────────────────────────────────

@dataclass
class ContextualRule:
    """
    A mapping rule: (signal_type, context_kind) → interpretation.

    When the Contextualizer encounters a temperature signal in a medical context,
    it applies the medical temperature rule (not the baking rule).
    """
    signal_tag    : str
    context_kind  : Optional[ContextKind]   # None = applies to all contexts
    sense_fn      : Callable[[DataSignal, Context], SenseVector]
    relevance_fn  : Callable[[DataSignal, Context], float]
    label_fn      : Callable[[DataSignal, Context], str]
    actionable_fn : Callable[[DataSignal, Context, float], bool]
    priority      : int = 0   # higher priority rules win

    def matches(self, signal_tag: str, ctx_kind: ContextKind) -> bool:
        tag_ok = (self.signal_tag == signal_tag or self.signal_tag == "*")
        ctx_ok = (self.context_kind is None or self.context_kind == ctx_kind)
        return tag_ok and ctx_ok


# ─── Contextualized Signal ────────────────────────────────────────────────────

@dataclass
class ContextualizedSignal:
    """
    Ψ(D,c,Σ) — the result: a raw signal transformed into meaningful information.

    Contains:
    • The original signal
    • The semantic vector (where in meaning-space this datum sits)
    • The relevance (how important in this context)
    • A human-readable label (e.g. "CRITICAL: battery at 8%")
    • Whether it gates actions (is_actionable)
    • The context frame that produced this interpretation
    """
    signal      : DataSignal
    sense       : SenseVector
    relevance   : float       # [0,1] — how important in this context
    label       : str         # human-readable interpretation
    is_actionable: bool       # should this influence available actions?
    context_name : str
    context_kind : str
    produced_at  : float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "tag":         self.signal.tag,
            "value":       self.signal.value,
            "unit":        self.signal.unit,
            "label":       self.label,
            "sense_dim":   self.sense.dimension,
            "relevance":   round(self.relevance, 3),
            "actionable":  self.is_actionable,
            "context":     self.context_name,
        }

    def __repr__(self):
        a = "⚡" if self.is_actionable else "·"
        return f"[{a} {self.label}  rel={self.relevance:.2f}  φ({self.sense.dimension})]"


# ─── Context Matrix M_C ───────────────────────────────────────────────────────

class ContextMatrix:
    """
    M_C ∈ ℝ^(C×D) — how contexts weight semantic dimensions.

    Row c = context type
    Col d = semantic dimension
    M_C[c][d] = how much dimension d matters in context c

    Usage:
      weighted_sense = M_C[c] ⊙ signal_sense  (Hadamard product)
      → adjusts the signal's semantic vector by context importance weights

    This makes the Contextualizer context-aware at a mathematical level,
    not just via lookup rules.
    """

    # Default weights: M_C[context_kind][semantic_dim]
    DEFAULT = {
        ContextKind.GLOBAL:       [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        ContextKind.TEMPORAL:     [0.3, 1.0, 0.2, 0.4, 0.3, 0.3, 0.4, 0.5],
        ContextKind.SPATIAL:      [0.4, 0.3, 1.0, 0.3, 0.4, 0.2, 0.5, 0.4],
        ContextKind.SEMANTIC:     [0.6, 0.4, 0.3, 0.8, 0.6, 0.5, 0.6, 0.7],
        ContextKind.RELATIONAL:   [0.4, 0.4, 0.3, 0.5, 0.9, 0.4, 0.4, 0.5],
        ContextKind.TRANSACTIONAL:[0.5, 0.6, 0.2, 0.6, 0.4, 0.9, 0.5, 0.4],
        ContextKind.CAUSAL:       [1.0, 0.6, 0.4, 0.5, 0.4, 0.3, 0.5, 0.7],
        ContextKind.COMPOSITE:    [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
        ContextKind.OBSERVATION:  [0.5, 0.4, 0.6, 0.4, 0.5, 0.3, 0.8, 0.9],
        ContextKind.GOVERNANCE:   [0.4, 0.5, 0.3, 1.0, 0.6, 0.7, 0.5, 0.4],
    }

    def __init__(self, custom: Dict[ContextKind, List[float]] = None):
        self._matrix = dict(self.DEFAULT)
        if custom: self._matrix.update(custom)

    def weights(self, kind: ContextKind) -> List[float]:
        return self._matrix.get(kind, self.DEFAULT[ContextKind.GLOBAL])

    def apply(self, sense: SenseVector, ctx_kind: ContextKind) -> SenseVector:
        """
        weighted = M_C[ctx_kind] ⊙ sense.vector  (element-wise product)
        Adjusts sense vector importance by context dimension weights.
        """
        w    = self.weights(ctx_kind)
        wvec = tuple(s * wi for s, wi in zip(sense.v, w))
        mag  = math.sqrt(sum(x*x for x in wvec)) or 1.0
        norm = tuple(x/mag for x in wvec)
        return SenseVector(sense.dimension, sense.meaning, sense.magnitude, norm)

    def to_matrix(self) -> List[List[float]]:
        """Export M_C as a 2D list for inspection or ML input."""
        return [self.weights(k) for k in sorted(self._matrix.keys(), key=lambda k: k.value)]


# ─── Contextualizer ───────────────────────────────────────────────────────────

class Contextualizer:
    """
    The Ψ engine: transforms DataSignals into ContextualizedSignals.

      Ψ(signal, context, Σ) → ContextualizedSignal

    Standalone component — can be used with or without a full EDP environment.
    Ships with:
      • Built-in rules for common signals (temperature, battery, position, velocity)
      • A ContextMatrix M_C for dimension weighting
      • Extensible via add_rule()

    Example:
      cx = Contextualizer()
      ctx = Context("Medical", ContextKind.SEMANTIC, ...)
      result = cx.process(DataSignal("temperature", 38.5, "°C"), ctx)
      # → ContextualizedSignal: "⚡ HIGH TEMP: 38.5°C — fever threshold exceeded"
    """

    def __init__(self, ctx_matrix: ContextMatrix = None):
        self._rules: List[ContextualRule] = []
        self._profiles: Dict[str, SignalProfile] = {}
        self._matrix  = ctx_matrix or ContextMatrix()
        self._history: List[ContextualizedSignal] = []
        self._register_builtins()

    def add_profile(self, profile: SignalProfile) -> "Contextualizer":
        self._profiles[profile.signal_tag] = profile
        return self

    def add_rule(self, rule: ContextualRule) -> "Contextualizer":
        self._rules.append(rule)
        self._rules.sort(key=lambda r: -r.priority)
        return self

    def process(self, signal: DataSignal, ctx: Context,
                 sigma: Dict = None) -> ContextualizedSignal:
        """
        Ψ(D, c, Σ) → ContextualizedSignal

        Applies matching rules in priority order.
        Falls back to default interpretation if no rule matches.
        """
        sigma = sigma or {}
        # Find best matching rule
        best = next(
            (r for r in self._rules if r.matches(signal.tag, ctx.kind)), None)

        if best:
            sense     = best.sense_fn(signal, ctx)
            relevance = best.relevance_fn(signal, ctx)
            label     = best.label_fn(signal, ctx)
            actionable = best.actionable_fn(signal, ctx, relevance)
        else:
            sense, relevance, label, actionable = self._default(signal, ctx)

        # Apply context matrix weighting: M_C[ctx.kind] ⊙ sense
        sense = self._matrix.apply(sense, ctx.kind)

        cdata = ContextualizedSignal(signal, sense, relevance, label,
                                      actionable, ctx.name, ctx.kind.value)
        self._history.append(cdata)
        return cdata

    def process_batch(self, signals: List[DataSignal],
                       ctx: Context) -> List[ContextualizedSignal]:
        """Process multiple signals in one context."""
        return [self.process(s, ctx) for s in signals]

    def actionable_signals(self, ctx: Context,
                            signals: List[DataSignal]) -> List[ContextualizedSignal]:
        """Return only signals that should influence action availability."""
        return [r for r in self.process_batch(signals, ctx) if r.is_actionable]

    @property
    def history(self) -> List[ContextualizedSignal]: return self._history

    def context_matrix_export(self) -> Dict:
        """Export M_C as structured data for inspection."""
        dims = ["causal","temporal","spatial","normative","social","financial","technical","emergent"]
        return {
            "matrix": self._matrix.to_matrix(),
            "rows": [k.value for k in sorted(self._matrix._matrix.keys(), key=lambda k: k.value)],
            "cols": dims,
            "description": "M_C[context][semantic_dim] = dimension weight in context"
        }

    def _default(self, signal: DataSignal, ctx: Context):
        val       = signal.numeric()
        profile   = self._profiles.get(signal.tag)
        norm_val  = profile.normalize(val) if profile else max(0., min(1., val/100.))
        base_sense = profile.base_sense if profile else SenseVector.technical(signal.tag, norm_val)
        relevance  = norm_val * ctx.basis.cosine(base_sense)
        relevance  = max(0., min(1., relevance))
        label      = f"{signal.tag}: {signal.value}{signal.unit}"
        return base_sense, relevance, label, (relevance > 0.6)

    # ── Built-in rules ────────────────────────────────────────────────────

    def _register_builtins(self):
        """Register built-in rules for common signal types."""

        # ── Battery ──────────────────────────────────────────────────────
        self.add_profile(SignalProfile(
            "battery", SenseVector.temporal("battery state", 1.0), "%", 0., 100.,
            {"critical":15., "low":25., "optimal":80.}))

        self.add_rule(ContextualRule(
            "*", None,  # battery applies in all contexts
            sense_fn     = lambda s, c: SenseVector.temporal("battery level", s.numeric()/100),
            relevance_fn = lambda s, c: max(0., 1.0 - s.numeric()/100),  # more critical as lower
            label_fn     = lambda s, c: (
                f"🔴 CRITICAL BATTERY: {s.value}%" if s.numeric()<15 else
                f"🟡 LOW BATTERY: {s.value}%" if s.numeric()<25 else
                f"🟢 Battery: {s.value}%"),
            actionable_fn = lambda s, c, r: s.numeric() < 25.0,
            priority      = 10
        ) if False else ContextualRule(
            "battery", None,
            sense_fn     = lambda s, c: SenseVector.temporal("battery level", s.numeric()/100),
            relevance_fn = lambda s, c: max(0.2, 1.0 - s.numeric()/100),
            label_fn     = lambda s, c: (
                f"CRITICAL BATTERY: {s.value}%" if s.numeric()<15 else
                f"LOW BATTERY: {s.value}%" if s.numeric()<25 else
                f"Battery: {s.value}%"),
            actionable_fn = lambda s, c, r: s.numeric() < 25.0,
            priority      = 10))

        # ── Temperature ───────────────────────────────────────────────────
        self.add_rule(ContextualRule(
            "temperature", ContextKind.SEMANTIC,  # medical/general
            sense_fn     = lambda s, c: SenseVector.normative("thermal state",
                                         min(1., abs(s.numeric()-37)/10)),
            relevance_fn = lambda s, c: min(1., abs(s.numeric()-37)/5),
            label_fn     = lambda s, c: (
                f"FEVER: {s.value}°C" if s.numeric()>38 else
                f"HYPOTHERMIA: {s.value}°C" if s.numeric()<35 else
                f"Normal temp: {s.value}°C"),
            actionable_fn = lambda s, c, r: s.numeric() > 38 or s.numeric() < 35,
            priority      = 5))

        self.add_rule(ContextualRule(
            "temperature", ContextKind.SPATIAL,  # drone/motor temp
            sense_fn     = lambda s, c: SenseVector.technical("motor thermal", s.numeric()/100),
            relevance_fn = lambda s, c: min(1., s.numeric()/80),
            label_fn     = lambda s, c: (
                f"MOTOR OVERHEAT: {s.value}°C" if s.numeric()>75 else
                f"Motor temp: {s.value}°C"),
            actionable_fn = lambda s, c, r: s.numeric() > 70,
            priority      = 5))

        # ── Position ──────────────────────────────────────────────────────
        self.add_rule(ContextualRule(
            "position", None,
            sense_fn     = lambda s, c: SenseVector.spatial("spatial position", 0.9),
            relevance_fn = lambda s, c: 0.9,  # positions are always relevant
            label_fn     = lambda s, c: f"Position: {s.value}{s.unit}",
            actionable_fn = lambda s, c, r: True,
            priority      = 3))

        # ── Velocity ──────────────────────────────────────────────────────
        self.add_rule(ContextualRule(
            "velocity", None,
            sense_fn     = lambda s, c: SenseVector.temporal("kinematic velocity",
                                         min(1., s.numeric()/20)),
            relevance_fn = lambda s, c: min(1., s.numeric()/10),
            label_fn     = lambda s, c: f"Velocity: {s.value}{s.unit}",
            actionable_fn = lambda s, c, r: s.numeric() > 15,  # high speed
            priority      = 3))

        # ── Error / alarm ─────────────────────────────────────────────────
        self.add_rule(ContextualRule(
            "error", None,
            sense_fn     = lambda s, c: SenseVector.causal("error signal", 0.95),
            relevance_fn = lambda s, c: 1.0,
            label_fn     = lambda s, c: f"ERROR: {s.value}",
            actionable_fn = lambda s, c, r: True,
            priority      = 20))  # highest priority

        # ── Generic boolean ───────────────────────────────────────────────
        self.add_rule(ContextualRule(
            "*", None,   # catchall
            sense_fn     = lambda s, c: SenseVector.technical(str(s.tag), 0.4),
            relevance_fn = lambda s, c: 0.3,
            label_fn     = lambda s, c: f"{s.tag}: {s.value}{s.unit}",
            actionable_fn = lambda s, c, r: False,
            priority      = 0))


# ─── Demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from edp import Context, ContextKind, SenseVector

    cx = Contextualizer()
    print("── Contextualizer Demo ──────────────────────────────────────────\n")

    medical_ctx = Context("Medical.Ward", ContextKind.SEMANTIC,
                           basis=SenseVector.normative("medical", 0.9))
    drone_ctx   = Context("Drone.Flight", ContextKind.SPATIAL,
                           basis=SenseVector.spatial("drone ops", 0.95))
    finance_ctx = Context("Trading", ContextKind.TRANSACTIONAL,
                           basis=SenseVector.financial("trading", 0.9))

    signals = [
        DataSignal("temperature", 39.2, "°C", "thermometer"),
        DataSignal("battery",     12.0, "%",  "sensor"),
        DataSignal("temperature", 82.0, "°C", "motor_sensor"),
        DataSignal("error",       "GPS_LOCK_LOST", "", "gps_module"),
    ]

    contexts = [
        (medical_ctx, signals[0]),
        (medical_ctx, signals[1]),
        (drone_ctx,   signals[2]),
        (drone_ctx,   signals[3]),
    ]

    for ctx, sig in contexts:
        result = cx.process(sig, ctx)
        print(f"  {result}")

    print(f"\n  Context Matrix M_C shape: {len(cx.context_matrix_export()['rows'])}×8")
    print(f"  Actionable signals processed: {sum(1 for h in cx.history if h.is_actionable)}/{len(cx.history)}")
