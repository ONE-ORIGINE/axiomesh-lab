"""
savoir.py  —  SAVOIR: Situated Assertive Vector of Irrefutable Observations and Reasoning
════════════════════════════════════════════════════════════════════════════════════════
OneOrigine / ImperialSchool Research  —  I.S. License

THE PROBLEM SAVOIR SOLVES
──────────────────────────
LLMs probabilize everything — including certainties.

When a robot arm moves a cup from A to B:
  • The robot KNOWS the cup is now at B (sensor-verified)
  • An LLM asked "where is the cup?" will estimate: P(B|context) ~ 0.87
    It recalculates from scratch. It may say 0.87 when it should say 1.0.

This is not a failure of intelligence. It is a failure of architecture.
Statistical models have no distinction between:
  - "I computed this" (certainty)
  - "I estimated this" (probability)

SAVOIR is the layer that makes this distinction explicit.

THE SAVOIR CONCEPT
──────────────────
SAVOIR sits between the physical/logical world and the EDP environment.
It manages a certainty tensor: C[fact][source][confidence]

  confidence = 1.0  →  KNOWN (sensor-verified, physically confirmed)
  confidence ∈ (0.5, 1.0)  →  PROBABLE (inferred, estimated)
  confidence ≤ 0.5  →  UNCERTAIN (hypothesis, low confidence)

When the EDP/MEP asks "what is the state?", SAVOIR provides:
  • Verified facts at certainty 1.0 → these gate hard circumstances
  • Estimated facts at confidence < 1.0 → these gate soft circumstances
  • The LLM receives the SAVOIR snapshot → it knows what IS certain

MATHEMATICAL STRUCTURE
───────────────────────
Environmental state matrix:
  M_Σ ∈ ℝ^(N×D)  where N=elements, D=property dimensions
  M_Σ[i][j] = (value, certainty) — the known state of element i, property j

Context matrix (how contexts relate to facts):
  M_C ∈ ℝ^(C×D)  where C=contexts, D=semantic dimensions
  M_C[c][d] = relevance of semantic dimension d in context c

Reaction transition matrix:
  M_R ∈ ℝ^(A×S)  where A=action types, S=state dimensions
  M_R[a][s] = how action a affects state dimension s

Environmental embedding:
  φ_env = flatten(M_Σ) ∈ ℝ^(N*D)
  This IS the persistent memory vector the AI can query.

THE KEY INSIGHT
───────────────
EDP gives structure: context, circumstance, action, reaction.
SAVOIR gives certainty: what is KNOWN vs what is ESTIMATED.
Together: an AI that does not confuse "I predicted X" with "X is verified."

This is the bridge toward a robot that KNOWS where objects are,
not one that estimates their position 30 times per second.
"""

from __future__ import annotations

import math, time, uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from edp import SenseVector, SENSE_NULL

__version__ = "1.0.0"
__author__  = "OneOrigine"
__license__ = "I.S."

# ─── Certainty Levels ─────────────────────────────────────────────────────────

class CertaintyLevel(float, Enum):
    """
    Not a probability. A classification of epistemic status.

    KNOWN (1.0):      Sensor-verified, physically confirmed, logically certain.
    VERIFIED (0.95):  Multi-source confirmed, high-trust inference.
    PROBABLE (0.75):  Strong inference, single reliable source.
    ESTIMATED (0.50): Computed estimate, moderate confidence.
    UNCERTAIN (0.25): Low-confidence guess, needs verification.
    UNKNOWN (0.0):    No information — should not gate hard constraints.
    """
    KNOWN     = 1.00
    VERIFIED  = 0.95
    PROBABLE  = 0.75
    ESTIMATED = 0.50
    UNCERTAIN = 0.25
    UNKNOWN   = 0.00

    def __str__(self):
        return {1.00:"KNOWN",0.95:"VERIFIED",0.75:"PROBABLE",
                0.50:"ESTIMATED",0.25:"UNCERTAIN",0.00:"UNKNOWN"}.get(self.value,"?")

    @property
    def is_certain(self)  -> bool: return self.value >= 0.95
    @property
    def is_probable(self) -> bool: return 0.50 <= self.value < 0.95
    @property
    def is_uncertain(self)-> bool: return self.value < 0.50


# ─── Fact ─────────────────────────────────────────────────────────────────────

@dataclass
class Fact:
    """
    A single piece of knowledge with certainty.

    NOT a probability distribution. A verified or estimated assertion.

    key:       unique identifier (e.g. "drone.position", "object.A.location")
    value:     the actual value (position, temperature, boolean, etc.)
    certainty: epistemic confidence level
    source:    sensor/system that produced this fact
    sense:     semantic vector — enables EDP integration
    valid_until: expiry time (ms since epoch), 0 = permanent
    """
    key        : str
    value      : Any
    certainty  : CertaintyLevel
    source     : str              = "unknown"
    sense      : SenseVector      = field(default_factory=lambda: SENSE_NULL)
    produced_at: float            = field(default_factory=time.time)
    valid_until: float            = 0.0    # 0 = no expiry
    fact_id    : str              = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def is_valid(self) -> bool:
        if self.valid_until == 0: return True
        return time.time() < self.valid_until

    @property
    def is_certain(self)  -> bool: return self.certainty.is_certain
    @property
    def is_probable(self) -> bool: return self.certainty.is_probable

    @property
    def age_ms(self) -> float:
        return (time.time() - self.produced_at) * 1000

    def degrade(self, factor: float = 0.95) -> "Fact":
        """
        Facts can degrade over time if not refreshed.
        A position fact 30 seconds old is less certain than a fresh one.
        """
        if self.certainty.is_certain:
            new_c = max(CertaintyLevel.ESTIMATED,
                        CertaintyLevel(self.certainty.value * factor))
            return Fact(self.key, self.value, new_c, self.source, self.sense,
                        self.produced_at, self.valid_until, self.fact_id)
        return self

    def __repr__(self):
        return f"Fact({self.key}={self.value!r} [{self.certainty}] @{self.source})"


# ─── Environmental State Matrix M_Σ ──────────────────────────────────────────

class EnvironmentalStateMatrix:
    """
    M_Σ ∈ ℝ^(N×D) — the numerical representation of environment state.

    Rows: elements (drone, object_A, object_B, ...)
    Cols: semantic dimensions (position_x, position_y, position_z,
                                velocity, battery, certainty, ...)

    Flattening M_Σ → φ_env ∈ ℝ^(N*D) gives the environment embedding.
    This is the persistent memory vector the AI can query — not a text prompt.

    This is the architectural answer to: "How does a robot KNOW where things are?"
    """

    def __init__(self, element_ids: List[str],
                 property_dims: List[str]):
        self.element_ids   = element_ids
        self.property_dims = property_dims
        self.N = len(element_ids)
        self.D = len(property_dims)
        # M_Σ[i][j] = (value, certainty)
        self._matrix: Dict[str, Dict[str, Tuple[float, float]]] = {
            eid: {dim: (0.0, 0.0) for dim in property_dims}
            for eid in element_ids
        }
        self._updated_at: float = time.time()

    def set(self, element_id: str, prop: str,
            value: float, certainty: float = 1.0):
        if element_id not in self._matrix:
            self._matrix[element_id] = {dim:(0.,0.) for dim in self.property_dims}
            self.element_ids.append(element_id)
            self.N += 1
        self._matrix[element_id][prop] = (value, certainty)
        self._updated_at = time.time()

    def get(self, element_id: str, prop: str) -> Tuple[float, float]:
        """Returns (value, certainty)."""
        return self._matrix.get(element_id, {}).get(prop, (0.0, 0.0))

    def certainty_of(self, element_id: str, prop: str) -> float:
        return self._matrix.get(element_id, {}).get(prop, (0., 0.))[1]

    def flatten(self) -> List[float]:
        """
        φ_env = flatten(M_Σ) — the environment embedding vector.
        Concatenates [value, certainty] pairs for all (element, property).
        """
        result = []
        for eid in self.element_ids:
            for dim in self.property_dims:
                v, c = self._matrix.get(eid, {}).get(dim, (0., 0.))
                result.extend([v, c])
        return result

    def to_value_vector(self, element_id: str) -> List[float]:
        """φ_element ∈ ℝ^D — the state vector of one element."""
        return [self._matrix.get(element_id, {}).get(d, (0.,0.))[0]
                for d in self.property_dims]

    def certainty_vector(self, element_id: str) -> List[float]:
        """Certainty vector for one element — separates knowledge from estimates."""
        return [self._matrix.get(element_id, {}).get(d, (0.,0.))[1]
                for d in self.property_dims]

    def average_certainty(self) -> float:
        vals = [c for row in self._matrix.values() for _,c in row.values()]
        return sum(vals)/len(vals) if vals else 0.

    def __repr__(self):
        return (f"StateMatrix({self.N}×{self.D}"
                f" avg_certainty={self.average_certainty():.2f})")


# ─── Reaction Transition Matrix M_R ──────────────────────────────────────────

class ReactionTransitionMatrix:
    """
    M_R ∈ ℝ^(A×S) — how actions affect state dimensions.

    M_R[action_type][state_dim] = expected_delta

    Usage:
      When action "drone.move" executes, M_R tells SAVOIR:
        - position_x expected to change by +payload["dx"]
        - battery expected to change by -0.01
        - velocity expected to be payload["speed"]

    This allows SAVOIR to UPDATE state with certainty after action,
    rather than re-estimating from scratch.
    """

    def __init__(self):
        self._matrix: Dict[str, Dict[str, float]] = {}

    def register(self, action_type: str, effects: Dict[str, float]):
        """
        Register expected state transitions for an action.
        effects: {state_dim: expected_delta}
        """
        self._matrix[action_type] = effects

    def expected_effects(self, action_type: str) -> Dict[str, float]:
        return self._matrix.get(action_type, {})

    def apply(self, state: EnvironmentalStateMatrix, action_type: str,
               element_id: str, payload: Dict, certainty: float = 1.0):
        """
        Apply expected state transition after action.
        Sets state values with certainty = action_certainty
        (not re-estimated — physically applied).
        """
        effects = self.expected_effects(action_type)
        for dim, delta in effects.items():
            current_val, _ = state.get(element_id, dim)
            # For move actions, delta comes from payload if available
            actual_delta = payload.get(f"d_{dim}", delta)
            new_val = current_val + actual_delta
            # Certainty: if action was physically executed, we KNOW the effect
            state.set(element_id, dim, new_val, certainty)


# ─── SAVOIR Knowledge Base ────────────────────────────────────────────────────

class Savoir:
    """
    SAVOIR: Situated Assertive Vector of Irrefutable Observations and Reasoning.

    The certainty layer over the EDP environment.
    Manages:
      1. Fact store — verified and estimated assertions
      2. Environmental state matrix M_Σ (the numerical env state)
      3. Reaction transition matrix M_R (expected state changes)
      4. Certainty-gated queries — "is X certain?"
      5. Snapshot for MEP/LLM consumption

    The LLM/agent receives a SAVOIR snapshot, not raw environment state.
    It can distinguish KNOWN facts from ESTIMATES.
    It cannot confuse "I predicted X" with "X is verified."
    """

    def __init__(self, element_ids: List[str] = None,
                 property_dims: List[str] = None):
        self._facts: Dict[str, Fact] = {}
        self._history: List[Fact] = []
        self.state_matrix = EnvironmentalStateMatrix(
            element_ids or [], property_dims or [])
        self.transition_matrix = ReactionTransitionMatrix()
        self._degradation_rate = 0.98  # certainty decay per second
        self._last_degraded = time.time()

    # ── Fact management ───────────────────────────────────────────────────

    def assert_known(self, key: str, value: Any, source: str = "sensor",
                      sense: SenseVector = None,
                      ttl_ms: float = 0) -> Fact:
        """
        Assert a KNOWN fact — physically verified, certainty = 1.0.
        This is what sensors, actuators with feedback, physical laws provide.
        """
        return self._store(key, value, CertaintyLevel.KNOWN, source, sense, ttl_ms)

    def assert_verified(self, key: str, value: Any, source: str = "multi-sensor",
                         sense: SenseVector = None) -> Fact:
        """Multi-source confirmed fact — certainty = 0.95."""
        return self._store(key, value, CertaintyLevel.VERIFIED, source, sense)

    def assert_probable(self, key: str, value: Any, source: str = "inference",
                         sense: SenseVector = None) -> Fact:
        """Inferred fact — certainty = 0.75."""
        return self._store(key, value, CertaintyLevel.PROBABLE, source, sense)

    def assert_estimated(self, key: str, value: Any, source: str = "model",
                          sense: SenseVector = None) -> Fact:
        """Model estimate — certainty = 0.50. NOT a verified fact."""
        return self._store(key, value, CertaintyLevel.ESTIMATED, source, sense)

    def _store(self, key: str, value: Any, certainty: CertaintyLevel,
               source: str, sense: SenseVector = None, ttl_ms: float = 0) -> Fact:
        valid_until = (time.time() + ttl_ms/1000) if ttl_ms > 0 else 0.
        fact = Fact(key, value, certainty, source,
                    sense or SENSE_NULL, time.time(), valid_until)
        self._facts[key] = fact
        self._history.append(fact)
        return fact

    # ── Queries ───────────────────────────────────────────────────────────

    def know(self, key: str) -> Optional[Fact]:
        """Return fact if known and valid. Returns None if expired."""
        f = self._facts.get(key)
        return f if f and f.is_valid else None

    def is_certain(self, key: str) -> bool:
        """Is this fact KNOWN (certainty ≥ 0.95)?"""
        f = self.know(key)
        return f is not None and f.is_certain

    def is_probable(self, key: str) -> bool:
        """Is this fact probable (certainty ≥ 0.50)?"""
        f = self.know(key)
        return f is not None and f.is_probable

    def certainty_of(self, key: str) -> float:
        f = self.know(key)
        return f.certainty.value if f else 0.0

    def value_of(self, key: str, default: Any = None) -> Any:
        f = self.know(key)
        return f.value if f else default

    # ── Post-action state update ───────────────────────────────────────────

    def record_action_outcome(self, action_type: str, element_id: str,
                               payload: Dict, certainty: CertaintyLevel):
        """
        After a physical action executes, update state matrix with KNOWN effect.

        This is the answer to: "Does the robot KNOW where the object is
        after moving it?" YES — if the action has a transition registered.

        The robot doesn't re-estimate. It KNOWS based on what it did.
        """
        self.transition_matrix.apply(
            self.state_matrix, action_type, element_id,
            payload, certainty.value)

        # Assert the outcome as a fact
        effects = self.transition_matrix.expected_effects(action_type)
        for dim, _ in effects.items():
            val, cert = self.state_matrix.get(element_id, dim)
            self.assert_known(f"{element_id}.{dim}", val,
                              source=f"action:{action_type}")

    # ── Certainty degradation ─────────────────────────────────────────────

    def degrade_over_time(self):
        """
        Facts degrade in certainty as time passes without refresh.
        A GPS position 30 seconds old is less certain than a fresh one.
        A physically manipulated object position remains KNOWN until
        something else interacts with it.
        """
        now   = time.time()
        dt    = now - self._last_degraded
        if dt < 1.0: return  # degrade at most once per second

        for key, fact in list(self._facts.items()):
            if fact.certainty == CertaintyLevel.KNOWN:
                continue  # KNOWN facts don't degrade (physically verified)
            if fact.certainty.value < CertaintyLevel.UNKNOWN.value + 0.01:
                del self._facts[key]  # expired uncertainty
                continue
            self._facts[key] = fact.degrade(self._degradation_rate)

        self._last_degraded = now

    # ── SAVOIR snapshot for MEP/LLM ───────────────────────────────────────

    def snapshot(self, include_uncertain: bool = False) -> Dict:
        """
        Generate a SAVOIR snapshot for the AI to consume.
        KNOWN facts are marked clearly — the AI knows what is verified.
        """
        self.degrade_over_time()
        known     = {k:f for k,f in self._facts.items() if f.is_certain and f.is_valid}
        probable  = {k:f for k,f in self._facts.items() if f.is_probable and not f.is_certain and f.is_valid}
        uncertain = {k:f for k,f in self._facts.items() if not f.is_probable and f.is_valid}

        snap = {
            "known_facts": {k: {"value": f.value, "certainty": str(f.certainty),
                                 "source": f.source, "age_ms": round(f.age_ms)}
                            for k,f in known.items()},
            "probable_facts": {k: {"value": f.value, "certainty": round(f.certainty.value,3),
                                    "source": f.source}
                               for k,f in probable.items()},
            "state_matrix": {
                "shape": f"{self.state_matrix.N}×{self.state_matrix.D}",
                "avg_certainty": round(self.state_matrix.average_certainty(), 3),
                "embedding_dim": self.state_matrix.N * self.state_matrix.D * 2,
            },
            "total_known": len(known),
            "total_probable": len(probable),
        }
        if include_uncertain:
            snap["uncertain_facts"] = {k: {"value":f.value,"certainty":round(f.certainty.value,3)}
                                        for k,f in uncertain.items()}
        return snap

    def to_llm_context(self) -> str:
        """
        Format SAVOIR knowledge as clear context for an LLM.
        Explicitly separates KNOWN from ESTIMATED — the LLM cannot confuse them.
        """
        snap = self.snapshot()
        lines = ["SAVOIR KNOWLEDGE BASE:", ""]
        if snap["known_facts"]:
            lines.append("KNOWN (verified, certainty=1.0):")
            for k,v in snap["known_facts"].items():
                lines.append(f"  ✓ {k} = {v['value']}  [source: {v['source']}]")
        if snap["probable_facts"]:
            lines.append("\nPROBABLE (inferred, certainty<1.0):")
            for k,v in snap["probable_facts"].items():
                lines.append(f"  ~ {k} ≈ {v['value']}  [c={v['certainty']:.2f}]")
        lines.append(f"\nState matrix: {snap['state_matrix']['shape']}"
                     f"  avg_certainty={snap['state_matrix']['avg_certainty']:.2f}")
        return "\n".join(lines)

    @property
    def known_count(self) -> int: return sum(1 for f in self._facts.values() if f.is_certain)
    @property
    def all_facts(self) -> Dict[str, Fact]: return dict(self._facts)
