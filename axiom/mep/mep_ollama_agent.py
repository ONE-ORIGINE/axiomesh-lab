"""
mep_ollama_agent.py
═══════════════════════════════════════════════════════════════════
MEP/2.0 × Ollama  —  AI Agent operating inside a structured environment

Architecture:
  ┌─────────────────────────────────────────────────────────┐
  │  Ollama (local LLM)                                     │
  │  "llama3" / "mistral" / any model                       │
  │   ↕ intent translation (natural language → MEP action)  │
  ├─────────────────────────────────────────────────────────┤
  │  MEP/2.0 Protocol Layer                                 │
  │  ContextEnvelope · CausalGraph · WHY/WHY-NOT           │
  ├─────────────────────────────────────────────────────────┤
  │  ESchool Environment                                    │
  │  Contexts · Circumstances · Actions · Reactions         │
  │  Interaction detection · Phenomenon detection           │
  └─────────────────────────────────────────────────────────┘

What this demonstrates:
  1. AI receives a rich ContextEnvelope — not just text, but:
       - available actions (ranked by causal gravity)
       - active circumstances (what conditions hold right now)
       - situation snapshot (the qualitative state of the environment)
       - recent events (what just happened)

  2. AI translates its intent into a MEP action via structured reasoning

  3. Every action is causally traced:
       AI's decision → MEP action → environment reaction → causal node

  4. AI can query WHY an action succeeded and WHY-NOT it was blocked

  5. Phenomena emerge from sustained AI action patterns (mass grading, etc.)

  6. The AI never directly mutates the environment — it operates through
     the causal layer, respecting circumstance gates

Usage:
  # With Ollama running (ollama serve):
  python3 mep_ollama_agent.py

  # Without Ollama (demo mode — uses structured prompts, prints responses):
  python3 mep_ollama_agent.py --demo
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
import math
import textwrap
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple
import urllib.request
import urllib.error
import argparse

# ─── ANSI colors ────────────────────────────────────────────────────────────

class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    MAGENTA= "\033[95m"
    BLUE   = "\033[94m"
    WHITE  = "\033[97m"

def hr(char="═", n=65, color=C.DIM): print(f"{color}{char*n}{C.RESET}")
def section(title): hr(); print(f"{C.BOLD}{C.CYAN}  {title}{C.RESET}"); hr()
def tag(label, msg, color=C.GREEN): print(f"  {color}[{label}]{C.RESET} {msg}")

# =============================================================================
# SENSE VECTOR — 8-dimensional semantic space
# =============================================================================

DIMS = 8
AXES = ["causal","temporal","spatial","normative","social","financial","technical","emergent"]

@dataclass(frozen=True)
class Sense:
    dimension : str
    meaning   : str
    magnitude : float
    vector    : Tuple[float, ...]

    @classmethod
    def of(cls, dim: str, meaning: str, axis: int, magnitude: float = 1.0) -> "Sense":
        vec = [0.0] * DIMS
        vec[axis] = magnitude
        return cls(dim, meaning, magnitude, tuple(vec))

    def affinity(self, other: "Sense") -> float:
        dot  = sum(a*b for a,b in zip(self.vector, other.vector))
        na   = math.sqrt(sum(x*x for x in self.vector))
        nb   = math.sqrt(sum(x*x for x in other.vector))
        return dot / (na * nb) if na > 0 and nb > 0 else 0.0

    @classmethod
    def normative(cls, meaning: str, magnitude: float = 1.0) -> "Sense":
        return cls.of("normative", meaning, 3, magnitude)

    @classmethod
    def temporal(cls, meaning: str, magnitude: float = 1.0) -> "Sense":
        return cls.of("temporal", meaning, 1, magnitude)

    @classmethod
    def technical(cls, meaning: str, magnitude: float = 1.0) -> "Sense":
        return cls.of("technical", meaning, 6, magnitude)

    @classmethod
    def social(cls, meaning: str, magnitude: float = 1.0) -> "Sense":
        return cls.of("social", meaning, 4, magnitude)

SENSE_EMPTY = Sense("none","",0.0,tuple([0.0]*DIMS))

# =============================================================================
# CIRCUMSTANCE — Live boolean predicate
# =============================================================================

class Circumstance:
    def __init__(self, cid: str, desc: str, fn: Callable, role: str = "enabler",
                 weight: float = 1.0):
        self.id = cid; self.description = desc
        self._fn = fn; self.role = role; self.weight = weight

    def evaluate(self, context: "ContextFrame", frame: Dict) -> bool:
        return self._fn(context, frame)

    def __and__(self, o: "Circumstance") -> "Circumstance":
        return Circumstance(f"{self.id}&{o.id}", f"({self.description}) AND ({o.description})",
            lambda c, f: self._fn(c, f) and o._fn(c, f))

    def __or__(self, o: "Circumstance") -> "Circumstance":
        return Circumstance(f"{self.id}|{o.id}", f"({self.description}) OR ({o.description})",
            lambda c, f: self._fn(c, f) or o._fn(c, f))

    def __invert__(self) -> "Circumstance":
        return Circumstance(f"!{self.id}", f"NOT ({self.description})",
            lambda c, f: not self._fn(c, f), "blocker", -self.weight)

    def to_dict(self) -> Dict:
        return {"id": self.id, "description": self.description, "role": self.role}

# Common circumstance factories
def circ_flag(cid: str, desc: str, key: str, expected: bool = True) -> Circumstance:
    return Circumstance(cid, desc,
        lambda ctx, _: ctx.data.get(key) == expected)

def circ_always(cid: str) -> Circumstance:
    return Circumstance(cid, cid, lambda *_: True)

def circ_role(cid: str, role_type: type) -> Circumstance:
    return Circumstance(cid, f"Actor is {role_type.__name__}",
        lambda ctx, frame: any(e.get("type") == role_type.__name__
                               for e in ctx.scoped_elements
                               if e.get("id") == frame.get("actor_id")))

# =============================================================================
# CONTEXT FRAME — Multidimensional, exportable, AI-consumable
# =============================================================================

@dataclass
class ContextFrame:
    context_id      : str = field(default_factory=lambda: str(uuid.uuid4()))
    name            : str = ""
    kind            : str = "semantic"
    basis           : Sense = field(default_factory=lambda: SENSE_EMPTY)
    parent_id       : Optional[str] = None
    depth           : int = 0
    data            : Dict[str, Any] = field(default_factory=dict)
    scoped_elements : List[Dict]     = field(default_factory=list)
    circumstances   : List[Circumstance] = field(default_factory=list)
    _registered_actions : List[Dict] = field(default_factory=list, repr=False)

    def add_circumstance(self, c: Circumstance) -> "ContextFrame":
        self.circumstances.append(c); return self

    def register_action(self, action: "EnvAction",
                        actor_filter: Optional[Callable] = None) -> "ContextFrame":
        self._registered_actions.append({"action": action, "filter": actor_filter})
        return self

    def get_available_actions(self, actor: Dict, frame: Dict) -> List[Tuple["EnvAction",float]]:
        result = []
        for entry in self._registered_actions:
            action = entry["action"]
            filt   = entry["filter"]
            if filt and not filt(actor): continue
            if not all(c.evaluate(self, frame) for c in action.guards): continue
            gravity = action.sense.affinity(self.basis)
            result.append((action, gravity))
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def evaluate_circumstances(self, frame: Dict) -> List[Dict]:
        return [{"id": c.id, "description": c.description,
                 "holds": c.evaluate(self, frame), "role": c.role}
                for c in self.circumstances]

    def to_envelope(self, actor: Dict, frame: Dict) -> "ContextEnvelope":
        available = self.get_available_actions(actor, frame)
        circs     = self.evaluate_circumstances(frame)
        return ContextEnvelope(
            context_id      = self.context_id,
            name            = self.name,
            kind            = self.kind,
            depth           = self.depth,
            data            = dict(self.data),
            circumstances   = circs,
            available_actions = [
                {"type": a.type, "category": a.category, "description": a.description,
                 "gravity": round(g, 3), "sense_dimension": a.sense.dimension,
                 "can_chain": a.can_chain}
                for a, g in available
            ],
        )

    def set(self, key: str, val: Any) -> "ContextFrame":
        self.data[key] = val; return self

    def include(self, element: Dict) -> "ContextFrame":
        self.scoped_elements.append(element); return self

# =============================================================================
# CONTEXT ENVELOPE — The AI-consumable snapshot (the key MEP innovation)
# =============================================================================

@dataclass
class ContextEnvelope:
    """
    What the AI receives. Not a text prompt — a structured semantic frame.
    Contains available actions (ranked by causal gravity), holding
    circumstances, situation snapshot, and recent causal history.
    """
    context_id        : str = ""
    name              : str = ""
    kind              : str = "semantic"
    depth             : int = 0
    data              : Dict[str, Any]  = field(default_factory=dict)
    circumstances     : List[Dict]      = field(default_factory=list)
    available_actions : List[Dict]      = field(default_factory=list)
    situation         : Optional[Dict]  = None
    recent_events     : List[Dict]      = field(default_factory=list)

    def to_prompt(self) -> str:
        """Convert this envelope to a structured prompt for the LLM."""
        holding = [c for c in self.circumstances if c.get("holds")]
        blocked = [c for c in self.circumstances
                   if c.get("holds") and c.get("role") == "blocker"]

        actions_str = "\n".join(
            f"  {i+1}. [{a['category'].upper()}] {a['type']}"
            f" (gravity={a['gravity']:.2f}) — {a['description']}"
            for i, a in enumerate(self.available_actions)
        ) or "  (none available)"

        holding_str = "\n".join(
            f"  ✓ {c['description']}" for c in holding
        ) or "  (none)"

        blocked_str = "\n".join(
            f"  ✗ {c['description']}" for c in blocked
        ) or "  (none)"

        situation_str = ""
        if self.situation:
            s = self.situation
            situation_str = (f"\nSITUATION: {s.get('kind','?').upper()} "
                             f"— severity={s.get('severity','?')}, "
                             f"saturation={s.get('saturation',0):.0%}")

        data_str = "\n".join(
            f"  {k}: {v}" for k, v in self.data.items()
            if not k.startswith("_")
        ) or "  (empty)"

        events_str = ""
        if self.recent_events:
            events_str = "\nRECENT EVENTS:\n" + "\n".join(
                f"  • {e.get('type','')} — {e.get('summary','')}"
                for e in self.recent_events[-3:]
            )

        return textwrap.dedent(f"""
            CONTEXT: {self.name} [{self.kind}] (depth={self.depth}){situation_str}

            ACTIVE CIRCUMSTANCES (conditions holding right now):
            {holding_str}

            BLOCKED BY:
            {blocked_str}

            CONTEXT DATA:
            {data_str}
            {events_str}

            AVAILABLE ACTIONS (ranked by causal gravity — how natural in this context):
            {actions_str}
        """).strip()

# =============================================================================
# CAUSAL GRAPH — Persistent session memory
# =============================================================================

@dataclass
class CausalNode:
    node_id       : str
    node_type     : str   # "action" | "reaction"
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
        self._nodes: Dict[str, CausalNode] = {}
        self._children: Dict[str, List[str]] = {}

    def add(self, node: CausalNode):
        self._nodes[node.node_id] = node
        if node.causation_id:
            self._children.setdefault(node.causation_id, []).append(node.node_id)

    def ancestry(self, node_id: str) -> List[CausalNode]:
        chain, current_id = [], node_id
        while current_id and (node := self._nodes.get(current_id)):
            chain.insert(0, node)
            current_id = node.causation_id or ""
        return chain

    def explain_why(self, reaction_type: str) -> str:
        matches = [n for n in self._nodes.values() if n.action_type == reaction_type]
        if not matches: return f"No record found for '{reaction_type}'"
        node  = matches[-1]
        chain = self.ancestry(node.node_id)
        lines = [f"Causal chain for '{reaction_type}':"]
        for n in chain:
            lines.append(f"  [depth {n.depth}] {n.node_type}: {n.action_type} "
                         f"({n.status}) — {n.summary}")
        return "\n".join(lines)

    def stats(self) -> Dict:
        actions   = [n for n in self._nodes.values() if n.node_type == "action"]
        reactions = [n for n in self._nodes.values() if n.node_type == "reaction"]
        types     = list({n.action_type for n in self._nodes.values()})
        return {
            "total_nodes": len(self._nodes),
            "actions":     len(actions),
            "reactions":   len(reactions),
            "action_types": types,
        }

# =============================================================================
# ACTION & REACTION
# =============================================================================

@dataclass
class EnvAction:
    type        : str
    category    : str
    description : str
    sense       : Sense
    guards      : List[Circumstance] = field(default_factory=list)
    can_chain   : bool = True
    _handler    : Any  = field(default=None, repr=False)

    def can_execute(self, actor: Dict, context: ContextFrame, frame: Dict) -> bool:
        return all(g.evaluate(context, frame) for g in self.guards)

    async def execute(self, actor: Dict, payload: Dict,
                      context: ContextFrame, frame: Dict) -> "EnvReaction":
        # Guard check
        for g in self.guards:
            if not g.evaluate(context, frame):
                return EnvReaction(
                    action_type=self.type, status="rejected",
                    message=f"Guard not met: {g.description}",
                    impact_scope="none")
        if not self._handler:
            return EnvReaction(action_type=self.type, status="success",
                               message="OK (no handler)")
        return await self._handler(actor, payload, context, frame)

@dataclass
class EnvReaction:
    action_type   : str
    status        : str       # success | rejected | deferred
    message       : Optional[str] = None
    result        : Any           = None
    impact_scope  : str           = "actor"   # actor | specific | environment | all
    target_id     : Optional[str] = None
    temporality   : str           = "immediate"
    chain_actions : List[str]     = field(default_factory=list)  # action types to spawn
    reaction_id   : str           = field(default_factory=lambda: str(uuid.uuid4()))
    produced_at   : float         = field(default_factory=time.time)

    def to_summary(self) -> str:
        icon = "✓" if self.status == "success" else "✗"
        base = f"{icon} {self.action_type} → {self.status}"
        if self.message: base += f" | {self.message}"
        if self.result:  base += f" → {json.dumps(self.result, default=str)[:80]}"
        return base

# =============================================================================
# SLIDING WINDOW PHENOMENON DETECTOR
# =============================================================================

class PhenomenonDetector:
    def __init__(self):
        self._history: List[Tuple[float, str]] = []  # (timestamp, reaction_type)
        self._detected: List[Dict] = []

    def record(self, reaction_type: str):
        self._history.append((time.time(), reaction_type))
        # Keep last 200
        if len(self._history) > 200:
            self._history = self._history[-200:]

    def detect(self, pattern_type: str, threshold: int,
               window_seconds: float) -> Optional[Dict]:
        cutoff  = time.time() - window_seconds
        matches = [t for t, rt in self._history
                   if rt == pattern_type and t >= cutoff]
        if len(matches) >= threshold:
            phenom = {
                "name": f"{pattern_type}.phenomenon",
                "pattern_type": pattern_type,
                "magnitude": min(1.0, len(matches) / threshold),
                "count": len(matches),
                "window_s": window_seconds,
                "detected_at": time.time()
            }
            self._detected.append(phenom)
            return phenom
        return None

    @property
    def all_detected(self) -> List[Dict]:
        return self._detected

# =============================================================================
# ESCHOOL ENVIRONMENT — MEP Server
# =============================================================================

class ESchoolEnv:
    """
    A structured school environment exposing itself via the MEP protocol.
    AI agents interact with it through contexts, not direct API calls.
    """

    def __init__(self):
        self.name = "ESchool"
        self._students:      Dict[str, Dict] = {}
        self._teachers:      Dict[str, Dict] = {}
        self._admins:        Dict[str, Dict] = {}
        self._courses:       Dict[str, Dict] = {}
        self._schools:       Dict[str, Dict] = {}
        self._grades:        List[Dict]      = []
        self._enrollments:   List[Dict]      = []
        self._events:        List[Dict]      = []
        self._causal:        CausalGraph     = CausalGraph()
        self._phenomena:     PhenomenonDetector = PhenomenonDetector()
        self._tick:          int = 0

        # ── Global circumstances ─────────────────────────────────────────
        self.C_SYSTEM_ACTIVE    = circ_always("system.active")
        self.C_ENROLLMENT_OPEN  = circ_flag("enrollment.open",
                                             "Enrollment period is active",
                                             "enrollmentOpen", True)
        self.C_ENROLLMENT_CLOSED = ~self.C_ENROLLMENT_OPEN

        # ── Contexts ─────────────────────────────────────────────────────
        self.root_ctx  = self._build_root_context()
        self.admin_ctx = self._build_admin_context()
        self.acad_ctx  = self._build_academic_context()
        self.enrl_ctx  = self._build_enrollment_context()

        self._global_data = {"enrollmentOpen": False, "systemActive": True}

    # ── Context construction ──────────────────────────────────────────────

    def _build_root_context(self) -> ContextFrame:
        ctx = ContextFrame(
            name  = "ESchool.Root",
            kind  = "global",
            basis = Sense.technical("global environment frame", 0.5))
        ctx.add_circumstance(self.C_SYSTEM_ACTIVE)
        return ctx

    def _build_admin_context(self) -> ContextFrame:
        ctx = ContextFrame(
            name   = "ESchool.Administrative",
            kind   = "semantic",
            basis  = Sense.normative("administrative operations", 0.9),
            parent_id = self.root_ctx.context_id,
            depth  = 1)
        ctx.add_circumstance(self.C_SYSTEM_ACTIVE)
        # Actions
        ctx.register_action(self._make_create_school(),  lambda a: a.get("type") == "Admin")
        ctx.register_action(self._make_create_course(),  lambda a: a.get("type") in ("Admin","Teacher"))
        ctx.register_action(self._make_assign_teacher(), lambda a: a.get("type") == "Admin")
        ctx.register_action(self._make_open_enrollment(), lambda a: a.get("type") == "Admin")
        ctx.register_action(self._make_close_enrollment(), lambda a: a.get("type") == "Admin")
        ctx.register_action(self._make_system_snapshot())
        return ctx

    def _build_academic_context(self) -> ContextFrame:
        ctx = ContextFrame(
            name   = "ESchool.Academic",
            kind   = "semantic",
            basis  = Sense.normative("academic grading assessment", 0.95),
            parent_id = self.root_ctx.context_id,
            depth  = 1)
        ctx.add_circumstance(self.C_SYSTEM_ACTIVE)
        ctx.register_action(self._make_record_grade(), lambda a: a.get("type") == "Teacher")
        ctx.register_action(self._make_query_student_grades())
        ctx.register_action(self._make_system_snapshot())
        return ctx

    def _build_enrollment_context(self) -> ContextFrame:
        ctx = ContextFrame(
            name   = "ESchool.Enrollment",
            kind   = "temporal",
            basis  = Sense.temporal("enrollment period operations", 0.9),
            parent_id = self.root_ctx.context_id,
            depth  = 1)
        ctx.add_circumstance(self.C_SYSTEM_ACTIVE)
        ctx.add_circumstance(self.C_ENROLLMENT_OPEN)
        ctx.register_action(self._make_enroll_student())
        ctx.register_action(self._make_withdraw_student())
        ctx.register_action(self._make_query_enrollment())
        return ctx

    # ── Action factories ──────────────────────────────────────────────────

    def _make_create_school(self) -> EnvAction:
        async def handler(actor, payload, ctx, frame):
            name    = payload.get("name", "Unnamed School")
            code    = payload.get("code", uuid.uuid4().hex[:6].upper())
            country = payload.get("countryCode", "INT")
            sid     = str(uuid.uuid4())
            self._schools[sid] = {"id":sid,"name":name,"code":code,"country":country,
                                   "type":"School","createdAt":time.time()}
            self._emit_event("school.created", f"School '{name}' ({code}) created")
            return EnvReaction("school.create","success",
                               f"School '{name}' created successfully",
                               result={"id":sid,"name":name,"code":code},
                               impact_scope="environment")
        return EnvAction("school.create","lifecycle","Create a new school institution",
                         Sense.normative("school creation",0.9),
                         guards=[self.C_SYSTEM_ACTIVE], _handler=handler)

    def _make_create_course(self) -> EnvAction:
        async def handler(actor, payload, ctx, frame):
            name     = payload["name"]
            code     = payload.get("code", uuid.uuid4().hex[:6].upper())
            disc     = payload.get("discipline","General")
            capacity = int(payload.get("capacity", 30))
            credits  = int(payload.get("credits", 3))
            cid      = str(uuid.uuid4())
            self._courses[cid] = {
                "id":cid,"name":name,"code":code,"discipline":disc,
                "maxCapacity":capacity,"credits":credits,"enrolled":0,
                "type":"Course","grades":[],"enrolledStudents":[]}
            self._emit_event("course.created", f"Course '{name}' ({code}) created")
            return EnvReaction("course.create","success",
                               f"Course '{name}' created (cap={capacity})",
                               result={"id":cid,"name":name,"code":code},
                               impact_scope="environment")
        return EnvAction("course.create","lifecycle","Create a new course",
                         Sense.technical("course provisioning",0.8),
                         _handler=handler)

    def _make_assign_teacher(self) -> EnvAction:
        async def handler(actor, payload, ctx, frame):
            tid = payload.get("teacherId","")
            cid = payload.get("courseId","")
            t   = self._teachers.get(tid)
            c   = self._courses.get(cid)
            if not t: return EnvReaction("course.assign-teacher","rejected","Teacher not found")
            if not c: return EnvReaction("course.assign-teacher","rejected","Course not found")
            c["teacherId"] = tid
            c["teacherName"] = t["name"]
            self._emit_event("teacher.assigned",
                             f"Teacher {t['name']} → {c['name']}")
            return EnvReaction("course.assign-teacher","success",
                               f"{t['name']} assigned to {c['name']}",
                               result={"teacherName":t["name"],"courseName":c["name"]},
                               impact_scope="specific", target_id=cid)
        return EnvAction("course.assign-teacher","command","Assign a teacher to a course",
                         Sense.social("teacher assignment",0.7),_handler=handler)

    def _make_open_enrollment(self) -> EnvAction:
        async def handler(actor, payload, ctx, frame):
            self._global_data["enrollmentOpen"] = True
            self._update_contexts_data("enrollmentOpen", True)
            self._emit_event("enrollment.opened","Enrollment period is now open")
            return EnvReaction("enrollment.open","success",
                               "Enrollment period opened",impact_scope="environment")
        return EnvAction("enrollment.open","command","Open the enrollment period",
                         Sense.temporal("enrollment activation",0.85),_handler=handler)

    def _make_close_enrollment(self) -> EnvAction:
        async def handler(actor, payload, ctx, frame):
            self._global_data["enrollmentOpen"] = False
            self._update_contexts_data("enrollmentOpen", False)
            self._emit_event("enrollment.closed","Enrollment period is now closed")
            return EnvReaction("enrollment.close","success",
                               "Enrollment period closed",impact_scope="environment")
        return EnvAction("enrollment.close","command","Close the enrollment period",
                         Sense.temporal("enrollment deactivation",0.85),_handler=handler)

    def _make_enroll_student(self) -> EnvAction:
        async def handler(actor, payload, ctx, frame):
            sid = payload.get("studentId","")
            cid = payload.get("courseId","")
            s   = self._students.get(sid)
            c   = self._courses.get(cid)
            if not s: return EnvReaction("student.enroll","rejected","Student not found")
            if not c: return EnvReaction("student.enroll","rejected","Course not found")
            if c["enrolled"] >= c["maxCapacity"]:
                return EnvReaction("student.enroll","rejected",
                                   f"Course at capacity ({c['maxCapacity']})")
            if sid in c["enrolledStudents"]:
                return EnvReaction("student.enroll","rejected","Already enrolled")
            c["enrolledStudents"].append(sid)
            c["enrolled"] += 1
            s.setdefault("enrolledCourses",[]).append(cid)
            self._enrollments.append({"studentId":sid,"courseId":cid,"at":time.time()})
            self._emit_event("enrollment.confirmed",
                             f"{s['name']} enrolled in {c['name']}")
            return EnvReaction("student.enroll","success",
                               f"{s['name']} enrolled in {c['name']}",
                               result={"studentName":s["name"],"courseName":c["name"]},
                               impact_scope="specific", target_id=cid)
        capacity_circ = Circumstance("has.capacity","Course has available seats",
            lambda ctx, frame: any(
                c.get("enrolled",0) < c.get("maxCapacity",0)
                for c in self._courses.values()
                if c["id"] == frame.get("payload",{}).get("courseId","")),
            weight=1.0)
        return EnvAction("student.enroll","command","Enroll a student in a course",
                         Sense.temporal("enrollment operation",0.95),
                         guards=[self.C_ENROLLMENT_OPEN], _handler=handler)

    def _make_withdraw_student(self) -> EnvAction:
        async def handler(actor, payload, ctx, frame):
            sid = payload.get("studentId","")
            cid = payload.get("courseId","")
            s   = self._students.get(sid)
            c   = self._courses.get(cid)
            if not s or not c:
                return EnvReaction("student.withdraw","rejected","Student or course not found")
            if sid not in c.get("enrolledStudents",[]):
                return EnvReaction("student.withdraw","rejected","Not enrolled")
            c["enrolledStudents"].remove(sid)
            c["enrolled"] -= 1
            if cid in s.get("enrolledCourses",[]):
                s["enrolledCourses"].remove(cid)
            self._emit_event("withdrawal.confirmed",
                             f"{s['name']} withdrew from {c['name']}")
            return EnvReaction("student.withdraw","success",
                               f"{s['name']} withdrew from {c['name']}",
                               impact_scope="specific", target_id=cid)
        return EnvAction("student.withdraw","command","Withdraw a student from a course",
                         Sense.temporal("withdrawal operation",0.7),_handler=handler)

    def _make_record_grade(self) -> EnvAction:
        async def handler(actor, payload, ctx, frame):
            sid   = payload.get("studentId","")
            cid   = payload.get("courseId","")
            grade = float(payload.get("grade", 0))
            s     = self._students.get(sid)
            c     = self._courses.get(cid)
            if not s or not c:
                return EnvReaction("grade.record","rejected","Student or course not found")
            if not (0 <= grade <= 100):
                return EnvReaction("grade.record","rejected","Grade must be 0-100")
            letter = ("A" if grade>=90 else "B" if grade>=80 else
                      "C" if grade>=70 else "D" if grade>=60 else "F")
            rec = {"studentId":sid,"courseId":cid,"grade":grade,
                   "letter":letter,"at":time.time(),"teacher":actor.get("name","")}
            self._grades.append(rec)
            c.setdefault("grades",[]).append(rec)
            # Update student GPA
            student_grades = [r["grade"] for r in self._grades if r["studentId"]==sid]
            s["gpa"] = round(sum(student_grades)/len(student_grades), 2)
            passing = grade >= 60
            self._emit_event("grade.recorded",
                             f"{s['name']}: {grade:.1f} ({letter}) in {c['name']}")
            # Phenomenon detection: mass failure
            if not passing:
                phenom = self._phenomena.detect("grade.record.fail", 3, 300)
                if phenom:
                    tag("PHENOMENON", f"MassFailure detected — magnitude={phenom['magnitude']:.0%}",
                        C.MAGENTA)
            self._phenomena.record("grade.record.fail" if not passing else "grade.record.pass")
            chain_actions = [] if passing else ["notification.dispatch"]
            return EnvReaction("grade.record","success",
                               f"{s['name']}: {grade:.1f} ({letter}) in {c['name']}",
                               result={"grade":grade,"letter":letter,"passing":passing,
                                       "studentGPA":s["gpa"]},
                               impact_scope="specific", target_id=cid,
                               chain_actions=chain_actions)
        return EnvAction("grade.record","command","Record a student grade",
                         Sense.normative("academic assessment",0.95),
                         guards=[self.C_SYSTEM_ACTIVE], _handler=handler)

    def _make_query_student_grades(self) -> EnvAction:
        async def handler(actor, payload, ctx, frame):
            sid     = payload.get("studentId","")
            student = self._students.get(sid)
            if not student:
                return EnvReaction("student.query-grades","rejected","Student not found")
            grades  = [r for r in self._grades if r["studentId"]==sid]
            return EnvReaction("student.query-grades","success",
                               f"Found {len(grades)} grades for {student['name']}",
                               result={"student":student["name"],"gpa":student.get("gpa",0),
                                       "grades":grades})
        return EnvAction("student.query-grades","query","Query grades for a student",
                         Sense.normative("grade query",0.7),_handler=handler)

    def _make_query_enrollment(self) -> EnvAction:
        async def handler(actor, payload, ctx, frame):
            cid    = payload.get("courseId","")
            course = self._courses.get(cid)
            if not course:
                return EnvReaction("enrollment.query","rejected","Course not found")
            students = [self._students[sid]["name"]
                        for sid in course.get("enrolledStudents",[])
                        if sid in self._students]
            return EnvReaction("enrollment.query","success",
                               f"{course['name']}: {course['enrolled']}/{course['maxCapacity']}",
                               result={"courseName":course["name"],
                                       "enrolled":course["enrolled"],
                                       "capacity":course["maxCapacity"],
                                       "students":students})
        return EnvAction("enrollment.query","query","Query enrollment for a course",
                         Sense.temporal("enrollment status",0.6),_handler=handler)

    def _make_system_snapshot(self) -> EnvAction:
        async def handler(actor, payload, ctx, frame):
            return EnvReaction("system.snapshot","success","Snapshot taken",
                               result={
                                   "schools":len(self._schools),
                                   "courses":len(self._courses),
                                   "students":len(self._students),
                                   "teachers":len(self._teachers),
                                   "grades":len(self._grades),
                                   "enrollments":len(self._enrollments),
                                   "enrollmentOpen":self._global_data.get("enrollmentOpen"),
                                   "phenomena":len(self._phenomena.all_detected),
                               })
        return EnvAction("system.snapshot","query","Snapshot the environment state",
                         Sense.technical("system state query",0.3),_handler=handler)

    # ── Environment management ────────────────────────────────────────────

    def add_student(self, name: str, code: str) -> Dict:
        sid = str(uuid.uuid4())
        e   = {"id":sid,"name":name,"code":code,"type":"Student","gpa":0,"enrolledCourses":[]}
        self._students[sid] = e; return e

    def add_teacher(self, name: str, code: str, discipline: str) -> Dict:
        tid = str(uuid.uuid4())
        e   = {"id":tid,"name":name,"code":code,"type":"Teacher","discipline":discipline}
        self._teachers[tid] = e; return e

    def add_admin(self, name: str, code: str, role: str = "SystemAdmin") -> Dict:
        aid = str(uuid.uuid4())
        e   = {"id":aid,"name":name,"code":code,"type":"Admin","role":role}
        self._admins[aid] = e; return e

    def _update_contexts_data(self, key: str, val: Any):
        for ctx in [self.root_ctx, self.admin_ctx, self.acad_ctx, self.enrl_ctx]:
            ctx.data[key] = val

    def _emit_event(self, event_type: str, summary: str):
        self._events.append({"type":event_type,"summary":summary,"at":time.time()})
        self._tick += 1

    async def dispatch(self, actor: Dict, action_type: str, payload: Dict,
                       ctx: ContextFrame, correlation_id: str) -> EnvReaction:
        """Execute an action through the environment."""
        # Find action across contexts
        frame = {"actor_id": actor["id"], "payload": payload}
        for registered in ctx._registered_actions:
            if registered["action"].type == action_type:
                action   = registered["action"]
                reaction = await action.execute(actor, payload, ctx, frame)
                # Record in causal graph
                node = CausalNode(
                    node_id        = str(uuid.uuid4()),
                    node_type      = "action",
                    action_type    = action_type,
                    actor_id       = actor["id"],
                    context_name   = ctx.name,
                    correlation_id = correlation_id,
                    causation_id   = None,
                    depth          = 0,
                    timestamp      = time.time(),
                    status         = reaction.status,
                    summary        = reaction.message or "",
                )
                self._causal.add(node)
                return reaction
        return EnvReaction(action_type, "rejected",
                           f"Action '{action_type}' not found in context '{ctx.name}'")

    def get_envelope(self, actor: Dict, ctx: ContextFrame) -> ContextEnvelope:
        frame    = {"actor_id": actor["id"]}
        ctx.data.update(self._global_data)  # sync global state
        envelope = ctx.to_envelope(actor, frame)
        envelope.situation = {
            "kind":     "operational" if not self._phenomena.all_detected else "degraded",
            "severity": "low" if not self._phenomena.all_detected else "high",
            "saturation": len(self._students) / max(1, len(self._students)+50)
        }
        envelope.recent_events = self._events[-5:]
        return envelope

    def explain(self, action_type: str) -> str:
        return self._causal.explain_why(action_type)

    @property
    def causal_stats(self) -> Dict:
        return self._causal.stats()

# =============================================================================
# OLLAMA CLIENT
# =============================================================================

class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434", model: str = "phi4-mini-reasoning:latest"):
        self.host  = host
        self.model = model
        self._available = self._check()

    def _check(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=2) as r:
                return r.status == 200
        except Exception:
            return False

    @property
    def is_available(self) -> bool:
        return self._available

    def chat(self, system_prompt: str, user_message: str,
             temperature: float = 0.3) -> str:
        """Send a chat message to Ollama and return the response."""
        if not self._available:
            raise RuntimeError("Ollama not available")

        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": user_message},
            ],
            "options": {"temperature": temperature},
            "stream": False,
        }).encode()

        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST")

        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read())
                return data["message"]["content"]
        except urllib.error.URLError as e:
            raise RuntimeError(f"Ollama request failed: {e}")

# =============================================================================
# MEP AI AGENT — The intelligence operating inside the MEP environment
# =============================================================================

class MepAgent:
    """
    An AI agent that operates inside a MEP environment via Ollama.

    The agent:
    1. Receives a ContextEnvelope (structured semantic frame)
    2. Decides which action to take based on its goal + the available actions
    3. Assembles the payload
    4. Dispatches the action through MEP
    5. Interprets the reaction causally
    6. Can ask WHY an action was blocked
    """

    SYSTEM_PROMPT = textwrap.dedent("""
    You are an intelligent AI agent operating inside a structured educational
    management environment (ESchool) via the MEP (Model Environment Protocol).

    You receive a CONTEXT ENVELOPE — a structured semantic frame that tells you:
    - What actions are available (ranked by causal gravity)
    - What circumstances are currently active
    - The current state of the environment
    - Recent events

    You MUST respond with ONLY a JSON object in this exact format:
    {
        "decision": "execute" | "skip" | "query",
        "action_type": "<exact action type from available actions>",
        "payload": { <required parameters for this action> },
        "reasoning": "<brief explanation of why this action>"
    }

    Rules:
    - Only choose actions that appear in AVAILABLE ACTIONS
    - Respect the circumstances — if enrollment is closed, don't try to enroll
    - Prefer higher-gravity actions when multiple options exist
    - "skip" means you choose to do nothing this turn
    - "query" means you want to inspect state before acting
    """).strip()

    def __init__(self, env: ESchoolEnv, ollama: OllamaClient,
                 actor: Dict, demo_mode: bool = False):
        self.env       = env
        self.ollama    = ollama
        self.actor     = actor
        self.demo_mode = demo_mode
        self.session_id  = str(uuid.uuid4())
        self._decision_history: List[Dict] = []

    async def decide(self, goal: str, context: ContextFrame) -> Optional[Dict]:
        """Ask the LLM to decide what action to take given the goal and context."""
        envelope = self.env.get_envelope(self.actor, context)
        ctx_prompt = envelope.to_prompt()

        user_msg = textwrap.dedent(f"""
        CURRENT GOAL: {goal}

        {ctx_prompt}

        Based on your goal and the available actions, decide what to do.
        Respond only with the JSON decision object.
        """).strip()

        if self.demo_mode:
            return self._demo_decide(goal, envelope)

        try:
            response = self.ollama.chat(self.SYSTEM_PROMPT, user_msg, temperature=0.2)
            # Extract JSON from response
            raw = response.strip()
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"): raw = raw[4:]
            decision = json.loads(raw)
            decision["_envelope"] = asdict(envelope)
            return decision
        except Exception as e:
            tag("AGENT-ERR", f"LLM error: {e}", C.RED)
            return None

    def _demo_decide(self, goal: str, envelope: ContextEnvelope) -> Optional[Dict]:
        """Demo mode: deterministic decisions based on goal keywords."""
        goal_lower = goal.lower()
        available  = {a["type"]: a for a in envelope.available_actions}

        candidates = {
            "create school":    ("school.create",    {"name":"DemoSchool","code":"DEMO","countryCode":"FR"}),
            "create course":    ("course.create",    {"name":"Demo Course","code":"DC101","discipline":"General","capacity":"25","credits":"3"}),
            "open enrollment":  ("enrollment.open",  {}),
            "close enrollment": ("enrollment.close", {}),
            "enroll":           ("student.enroll",   {}),  # payload filled later
            "grade":            ("grade.record",     {}),  # payload filled later
            "snapshot":         ("system.snapshot",  {}),
            "query":            ("system.snapshot",  {}),
        }

        for kw, (atype, payload) in candidates.items():
            if kw in goal_lower and atype in available:
                return {"decision":"execute","action_type":atype,
                        "payload":payload,"reasoning":f"Goal '{goal}' matched keyword '{kw}'"}

        # Fallback: pick highest-gravity available action
        if envelope.available_actions:
            best = envelope.available_actions[0]
            return {"decision":"execute","action_type":best["type"],
                    "payload":{},"reasoning":"Highest-gravity available action"}

        return {"decision":"skip","action_type":"","payload":{},"reasoning":"No actions available"}

    async def act(self, goal: str, context: ContextFrame,
                  extra_payload: Optional[Dict] = None) -> Optional[EnvReaction]:
        """Full cycle: decide → dispatch → interpret reaction."""
        decision = await self.decide(goal, context)
        if not decision or decision.get("decision") == "skip":
            tag("AGENT", f"Decision: SKIP — {decision.get('reasoning','')}", C.DIM)
            return None

        action_type = decision["action_type"]
        payload     = {**decision.get("payload", {}), **(extra_payload or {})}
        reasoning   = decision.get("reasoning", "")
        corr_id     = str(uuid.uuid4())

        tag("AGENT", f"{C.BOLD}DECIDES:{C.RESET} {action_type}", C.CYAN)
        tag("REASON", reasoning, C.DIM)
        if payload:
            tag("PAYLOAD", json.dumps(payload, default=str)[:100], C.DIM)

        reaction = await self.env.dispatch(self.actor, action_type, payload,
                                           context, corr_id)

        status_color = C.GREEN if reaction.status == "success" else C.RED
        tag("REACTION", f"{status_color}{reaction.to_summary()}{C.RESET}")

        self._decision_history.append({
            "goal": goal, "action": action_type, "status": reaction.status,
            "reasoning": reasoning, "timestamp": time.time()
        })

        return reaction

# =============================================================================
# MAIN — Full demonstration
# =============================================================================

async def main(demo_mode: bool = True, model: str = "llama3"):

    hr("╔", 65); hr("║ ", 63)
    print(f"  {C.BOLD}{C.CYAN}MEP/2.0 × Ollama  —  AI Agent in a Structured Environment{C.RESET}")
    hr("║ ", 63); hr("╚", 65)
    print()

    # ── Initialize environment ───────────────────────────────────────────

    env   = ESchoolEnv()
    ollama = OllamaClient(model=model)

    if ollama.is_available:
        tag("OLLAMA", f"Connected to Ollama (model={model})", C.GREEN)
    else:
        tag("OLLAMA", f"Not available — running in DEMO mode", C.YELLOW)
        demo_mode = True

    # ── Seed actors ──────────────────────────────────────────────────────

    admin   = env.add_admin("Dr. Elena Vasquez", "ADM001", "SystemAdmin")
    teacher = env.add_teacher("Prof. Sophie Laurent", "TCH001", "Mathematics")
    env.add_teacher("Prof. James Kim", "TCH002", "Computer Science")

    students = [
        env.add_student("Alice Chen",    "STU001"),
        env.add_student("Bob Martinez",  "STU002"),
        env.add_student("Clara Okonkwo", "STU003"),
        env.add_student("David Park",    "STU004"),
    ]

    # Include actors in their respective contexts
    for actor_dict in [admin, teacher]:
        for ctx in [env.admin_ctx, env.acad_ctx, env.enrl_ctx]:
            ctx.include(actor_dict)

    tag("BOOT", f"{len(env._students)+len(env._teachers)+len(env._admins)} elements seeded")
    print()

    # ── Build AI agent (Admin role) ──────────────────────────────────────

    agent_admin   = MepAgent(env, ollama, admin,   demo_mode=demo_mode)
    agent_teacher = MepAgent(env, ollama, teacher, demo_mode=demo_mode)

    # ═════════════════════════════════════════════════════════════════════
    # PHASE 1 — CONTEXT ENVELOPE SHOWCASE
    # The AI receives a structured semantic frame, not raw text
    # ═════════════════════════════════════════════════════════════════════

    section("PHASE 1 — Context Envelope (what the AI sees)")

    envelope = env.get_envelope(admin, env.admin_ctx)

    print(f"\n{C.BOLD}Raw ContextEnvelope delivered to AI:{C.RESET}")
    print(f"  Context:     {envelope.name} ({envelope.kind})")
    print(f"  Depth:       {envelope.depth}")
    print(f"  Data keys:   {list(envelope.data.keys())}")
    print(f"\n  Available Actions (ranked by CAUSAL GRAVITY):")
    for a in envelope.available_actions:
        bar  = "█" * int(a["gravity"] * 10)
        print(f"    [{bar:<10}] {a['gravity']:.3f}  {a['type']}")
        print(f"               {C.DIM}{a['description']}{C.RESET}")

    print(f"\n{C.BOLD}Rendered as structured prompt for LLM:{C.RESET}")
    print(C.DIM + textwrap.indent(envelope.to_prompt(), "    ") + C.RESET)

    # ═════════════════════════════════════════════════════════════════════
    # PHASE 2 — AI AGENT: Administrative workflow
    # ═════════════════════════════════════════════════════════════════════

    section("PHASE 2 — AI Agent: Administrative Setup")

    # Agent creates a school
    await agent_admin.act("Create a new school institution called Global Science Academy",
                           env.admin_ctx,
                           extra_payload={"name":"Global Science Academy",
                                          "code":"GSA","countryCode":"FR"})
    print()

    # Agent creates courses
    for course_data in [
        {"name":"Advanced Mathematics","code":"MATH201","discipline":"Mathematics",
         "capacity":"25","credits":"4"},
        {"name":"Data Structures","code":"CS301","discipline":"Computer Science",
         "capacity":"30","credits":"3"},
        {"name":"Physics Lab","code":"PHY101","discipline":"Physics",
         "capacity":"20","credits":"3"},
    ]:
        await agent_admin.act(f"Create course {course_data['name']}",
                               env.admin_ctx, extra_payload=course_data)

    print()
    tag("STATUS", f"Courses registered: {len(env._courses)}")
    course_ids = list(env._courses.keys())

    # ═════════════════════════════════════════════════════════════════════
    # PHASE 3 — CIRCUMSTANCE GATING: enrollment.open = false → blocked
    # ═════════════════════════════════════════════════════════════════════

    section("PHASE 3 — Circumstance Gating Demo")

    print(f"{C.BOLD}Testing: enrollment.open = FALSE{C.RESET}")
    print(f"The AI can see the enrollment context, but the 'enrollment.open'")
    print(f"circumstance is FALSE — actions gated by it are INVISIBLE.\n")

    env_blocked = env.get_envelope(admin, env.enrl_ctx)
    print(f"  Available in EnrollmentContext (enrollment closed):")
    for a in env_blocked.available_actions:
        print(f"    {C.RED}✗{C.RESET} {a['type']}")
    if not env_blocked.available_actions:
        print(f"    {C.RED}(none — enrollment.open circumstance blocks all){C.RESET}")
    print()

    # Open enrollment
    await agent_admin.act("Open the enrollment period for students", env.admin_ctx)
    print()

    env_open = env.get_envelope(admin, env.enrl_ctx)
    print(f"  Available in EnrollmentContext (enrollment OPEN):")
    for a in env_open.available_actions:
        print(f"    {C.GREEN}✓{C.RESET} {a['type']} (gravity={a['gravity']:.3f})")

    # ═════════════════════════════════════════════════════════════════════
    # PHASE 4 — AI AGENT: Enrollment operations
    # ═════════════════════════════════════════════════════════════════════

    section("PHASE 4 — AI Agent: Student Enrollments")

    if course_ids:
        math_id = course_ids[0]
        cs_id   = course_ids[1] if len(course_ids) > 1 else course_ids[0]

        for student in students[:3]:
            await agent_admin.act(
                f"Enroll student {student['name']} in Advanced Mathematics",
                env.enrl_ctx,
                extra_payload={"studentId": student["id"], "courseId": math_id})

        for student in students[:2]:
            await agent_admin.act(
                f"Enroll student {student['name']} in Data Structures",
                env.enrl_ctx,
                extra_payload={"studentId": student["id"], "courseId": cs_id})

    print()
    math_course = env._courses.get(course_ids[0], {})
    tag("STATUS", f"Math: {math_course.get('enrolled',0)} enrolled | "
                  f"CS: {env._courses.get(course_ids[1] if len(course_ids)>1 else course_ids[0],{}).get('enrolled',0)} enrolled")

    # ═════════════════════════════════════════════════════════════════════
    # PHASE 5 — TEACHER AGENT: Academic operations + chain reactions
    # ═════════════════════════════════════════════════════════════════════

    section("PHASE 5 — Teacher AI Agent: Grading + Causal Chains")

    await agent_admin.act("Close enrollment period", env.admin_ctx)
    print()

    print(f"{C.BOLD}Teacher agent recording grades...{C.RESET}")
    print(f"Failing grades spawn CHAIN REACTION → notification.dispatch\n")

    grades_to_record = [
        (students[0]["id"], course_ids[0], 88.5, "Alice"),
        (students[1]["id"], course_ids[0], 42.0, "Bob   ← FAILING → chain reaction"),
        (students[2]["id"], course_ids[0], 55.0, "Clara ← FAILING → chain reaction"),
        (students[0]["id"], course_ids[1] if len(course_ids)>1 else course_ids[0], 93.0, "Alice"),
        (students[1]["id"], course_ids[1] if len(course_ids)>1 else course_ids[0], 38.0, "Bob   ← FAILING → chain reaction"),
    ]

    for sid, cid, grade, label in grades_to_record:
        print(f"  Recording grade for {label}...")
        await agent_teacher.act(
            f"Record grade {grade} for student",
            env.acad_ctx,
            extra_payload={"studentId":sid, "courseId":cid, "grade":str(grade)})

    # ═════════════════════════════════════════════════════════════════════
    # PHASE 6 — WHY / WHY-NOT EXPLAINABILITY
    # The AI queries the causal graph
    # ═════════════════════════════════════════════════════════════════════

    section("PHASE 6 — WHY / WHY-NOT Explainability")

    print(f"{C.BOLD}AI queries: why did 'grade.record' succeed?{C.RESET}\n")
    explanation = env.explain("grade.record")
    print(C.DIM + textwrap.indent(explanation, "  ") + C.RESET)

    print(f"\n{C.BOLD}Causal graph statistics:{C.RESET}")
    stats = env.causal_stats
    for k, v in stats.items():
        if k != "action_types":
            print(f"  {k}: {v}")
    print(f"  action_types: [{', '.join(stats.get('action_types',[]))}]")

    # ═════════════════════════════════════════════════════════════════════
    # PHASE 7 — CONTEXT TOPOLOGY — semantic distance between contexts
    # ═════════════════════════════════════════════════════════════════════

    section("PHASE 7 — Context Topology (Semantic Distance)")

    pairs = [
        (env.admin_ctx, env.acad_ctx,  "Administrative ↔ Academic"),
        (env.admin_ctx, env.enrl_ctx,  "Administrative ↔ Enrollment"),
        (env.acad_ctx,  env.enrl_ctx,  "Academic       ↔ Enrollment"),
    ]

    print(f"{'Pair':<40} {'Distance':>10}  {'Proximity'}")
    print(f"{'─'*40} {'─'*10}  {'─'*20}")
    for ctx_a, ctx_b, label in pairs:
        aff  = ctx_a.basis.affinity(ctx_b.basis)
        dist = math.acos(max(-1, min(1, aff))) / math.pi
        bar  = "█" * int((1 - dist) * 20)
        print(f"{label:<40} {dist:>10.3f}  {bar}")

    print(f"\n{C.DIM}0.0 = identical meaning | 1.0 = orthogonal (opposite axes){C.RESET}")
    print(f"\nActions in the Academic context ranked by gravity:")
    for a, g in env.acad_ctx.get_available_actions(teacher, {"actor_id":teacher["id"]}):
        bar = "█" * int(g * 20)
        print(f"  [{bar:<20}] {g:.3f}  {a.type}")

    # ═════════════════════════════════════════════════════════════════════
    # PHASE 8 — SYSTEM SNAPSHOT via AI agent
    # ═════════════════════════════════════════════════════════════════════

    section("PHASE 8 — Final State Snapshot")

    snapshot_reaction = await agent_admin.act("Take a snapshot of the system state",
                                               env.admin_ctx)
    if snapshot_reaction and snapshot_reaction.result:
        s = snapshot_reaction.result
        print(f"\n  {C.BOLD}Environment State:{C.RESET}")
        for k, v in s.items():
            print(f"    {k:<25}: {v}")

    # Student summary
    print(f"\n  {C.BOLD}Student GPAs:{C.RESET}")
    for student in students:
        sid = student["id"]
        gpa = env._students.get(sid, {}).get("gpa", 0)
        enrolled = env._students.get(sid, {}).get("enrolledCourses", [])
        print(f"    {student['name']:<20}: GPA={gpa:.2f}  |  {len(enrolled)} course(s)")

    # Phenomena
    if env._phenomena.all_detected:
        print(f"\n  {C.BOLD}Phenomena Detected:{C.RESET}")
        for p in env._phenomena.all_detected:
            print(f"    {C.MAGENTA}★ {p['name']}{C.RESET} — "
                  f"magnitude={p['magnitude']:.0%}, count={p['count']}")

    # Agent decision history
    print(f"\n  {C.BOLD}Agent Decision History:{C.RESET}")
    for h in agent_admin._decision_history + agent_teacher._decision_history:
        icon = C.GREEN + "✓" if h["status"] == "success" else C.RED + "✗"
        print(f"    {icon}{C.RESET} {h['action']:<35} ← {C.DIM}{h['goal'][:50]}{C.RESET}")

    # Final
    hr()
    print(f"\n  {C.BOLD}{C.CYAN}MEP/2.0 demonstration complete.{C.RESET}")
    print(f"  The AI never called a raw API. It operated through:")
    print(f"    {C.GREEN}• Context envelopes  (structured semantic frames){C.RESET}")
    print(f"    {C.GREEN}• Causal gravity     (action ranking by affinity){C.RESET}")
    print(f"    {C.GREEN}• Live circumstances (boolean gates, composable){C.RESET}")
    print(f"    {C.GREEN}• Causal graph       (full decision traceability){C.RESET}")
    print(f"    {C.GREEN}• Phenomenon engine  (emergent pattern detection){C.RESET}")
    hr()

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MEP/2.0 × Ollama Agent")
    parser.add_argument("--demo",  action="store_true", help="Force demo mode (no Ollama)")
    parser.add_argument("--model", default="llama3",    help="Ollama model name")
    parser.add_argument("--host",  default="http://localhost:11434", help="Ollama host")
    args = parser.parse_args()

    asyncio.run(main(demo_mode=args.demo, model=args.model))
