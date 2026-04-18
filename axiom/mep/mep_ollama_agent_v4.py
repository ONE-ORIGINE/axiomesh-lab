"""
mep_ollama_agent_v4.py  —  MEP Semantic Intelligence Layer
════════════════════════════════════════════════════════════════════════════
OneOrigine / ImperialSchool Research  —  I.S. License

V4 SEMANTIC INNOVATIONS:
  • Harmony-guided intent translation (not just keyword matching)
  • Causal Δ tracking: records Δ_t = φ(Σ_{t+1}) − φ(Σ_t) per action
  • Situation-as-sense: environment situation converted to SenseVector
    and used to bias action selection via H(A,C,S_situation)
  • Multi-context navigation: AI selects WHICH context to operate in
    based on goal semantic distance to context basis vectors
  • Phenomenon-as-signal: active phenomena modulate current_sense,
    pulling AI attention toward the right semantic dimension
  • Contrastive memory: agent learns which (action,context) pairs
    succeeded vs failed — improves future harmony estimates

Usage:
  python mep_ollama_agent_v4.py --demo
  python mep_ollama_agent_v4.py --model gemma3:12b --inject-memory --inject-sit-tone
  python mep_ollama_agent_v4.py --provider openai --model gpt-4o-mini
  python mep_ollama_agent_v4.py --provider anthropic --model claude-3-haiku-20240307
"""

from __future__ import annotations

import argparse, asyncio, json, math, sys, textwrap, time, traceback, uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from edp import (
    Environment, Context, Action, Reaction, Element, Circumstance,
    SenseVector, HarmonyProfile, PhenomenonPattern,
    EnvironmentKind, ContextKind, ActionCategory,
    ReactionStatus, ImpactScope, Temporality, RawData,
    SENSE_NULL, compute_harmony
)
from mep import (
    MepGateway, MepAgent, MepSession, MepParser,
    OllamaProvider, OpenAIProvider, AnthropicProvider, make_provider,
    CircuitBreaker, LlmError, ContextEnvelope
)

# ─── ANSI ─────────────────────────────────────────────────────────────────────

class C:
    R="\033[0m"; B="\033[1m"; D="\033[2m"
    CY="\033[96m"; G="\033[92m"; Y="\033[93m"; RD="\033[91m"; MG="\033[95m"

def hr(n=70,ch="═",c=C.D): print(f"{c}{ch*n}{C.R}")
def section(t): hr(); print(f"{C.B}{C.CY}  {t}{C.R}"); hr()
def ok(l,m):   print(f"  {C.G}[{l}]{C.R} ✓ {m}")
def err(l,m):  print(f"  {C.RD}[{l}]{C.R} ✗ {m}")
def warn(l,m): print(f"  {C.Y}[{l}]{C.R} ! {m}")
def info(l,m): print(f"  {C.CY}[{l}]{C.R} → {m}")
def dim(l,m):  print(f"  {C.D}[{l}]{C.R}   {m}")

# ═════════════════════════════════════════════════════════════════════════════
# SEMANTIC INTELLIGENCE LAYER
# V4 innovations: harmony-guided, situation-as-sense, Δ tracking
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ContrastiveRecord:
    """
    Stores (action_type, context_name, goal_sense) → success/failure.
    Used to calibrate future harmony estimates.
    """
    action_type   : str
    context_name  : str
    goal_sense    : SenseVector
    action_sense  : SenseVector
    harmony       : float
    success       : bool
    at            : float = field(default_factory=time.time)


class SemanticIntelligenceLayer:
    """
    V4 innovation: a learned semantic layer over MEP dispatch.

    Capabilities:
    1. translate_goal_to_sense()   — convert natural-language goal to SenseVector
    2. select_best_context()       — choose context nearest to goal sense
    3. situation_as_sense()        — convert environment situation to SenseVector
       for harmony computation: H(A,C,S_situation)
    4. phenomenon_modulated_sense()— adjust sense based on active phenomena
    5. contrastive_memory_update() — record success/failure per action
    6. estimate_harmony_adjustment()— adjust future harmony based on memory
    """

    # Keyword → semantic axis mapping for goal translation
    _GOAL_AXES: Dict[str, SenseVector] = {
        "create":    SenseVector.technical("creation intent",    0.8),
        "enroll":    SenseVector.temporal("enrollment intent",   0.9),
        "grade":     SenseVector.normative("grading intent",     0.95),
        "withdraw":  SenseVector.temporal("withdrawal intent",   0.7),
        "snapshot":  SenseVector.technical("observation intent", 0.4),
        "list":      SenseVector.technical("listing intent",     0.3),
        "assign":    SenseVector.social("assignment intent",     0.7),
        "open":      SenseVector.temporal("activation intent",   0.8),
        "close":     SenseVector.temporal("deactivation intent", 0.7),
        "query":     SenseVector.normative("query intent",       0.5),
        "notify":    SenseVector.social("notification intent",   0.6),
        "stabilize": SenseVector.causal("stabilization intent",  0.9),
    }

    _SITUATION_SENSES: Dict[str, SenseVector] = {
        "operational": SenseVector.technical("nominal ops",      0.5),
        "degraded":    SenseVector.causal("degradation signal",  0.8),
        "critical":    SenseVector.causal("critical signal",     1.0),
    }

    def __init__(self):
        self._contrastive: List[ContrastiveRecord] = []
        self._harmony_adjustments: Dict[str, float] = {}  # action_type → Δ

    def translate_goal_to_sense(self, goal: str) -> SenseVector:
        """
        Ψ_goal(goal) → SenseVector  — goal as semantic position.
        Finds the closest axis keywords in the goal, blends their vectors.
        """
        goal_lower = goal.lower()
        matched = [(kw, sv) for kw, sv in self._GOAL_AXES.items()
                   if kw in goal_lower]
        if not matched:
            return SenseVector.causal("unknown goal", 0.3)

        # Blend matched sense vectors (mean of components)
        n = len(matched)
        blended = tuple(
            sum(sv.v[i] for _, sv in matched) / n
            for i in range(8))
        mag = math.sqrt(sum(x*x for x in blended)) or 1.0
        normed = tuple(x/mag for x in blended)
        return SenseVector("blended", f"goal:{goal[:30]}", 1.0, normed)

    def select_best_context(self, goal_sense: SenseVector,
                             contexts: List[Context]) -> Optional[Context]:
        """
        c* = argmin_c angular_distance(goal_sense, c.basis)
        Choose the context whose basis is semantically closest to the goal.
        """
        if not contexts: return None
        return min(contexts, key=lambda c: goal_sense.angular_distance(c.basis))

    def situation_as_sense(self, situation: Dict) -> SenseVector:
        """
        Convert environment situation to SenseVector for H(A,C,S_situation).
        Critical situations pull AI toward causal axis (stabilization priority).
        """
        kind = situation.get("kind", "operational")
        return self._SITUATION_SENSES.get(kind, self._SITUATION_SENSES["operational"])

    def phenomenon_modulated_sense(self, base_sense: SenseVector,
                                    phenomena: List[Dict]) -> SenseVector:
        """
        Modulate current sense based on active phenomena.
        High-magnitude phenomena pull sense toward emergent axis.
        """
        if not phenomena: return base_sense
        max_mag = max(p.get("magnitude",0) for p in phenomena if not p.get("dissolved"))
        if max_mag < 0.3: return base_sense

        # Blend toward emergent axis proportional to phenomenon magnitude
        emergent = SenseVector.emergent("phenomenon signal", max_mag)
        alpha    = 1.0 - max_mag * 0.5  # 0.5–1.0 weight on original
        blended  = tuple(
            alpha*a + (1-alpha)*b
            for a, b in zip(base_sense.v, emergent.v))
        mag = math.sqrt(sum(x*x for x in blended)) or 1.0
        normed = tuple(x/mag for x in blended)
        return SenseVector("modulated", f"{base_sense.meaning}+phenom",
                           1.0, normed)

    def record_outcome(self, action_type: str, context_name: str,
                        goal_sense: SenseVector, action_sense: SenseVector,
                        harmony: float, success: bool):
        """Contrastive memory update — learn from outcomes."""
        self._contrastive.append(ContrastiveRecord(
            action_type, context_name, goal_sense, action_sense, harmony, success))
        if len(self._contrastive) > 200:
            self._contrastive = self._contrastive[-200:]
        # Update harmony adjustment for this action
        recent = [r for r in self._contrastive[-20:] if r.action_type==action_type]
        if len(recent) >= 3:
            success_rate = sum(1 for r in recent if r.success) / len(recent)
            self._harmony_adjustments[action_type] = (success_rate - 0.5) * 0.2

    def harmony_adjustment(self, action_type: str) -> float:
        """δH from contrastive memory — positive = boost, negative = penalty."""
        return self._harmony_adjustments.get(action_type, 0.0)

    def rank_actions_semantically(self, available: List[Tuple],
                                   goal_sense: SenseVector,
                                   env_sense: SenseVector) -> List[Tuple]:
        """
        Re-rank available actions using full semantic profile:
        H_total = H_base + α·cos(φ_A, φ_goal) + β·cos(φ_A, φ_env) + δH_memory
        """
        ranked = []
        for a, h in available:
            goal_align  = a.sense.cosine(goal_sense)
            env_align   = a.sense.cosine(env_sense)
            mem_delta   = self.harmony_adjustment(a.type)
            H_total     = h.score + 0.15*goal_align + 0.10*env_align + mem_delta
            ranked.append((a, h, H_total, goal_align))

        ranked.sort(key=lambda x: -x[2])
        return [(a, h) for a, h, _, _ in ranked]


# ═════════════════════════════════════════════════════════════════════════════
# ESCHOOL ENVIRONMENT — EDP application
# ═════════════════════════════════════════════════════════════════════════════

class ESchoolElement(Element):
    """Domain element with EDP property discipline."""
    def __init__(self, name: str, etype: str, code: str,
                 sense: SenseVector = None, **stable_props):
        super().__init__(name, etype, sense)
        self.set_stable("code", code)
        self.set_stable("element_type_label", etype)
        for k, v in stable_props.items(): self.set_stable(k, v)

    async def on_impacted(self, reaction: Reaction, frame: Dict):
        # Domain-specific impact handlers
        rt = reaction.action_type
        if rt == "enrollment.confirmed" or rt == "student.enroll":
            cid = frame.get("payload", {}).get("courseId","")
            if self.element_type == "Student" and cid:
                enrolled = self.get("enrolledCourses", [])
                if cid not in enrolled:
                    self.set_dynamic("enrolledCourses", enrolled+[cid])
        elif rt == "grade.record":
            if reaction.result and self.element_type == "Student":
                grade = reaction.result.get("grade", 0)
                gpa = self.get("gpa", 0.0)
                n   = self.get("gradeCount", 0)
                new_gpa = (gpa * n + grade) / (n + 1)
                self.set_dynamic("gpa", round(new_gpa, 2))
                self.set_dynamic("gradeCount", n+1)


def build_eschool() -> Tuple[Environment, Dict[str, Context], Dict[str, ESchoolElement]]:
    """Build the full ESchool environment with all contexts, circumstances, actions."""

    env = Environment("ESchool", EnvironmentKind.REACTIVE)

    # ── Circumstances ──────────────────────────────────────────────────────
    C_SYS    = Circumstance.always("system.active")
    C_ENRL   = Circumstance.flag("enrollment.open","Enrollment period active","enrollmentOpen")
    C_NOT_ENRL = ~C_ENRL

    # ── Contexts ───────────────────────────────────────────────────────────
    admin_ctx = env.create_context("ESchool.Admin", ContextKind.SEMANTIC,
        basis=SenseVector.normative("administrative operations", 0.9),
        circumstances=[C_SYS])

    acad_ctx = env.create_context("ESchool.Academic", ContextKind.SEMANTIC,
        basis=SenseVector.normative("academic assessment", 0.95),
        circumstances=[C_SYS])

    enrl_ctx = env.create_context("ESchool.Enrollment", ContextKind.TEMPORAL,
        basis=SenseVector.temporal("enrollment operations", 0.9),
        circumstances=[C_SYS, C_ENRL])

    # ── Storage ────────────────────────────────────────────────────────────
    elements: Dict[str, ESchoolElement] = {}
    courses:  Dict[str, Dict] = {}
    grades:   List[Dict] = []
    gdata     = {"enrollmentOpen": False}

    def sync_flag(key: str, val):
        gdata[key] = val
        for ctx in [admin_ctx, acad_ctx, enrl_ctx]:
            ctx.set(key, val)

    # ── Action factories ───────────────────────────────────────────────────

    async def h_create_school(actor, p, ctx, frame):
        n=p.get("name","School"); code=p.get("code","SCH")
        sid=str(uuid.uuid4())
        el=ESchoolElement(n,"School",code,SenseVector.social(n,0.7),name=n)
        el.set_stable("name",n)
        await env.admit(el); elements[sid]=el
        return Reaction.ok("school.create",f"School '{n}' created",
            result={"id":sid,"name":n,"code":code},
            sense=SenseVector.normative("school created",0.7),
            impact=ImpactScope.on_env())

    async def h_create_course(actor, p, ctx, frame):
        n=p.get("name"); cap=int(p.get("capacity",30))
        if not n: return Reaction.reject("course.create","Missing 'name'")
        cid=str(uuid.uuid4())
        courses[cid]={"id":cid,"name":n,"code":p.get("code","C"+cid[:4]),
                      "discipline":p.get("discipline","General"),
                      "maxCapacity":cap,"credits":int(p.get("credits",3)),
                      "enrolled":0,"students":[],"grades":[]}
        return Reaction.ok("course.create",f"Course '{n}' cap={cap}",
            result={"id":cid,"name":n,"cap":cap},
            sense=SenseVector.technical("course provisioned",0.6),
            impact=ImpactScope.on_env())

    async def h_open_enrl(actor, p, ctx, frame):
        sync_flag("enrollmentOpen", True)
        return Reaction.ok("enrollment.open","Enrollment opened",
            sense=SenseVector.temporal("enrollment activated",0.9),
            impact=ImpactScope.on_env())

    async def h_close_enrl(actor, p, ctx, frame):
        sync_flag("enrollmentOpen", False)
        return Reaction.ok("enrollment.close","Enrollment closed",
            sense=SenseVector.temporal("enrollment deactivated",0.7),
            impact=ImpactScope.on_env())

    async def h_enroll(actor, p, ctx, frame):
        sid=p.get("studentId",""); cid=p.get("courseId","")
        s=elements.get(sid); c=courses.get(cid)
        if not s: return Reaction.reject("student.enroll","Student not found")
        if not c: return Reaction.reject("student.enroll","Course not found")
        if c["enrolled"]>=c["maxCapacity"]:
            return Reaction.reject("student.enroll",f"At capacity ({c['maxCapacity']})")
        if sid in c["students"]: return Reaction.reject("student.enroll","Already enrolled")
        c["students"].append(sid); c["enrolled"]+=1
        prev_Σ = SenseVector.temporal("pre-enrollment",float(c["enrolled"]-1)/c["maxCapacity"])
        new_Σ  = SenseVector.temporal("post-enrollment",float(c["enrolled"])/c["maxCapacity"])
        delta  = prev_Σ.delta(new_Σ)
        r = Reaction.ok("student.enroll",
            f"{s.name} enrolled in {c['name']}",
            result={"student":s.name,"course":c["name"]},
            sense=SenseVector.temporal("enrollment confirmed",0.95),
            impact=ImpactScope.on_element(cid, 0.8))
        r.causal_delta = delta  # Δ_t = φ(Σ_{t+1}) - φ(Σ_t)
        return r

    async def h_grade(actor, p, ctx, frame):
        sid=p.get("studentId",""); cid=p.get("courseId","")
        try: g=float(p.get("grade",0))
        except: return Reaction.reject("grade.record","Invalid grade")
        s=elements.get(sid); c=courses.get(cid)
        if not s: return Reaction.reject("grade.record","Student not found")
        if not c: return Reaction.reject("grade.record","Course not found")
        if not (0<=g<=100): return Reaction.reject("grade.record","Grade 0-100 required")
        lt="A"if g>=90 else"B"if g>=80 else"C"if g>=70 else"D"if g>=60 else"F"
        rec={"s":sid,"c":cid,"g":g,"l":lt,"at":time.time()}
        grades.append(rec); c["grades"].append(rec)
        # Student GPA update via on_impacted
        passing=g>=60
        chain=[] if passing else ["notification.dispatch"]
        r = Reaction.ok("grade.record",f"{s.name}: {g:.1f} ({lt})",
            result={"grade":g,"letter":lt,"passing":passing},
            sense=SenseVector.normative("grade assessed",float(g/100)),
            impact=ImpactScope.on_element(cid,0.7),
            chain=chain)
        return r

    async def h_snapshot(actor, p, ctx, frame):
        return Reaction.ok("system.snapshot","Snapshot",
            result={"schools":sum(1 for e in elements.values() if e.element_type=="School"),
                    "courses":len(courses),"students":sum(1 for e in elements.values() if e.element_type=="Student"),
                    "grades":len(grades),"enrollmentOpen":gdata.get("enrollmentOpen"),
                    "phenomena":sum(1 for p in env.phenomena if p.is_active)})

    async def h_list(actor, p, ctx, frame):
        return Reaction.ok("system.list","Elements",
            result={"students":[e.to_dict() for e in elements.values() if e.element_type=="Student"],
                    "courses":list(courses.values())})

    # Register actions in contexts
    for ctx_obj, acts in [
        (admin_ctx, [
            Action("school.create","lifecycle" if False else ActionCategory.LIFECYCLE,"Create school",
                   SenseVector.normative("school creation",0.9),handler=h_create_school),
            Action("course.create",ActionCategory.LIFECYCLE,"Create course",
                   SenseVector.technical("course provisioning",0.8),handler=h_create_course),
            Action("enrollment.open",ActionCategory.COMMAND,"Open enrollment",
                   SenseVector.temporal("enrollment activation",0.85),handler=h_open_enrl),
            Action("enrollment.close",ActionCategory.COMMAND,"Close enrollment",
                   SenseVector.temporal("enrollment deactivation",0.7),handler=h_close_enrl),
            Action("system.snapshot",ActionCategory.QUERY,"System snapshot",
                   SenseVector.technical("system state",0.3),handler=h_snapshot),
            Action("system.list",ActionCategory.QUERY,"List elements",
                   SenseVector.technical("element listing",0.2),handler=h_list),
        ]),
        (acad_ctx, [
            Action("grade.record",ActionCategory.COMMAND,"Record student grade",
                   SenseVector.normative("academic assessment",0.95),handler=h_grade),
            Action("system.snapshot",ActionCategory.QUERY,"System snapshot",
                   SenseVector.technical("system state",0.3),handler=h_snapshot),
        ]),
        (enrl_ctx, [
            Action("student.enroll",ActionCategory.COMMAND,"Enroll student in course",
                   SenseVector.temporal("enrollment",0.95),
                   guards=[Circumstance.flag("enrollment.open","Enrollment active","enrollmentOpen")],
                   handler=h_enroll),
            Action("system.snapshot",ActionCategory.QUERY,"System snapshot",
                   SenseVector.technical("system state",0.3),handler=h_snapshot),
        ]),
    ]:
        for a in acts: ctx_obj.reg(a)

    # Phenomena
    env.register_pattern(PhenomenonPattern(
        "MassFailure", "fail", threshold=3, window_s=600,
        attractor=SenseVector.emergent("academic failure cascade")))

    env.on_reaction(lambda r: None)  # silent base listener

    return env, {"admin":admin_ctx,"acad":acad_ctx,"enrl":enrl_ctx}, elements


# ═════════════════════════════════════════════════════════════════════════════
# MAIN DEMONSTRATION
# ═════════════════════════════════════════════════════════════════════════════

async def main(args):
    hr(70,"╔"); hr(70,"║")
    print(f"  {C.B}{C.CY}MEP v3 × EDP v4.1  —  Semantic Intelligence Layer  (V4){C.R}")
    hr(70,"║"); hr(70,"╚"); print()

    # ── Build environment ────────────────────────────────────────────────
    env, ctxs, elements = build_eschool()
    gw = MepGateway(env)

    # ── Provider ─────────────────────────────────────────────────────────
    llm = make_provider(args.provider, args.model, args.host, args.timeout, args.retries)
    demo = args.demo or not llm.available
    if not demo: info("LLM", f"{llm.provider_name()}  timeout={args.timeout}s")
    else:        warn("LLM", "Demo mode — deterministic decisions")
    if args.inject_memory: info("MEMORY", "Agent memory injection ON")
    else: dim("MEMORY", "Injection OFF (small-model safe)")

    # ── Semantic intelligence layer ───────────────────────────────────────
    sil = SemanticIntelligenceLayer()

    # ── Admit actors ─────────────────────────────────────────────────────
    admin   = ESchoolElement("Dr. Vasquez","Admin","ADM001",SenseVector.normative("admin",0.9))
    teacher = ESchoolElement("Prof. Laurent","Teacher","TCH001",SenseVector.normative("teacher",0.8))
    stus = [
        ESchoolElement(n,"Student",c,SenseVector.social(n,0.5))
        for n,c in [("Alice Chen","STU001"),("Bob Martinez","STU002"),("Clara O.","STU003")]
    ]
    for el in [admin,teacher]+stus:
        await env.admit(el)
        elements[el.element_id] = el

    for ctx in ctxs.values():
        ctx.include(admin.to_dict())
        ctx.include(teacher.to_dict())

    # Sessions
    s_admin   = gw.connect("admin-agent")
    s_teacher = gw.connect("teacher-agent")

    ag_admin = MepAgent(gw, llm, admin, s_admin, demo, "ADMIN",
                         inject_memory=args.inject_memory,
                         inject_sit_tone=args.inject_sit_tone)
    ag_teach = MepAgent(gw, llm, teacher, s_teacher, demo, "TEACH",
                         inject_memory=args.inject_memory,
                         inject_sit_tone=args.inject_sit_tone)

    ok("BOOT", f"{len(env.elements)} elements admitted"); print()

    # ═══ PHASE 1: Semantic Context Selection ══════════════════════════════
    section("PHASE 1 — Semantic Context Selection (V4 Innovation)")

    goal = "Create a course in Mathematics for 25 students"
    goal_sense = sil.translate_goal_to_sense(goal)
    print(f"\n  Goal: '{goal}'")
    print(f"  Goal sense: φ({goal_sense.dimension}:{goal_sense.meaning}  |{goal_sense.magnitude:.2f}|)")

    best_ctx = sil.select_best_context(goal_sense, env._contexts)
    print(f"  Best context for this goal: {best_ctx.name if best_ctx else 'None'}")
    print(f"\n  Context topology (angular distances to goal sense):")
    for ctx in env._contexts:
        d = goal_sense.angular_distance(ctx.basis)
        bar = "█"*int((1-d)*20)
        print(f"    [{bar:<20}] {d:.3f}  {ctx.name}")

    # ═══ PHASE 2: Harmony Map ════════════════════════════════════════════
    section("PHASE 2 — Full Harmony Map H(A,C,S)")

    env_situation = env.snapshot()
    sit_sense = sil.situation_as_sense({"kind": env_situation.get("situation","operational")})
    mod_sense = sil.phenomenon_modulated_sense(goal_sense, [])

    print(f"\n  Situation sense: φ({sit_sense.dimension}:{sit_sense.meaning})")
    print(f"\n  Action harmony in Admin context (H = α·ctx + β·sem − δ·dissonance):")
    admin_frame = {"actor_id": admin.element_id}
    available = ctxs["admin"].get_available_actions(admin.to_dict(), admin_frame, mod_sense)
    for a, h in available:
        print(f"    H={h.score:+.4f}  ctx={h.context_alignment:.3f}  "
              f"sem={h.semantic_alignment:.3f}  dis={h.dissonance:.3f}  "
              f"→ {a.type}")

    # Re-rank with SIL
    re_ranked = sil.rank_actions_semantically(available, goal_sense, sit_sense)
    print(f"\n  After semantic re-ranking (H_total with goal+env):")
    for a, h in re_ranked[:4]:
        print(f"    H_base={h.score:+.4f}  → {a.type}")

    # ═══ PHASE 3: Administrative Setup ════════════════════════════════════
    section("PHASE 3 — Administrative Agent: Setup")

    r = await ag_admin.act("Create school Global Science Academy", ctxs["admin"],
        extra={"name":"Global Science Academy","code":"GSA","countryCode":"FR"},
        sense=sil.translate_goal_to_sense("create school"))
    sil.record_outcome("school.create","ESchool.Admin",
        goal_sense,SenseVector.normative("school creation",0.9),0.9,r.is_success if r else False)

    course_ids = []
    for spec in [
        {"name":"Advanced Mathematics","code":"MATH201","discipline":"Mathematics","capacity":"25","credits":"4"},
        {"name":"Data Structures","code":"CS301","discipline":"Computer Science","capacity":"30","credits":"3"},
    ]:
        gs = sil.translate_goal_to_sense(f"create course {spec['name']}")
        r  = await ag_admin.act(f"Create course {spec['name']}", ctxs["admin"],
                                 extra=spec, sense=gs)
        if r and r.is_success and r.result:
            course_ids.append(r.result.get("id",""))
            sil.record_outcome("course.create","ESchool.Admin",gs,
                SenseVector.technical("course provisioning",0.8),0.8,True)

    ok("STATUS", f"{len(course_ids)} courses created")

    # ═══ PHASE 4: Circumstance Gating ════════════════════════════════════
    section("PHASE 4 — Circumstance Gating + WHY-NOT")

    print(f"\n{C.B}Enrollment context, enrollmentOpen=False:{C.R}")
    env_closed = gw.build_envelope(s_admin, admin, ctxs["enrl"])
    if not env_closed.actions:
        print(f"  {C.RD}Zero actions — enrollment.open=False makes them invisible{C.R}")

    # WHY-NOT query
    enrl_action = next((e["a"] for e in ctxs["enrl"]._actions
                        if e["a"].type=="student.enroll"), None)
    if enrl_action:
        why_not = gw.explain_why_not(admin, "student.enroll", ctxs["enrl"])
        print(f"  WHY-NOT student.enroll: {why_not}")

    # Open enrollment
    await ag_admin.act("Open enrollment period", ctxs["admin"],
        sense=sil.translate_goal_to_sense("open enrollment"))
    print(f"\n{C.B}After enrollment.open:{C.R}")
    for a in gw.build_envelope(s_admin, admin, ctxs["enrl"]).actions:
        print(f"  {C.G}✓{C.R} {a['type']}  H={a.get('score',0):.3f}")

    # ═══ PHASE 5: Enrollment with Causal Δ tracking ═══════════════════════
    section("PHASE 5 — Enrollment + Causal Δ Tracking")

    print(f"  {C.D}Δ_t = φ(Σ_{{t+1}}) − φ(Σ_t) recorded per enrollment{C.R}\n")
    if course_ids:
        mid = course_ids[0]; cid2 = course_ids[1] if len(course_ids)>1 else course_ids[0]
        for s in stus:
            gs = sil.translate_goal_to_sense(f"enroll {s.name}")
            r = await ag_admin.act(f"Enroll {s.name} in Math", ctxs["enrl"],
                extra={"studentId":s.element_id,"courseId":mid}, sense=gs)
            if r and r.causal_delta:
                print(f"  Δ_t after enrolling {s.name}: "
                      f"φ({r.causal_delta.dimension}) |Δ|={r.causal_delta.magnitude:.4f}")

    # ═══ PHASE 6: Teacher + Phenomena ═════════════════════════════════════
    section("PHASE 6 — Teacher Agent: Grading + Phenomenon Emergence")

    await ag_admin.act("Close enrollment", ctxs["admin"],
        sense=sil.translate_goal_to_sense("close enrollment"))

    grade_data = [
        (stus[0].element_id, course_ids[0] if course_ids else "", 88.0),
        (stus[1].element_id, course_ids[0] if course_ids else "", 42.0),
        (stus[2].element_id, course_ids[0] if course_ids else "", 37.0),
    ]
    for sid, cid, g in grade_data:
        s_el = elements.get(sid)
        if not s_el: continue
        gs = sil.translate_goal_to_sense(f"grade {g}")
        r  = await ag_teach.act(f"Record grade {g} for {s_el.name}", ctxs["acad"],
            extra={"studentId":sid,"courseId":cid,"grade":str(g)}, sense=gs)
        if r: sil.record_outcome("grade.record","ESchool.Academic",gs,
            SenseVector.normative("grading",0.95),0.95,r.is_success)

    # ═══ PHASE 7: Context Topology ════════════════════════════════════════
    section("PHASE 7 — Context Topology Matrix")

    topo = gw.context_topology()
    print(f"\n  {'Pair':<35} {'Dist':>6}  Semantic Proximity")
    hr(65,"─")
    for t in topo:
        d=t["distance"]; bar="█"*int((1-d)*20)
        print(f"  {t['ctx_a']:<18} ↔ {t['ctx_b']:<18} {d:>6.3f}  {bar}")

    # ═══ PHASE 8: Final ═══════════════════════════════════════════════════
    section("PHASE 8 — Final State + Causal Stats")

    r = await ag_admin.act("System snapshot", ctxs["admin"])
    if r and r.result:
        print(f"\n  {C.B}Environment:{C.R}")
        for k,v in r.result.items(): print(f"    {k:<22}: {v}")

    print(f"\n  {C.B}Causal Graph:{C.R}")
    s = env.causal.stats
    print(f"    nodes={s['nodes']}  ok={s['ok']}  rej={s['rej']}")
    print(f"    avg_dissonance={s['avg_dissonance']:.4f}  (lower=more coherent)")
    print(f"    types=[{', '.join(s['types'])}]")

    print(f"\n  {C.B}Contrastive Memory:{C.R}")
    for at, adj in sil._harmony_adjustments.items():
        print(f"    {at:<38} δH={adj:+.4f}")

    print(f"\n  {C.B}Circuit Breaker:{C.R}  {llm.cb.status()}")

    hr()
    print(f"\n  {C.B}{C.CY}MEP V4 Semantic Intelligence Layer — complete.{C.R}\n")
    hr()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MEP v3 × EDP v4.1 Semantic Intelligence")
    ap.add_argument("--demo",           action="store_true")
    ap.add_argument("--provider",       default="ollama",  choices=["ollama","openai","anthropic"])
    ap.add_argument("--model",          default="llama3")
    ap.add_argument("--host",           default="http://localhost:11434")
    ap.add_argument("--timeout",        type=float, default=45.0)
    ap.add_argument("--retries",        type=int,   default=3)
    ap.add_argument("--inject-memory",  action="store_true",
                    help="Inject agent memory (large models only, default OFF)")
    ap.add_argument("--inject-sit-tone",action="store_true",
                    help="Add situation tone to prompt (default OFF)")
    args = ap.parse_args()

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print(f"\n{C.Y}Interrupted.{C.R}")
    except Exception as e:
        print(f"\n{C.RD}Fatal: {e}{C.R}"); traceback.print_exc(); sys.exit(1)
