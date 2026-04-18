#!/usr/bin/env python3
"""
mep_cli.py  —  Interactive CLI for MEP Agent
════════════════════════════════════════════════════════════════════════════
OneOrigine / ImperialSchool Research  —  I.S. License

An interactive terminal session where YOU talk to the agent.
The agent has a role, operates in an EDP environment, and you can:
  • Give it natural-language goals ("enroll Alice in Math")
  • Ask WHY an action was taken
  • Ask WHY-NOT when something is blocked
  • Switch contexts dynamically
  • See the impact matrix at any time
  • Export causal graph to DOT format

Usage:
  python mep_cli.py                         # demo mode (no LLM)
  python mep_cli.py --model gemma3:12b      # with Ollama
  python mep_cli.py --provider openai       # with OpenAI
  python mep_cli.py --provider anthropic    # with Anthropic

  python mep_cli.py --role "university_admin"
  python mep_cli.py --role "drone_pilot"
  python mep_cli.py --role "hospital_admin"

CLI Commands (type in the prompt):
  /help             show all commands
  /ctx              show current context + harmony map
  /switch <name>    switch to another context
  /why <action>     explain why an action succeeded
  /whynot <action>  explain why an action is blocked
  /impact           show session impact matrix
  /causal           export causal graph (DOT format)
  /savoir           show SAVOIR knowledge base
  /env              show environment snapshot
  /history          show last N actions
  /export           export session data to JSON
  /quit             end session
"""

from __future__ import annotations

import argparse, asyncio, json, os, readline, sys, textwrap, time
from typing import Dict, List, Optional, Tuple

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from edp import (
    Environment, Context, Action, Reaction, Element, Circumstance,
    SenseVector, PhenomenonPattern, EnvironmentKind, ContextKind,
    ActionCategory, ImpactScope, Temporality, SENSE_NULL
)
from mep import (
    MepGateway, MepAgent, MepSession, make_provider, LlmError
)
from savoir import Savoir, CertaintyLevel
from impact_matrix import SessionTracker, ImpactMatrix
from contextualizer import Contextualizer, DataSignal

# ─── Color / terminal ─────────────────────────────────────────────────────────

class T:
    RST = "\033[0m";  B = "\033[1m";  D = "\033[2m";  I = "\033[3m"
    CY = "\033[96m";  G = "\033[92m"; Y = "\033[93m"
    RD = "\033[91m";  MG= "\033[95m"; BL= "\033[94m"
    BG_D = "\033[48;5;234m"  # dark bg for prompt

def hr(n=70, ch="═", c=T.D): print(f"{c}{ch*n}{T.RST}")
def box(t): hr(); print(f"{T.B}{T.CY}  {t}{T.RST}"); hr()
def ok(m):   print(f"  {T.G}✓{T.RST} {m}")
def err(m):  print(f"  {T.RD}✗{T.RST} {m}")
def warn(m): print(f"  {T.Y}!{T.RST} {m}")
def info(m): print(f"  {T.CY}→{T.RST} {m}")
def dim(m):  print(f"  {T.D}{m}{T.RST}")
def rxn(m):  print(f"  {T.BL}↩{T.RST} {m}")

# ─── ROLE DEFINITIONS ─────────────────────────────────────────────────────────

ROLES = {
    "university_admin": {
        "title":   "University Administrator",
        "persona": ("You are an AI assistant for the university administration office. "
                    "You help manage course enrollments, student records, and faculty assignments. "
                    "You operate according to university policies and academic regulations."),
        "env_name": "University EDP",
        "primary_context": "admin",
    },
    "drone_pilot": {
        "title":   "Drone Operations Manager",
        "persona": ("You are an AI flight operations controller for an autonomous drone fleet. "
                    "You coordinate missions, manage flight contexts, handle emergencies, and "
                    "ensure all operations comply with airspace regulations and safety protocols."),
        "env_name": "Drone Operations Center",
        "primary_context": "flight",
    },
    "hospital_admin": {
        "title":   "Hospital Operations AI",
        "persona": ("You are an AI assistant for hospital operations. "
                    "You help coordinate patient care, staff assignments, resource allocation, "
                    "and compliance with medical protocols. Patient safety is your top priority."),
        "env_name": "Hospital Operations",
        "primary_context": "admin",
    },
}

# ─── ENVIRONMENT BUILDER ──────────────────────────────────────────────────────

def build_environment_for_role(role_id: str) -> Tuple[Environment, Dict[str,Context], Dict, Savoir]:
    """Build a role-appropriate EDP environment."""
    role = ROLES.get(role_id, ROLES["university_admin"])
    env  = Environment(role["env_name"], EnvironmentKind.REACTIVE)
    sav  = Savoir()

    if role_id == "university_admin":
        return _build_university(env, sav)
    elif role_id == "drone_pilot":
        return _build_drone_ops(env, sav)
    else:
        return _build_university(env, sav)  # default


def _build_university(env, sav):
    # Circumstances
    C_SYS  = Circumstance.always("system.active")
    C_ENRL = Circumstance.flag("enrollment.open","Enrollment period active","enrollmentOpen")

    gdata = {"enrollmentOpen": False, "students": {}, "courses": {}, "grades": []}

    # Contexts
    admin_ctx = env.create_context("Administrative", ContextKind.SEMANTIC,
        basis=SenseVector.normative("university administration", 0.9),
        circumstances=[C_SYS])
    admin_ctx.set("enrollmentOpen", False)

    acad_ctx = env.create_context("Academic", ContextKind.SEMANTIC,
        basis=SenseVector.normative("academic operations", 0.95),
        circumstances=[C_SYS])

    enrl_ctx = env.create_context("Enrollment", ContextKind.TEMPORAL,
        basis=SenseVector.temporal("enrollment management", 0.9),
        circumstances=[C_SYS, C_ENRL])
    enrl_ctx.set("enrollmentOpen", False)

    # Actions
    async def h_add_student(a, p, ctx, f):
        name=p.get("name","Student"); sid=f"S{len(gdata['students'])+1:03d}"
        gdata["students"][sid]={"id":sid,"name":name,"enrolled":[],"gpa":0.}
        sav.assert_known(f"student.{sid}.name",name,"registry")
        return Reaction.ok("student.add",f"Student '{name}' added (ID:{sid})",
            result={"id":sid,"name":name},impact=ImpactScope.on_env(0.5))

    async def h_add_course(a, p, ctx, f):
        name=p.get("name","Course"); cap=int(p.get("capacity",30))
        cid=f"C{len(gdata['courses'])+1:03d}"
        gdata["courses"][cid]={"id":cid,"name":name,"capacity":cap,"enrolled":[],"grades":[]}
        sav.assert_known(f"course.{cid}.name",name,"registry")
        sav.assert_known(f"course.{cid}.capacity",cap,"registry")
        return Reaction.ok("course.add",f"Course '{name}' added (ID:{cid}, cap={cap})",
            result={"id":cid,"name":name},impact=ImpactScope.on_env(0.5))

    async def h_open_enrl(a, p, ctx, f):
        gdata["enrollmentOpen"]=True
        for cx in [admin_ctx,enrl_ctx]: cx.set("enrollmentOpen",True)
        sav.assert_known("enrollment.status","open","system")
        return Reaction.ok("enrollment.open","Enrollment period OPEN",impact=ImpactScope.on_env())

    async def h_close_enrl(a, p, ctx, f):
        gdata["enrollmentOpen"]=False
        for cx in [admin_ctx,enrl_ctx]: cx.set("enrollmentOpen",False)
        sav.assert_known("enrollment.status","closed","system")
        return Reaction.ok("enrollment.close","Enrollment period CLOSED",impact=ImpactScope.on_env())

    async def h_enroll(a, p, ctx, f):
        sid=p.get("studentId",""); cid=p.get("courseId","")
        s=gdata["students"].get(sid); c=gdata["courses"].get(cid)
        if not s: return Reaction.reject("student.enroll","Student not found")
        if not c: return Reaction.reject("student.enroll","Course not found")
        if len(c["enrolled"])>=c["capacity"]: return Reaction.reject("student.enroll","Course full")
        if sid in c["enrolled"]: return Reaction.reject("student.enroll","Already enrolled")
        c["enrolled"].append(sid); s["enrolled"].append(cid)
        sav.assert_known(f"student.{sid}.enrolled_in",cid,"system")
        return Reaction.ok("student.enroll",f"{s['name']} enrolled in {c['name']}",
            result={"student":s["name"],"course":c["name"]},impact=ImpactScope.on_env(0.7))

    async def h_grade(a, p, ctx, f):
        sid=p.get("studentId",""); cid=p.get("courseId","")
        try: g=float(p.get("grade",0))
        except: return Reaction.reject("grade.record","Invalid grade value")
        s=gdata["students"].get(sid); c=gdata["courses"].get(cid)
        if not s or not c: return Reaction.reject("grade.record","Student/course not found")
        if not (0<=g<=100): return Reaction.reject("grade.record","Grade must be 0-100")
        lt="A"if g>=90 else"B"if g>=80 else"C"if g>=70 else"D"if g>=60 else"F"
        rec={"s":sid,"c":cid,"g":g,"l":lt}
        gdata["grades"].append(rec); c["grades"].append(rec)
        all_g=[r["g"] for r in gdata["grades"] if r["s"]==sid]
        s["gpa"]=round(sum(all_g)/len(all_g),2)
        sav.assert_known(f"student.{sid}.gpa",s["gpa"],"grading_system")
        chain=[] if g>=60 else ["notification.send"]
        return Reaction.ok("grade.record",f"{s['name']}: {g:.0f} ({lt}) GPA→{s['gpa']}",
            result={"grade":g,"letter":lt,"gpa":s["gpa"]},
            impact=ImpactScope.on_env(0.6),chain=chain)

    async def h_snapshot(a, p, ctx, f):
        return Reaction.ok("system.snapshot","Snapshot",
            result={"students":len(gdata["students"]),"courses":len(gdata["courses"]),
                    "grades":len(gdata["grades"]),
                    "enrollmentOpen":gdata["enrollmentOpen"]})

    async def h_list(a, p, ctx, f):
        what=p.get("what","all")
        result={}
        if what in ("all","students"): result["students"]=list(gdata["students"].values())
        if what in ("all","courses"):  result["courses"] =list(gdata["courses"].values())
        return Reaction.ok("system.list","Element listing",result=result)

    for ctx_obj, actions in [
        (admin_ctx,[
            Action("student.add",    ActionCategory.LIFECYCLE,"Add a new student",
                   SenseVector.social("student registration",0.7),handler=h_add_student),
            Action("course.add",     ActionCategory.LIFECYCLE,"Add a new course",
                   SenseVector.technical("course creation",0.7),handler=h_add_course),
            Action("enrollment.open",ActionCategory.COMMAND,  "Open enrollment period",
                   SenseVector.temporal("enrollment activation",0.85),handler=h_open_enrl),
            Action("enrollment.close",ActionCategory.COMMAND, "Close enrollment period",
                   SenseVector.temporal("enrollment close",0.7),handler=h_close_enrl),
            Action("system.snapshot",ActionCategory.QUERY,   "System state snapshot",
                   SenseVector.technical("snapshot",0.3),handler=h_snapshot),
            Action("system.list",    ActionCategory.QUERY,   "List students or courses",
                   SenseVector.technical("listing",0.3),handler=h_list),
        ]),
        (acad_ctx,[
            Action("grade.record",   ActionCategory.COMMAND,"Record student grade",
                   SenseVector.normative("academic grading",0.95),handler=h_grade),
            Action("system.snapshot",ActionCategory.QUERY,  "System snapshot",
                   SenseVector.technical("snapshot",0.3),handler=h_snapshot),
            Action("system.list",    ActionCategory.QUERY,  "List elements",
                   SenseVector.technical("listing",0.3),handler=h_list),
        ]),
        (enrl_ctx,[
            Action("student.enroll", ActionCategory.COMMAND,"Enroll student in course",
                   SenseVector.temporal("enrollment",0.95),
                   guards=[C_ENRL],handler=h_enroll),
            Action("system.snapshot",ActionCategory.QUERY,  "System snapshot",
                   SenseVector.technical("snapshot",0.3),handler=h_snapshot),
        ]),
    ]:
        for act in actions: ctx_obj.reg(act)

    env.register_pattern(PhenomenonPattern(
        "MassFailure","rejected",3,300,
        attractor=SenseVector.emergent("academic failure")))

    return env, {"admin":admin_ctx,"acad":acad_ctx,"enrl":enrl_ctx}, gdata, sav


def _build_drone_ops(env, sav):
    C_BAT = Circumstance.when("battery.ok","Battery > 15%",
        lambda ctx,f: float(ctx.data.get("battery",100))>15)
    C_GPS = Circumstance.always("gps.lock")
    C_AIR = Circumstance.when("airborne","Drone is airborne",
        lambda ctx,f: ctx.data.get("airborne",False))

    gdata = {"pos":[0.,0.,5.],"battery":100.,"airborne":False}

    flight_ctx = env.create_context("Flight", ContextKind.SPATIAL,
        basis=SenseVector.spatial("flight operations",0.95),
        circumstances=[C_BAT,C_GPS])
    flight_ctx.set("battery",100.); flight_ctx.set("airborne",False)

    emrg_ctx = env.create_context("Emergency", ContextKind.CAUSAL,
        basis=SenseVector.causal("emergency response",0.99))

    async def h_move(a,p,ctx,f):
        dx,dy,dz=float(p.get("dx",0)),float(p.get("dy",0)),float(p.get("dz",0))
        gdata["pos"][0]+=dx; gdata["pos"][1]+=dy; gdata["pos"][2]+=dz
        gdata["battery"]-=0.5
        flight_ctx.set("battery",gdata["battery"])
        sav.assert_known("drone.pos",tuple(gdata["pos"]),"gps+imu")
        return Reaction.ok("drone.move",
            f"Moved Δ({dx:.1f},{dy:.1f},{dz:.1f}) → pos={[round(x,1) for x in gdata['pos']]}",
            result={"position":gdata["pos"],"battery":gdata["battery"]},
            impact=ImpactScope.on_actor(0.8))

    async def h_scan(a,p,ctx,f):
        return Reaction.ok("drone.scan",f"Scanning area at {[round(x,1) for x in gdata['pos']]}",
            result={"position":gdata["pos"]},impact=ImpactScope.on_env(0.3))

    async def h_rth(a,p,ctx,f):
        gdata["pos"]=[0.,0.,0.]; gdata["airborne"]=False
        sav.assert_known("drone.pos",(0.,0.,0.),"fc")
        return Reaction.ok("drone.rth","EMERGENCY RTH complete",
            sense=SenseVector.causal("emergency",1.0),impact=ImpactScope.on_actor(1.0))

    async def h_status(a,p,ctx,f):
        return Reaction.ok("drone.status","Status",
            result={"position":gdata["pos"],"battery":gdata["battery"],
                    "airborne":gdata["airborne"],"gps_lock":True})

    flight_ctx.reg(Action("drone.move",ActionCategory.COMMAND,"Move drone",
        SenseVector.spatial("translational movement",0.95),handler=h_move))
    flight_ctx.reg(Action("drone.scan",ActionCategory.QUERY,"Scan area",
        SenseVector.technical("sensor scan",0.6),handler=h_scan))
    flight_ctx.reg(Action("drone.status",ActionCategory.QUERY,"Get drone status",
        SenseVector.technical("telemetry",0.3),handler=h_status))
    for ctx_obj in [flight_ctx,emrg_ctx]:
        ctx_obj.reg(Action("drone.rth",ActionCategory.SIGNAL,"Return to home EMERGENCY",
            SenseVector.causal("emergency RTH",1.0),handler=h_rth))

    env.register_pattern(PhenomenonPattern("BatteryCritical","rejected",2,60,
        attractor=SenseVector.emergent("battery emergency")))

    return env, {"flight":flight_ctx,"emergency":emrg_ctx}, gdata, sav


# ─── CLI SESSION ──────────────────────────────────────────────────────────────

class MepCLI:
    """Interactive CLI session for the MEP agent."""

    def __init__(self, role_id: str, provider_name: str, model: str,
                 host: str, timeout: float, retries: int,
                 inject_memory: bool, demo: bool):
        self.role_id      = role_id
        self.role         = ROLES.get(role_id, ROLES["university_admin"])
        self.demo         = demo
        self._build_env(role_id)
        self._build_agent(provider_name, model, host, timeout, retries, inject_memory)
        self.tracker      = SessionTracker(f"cli-{role_id}")
        self.cx           = Contextualizer()
        self.history: List[Dict] = []
        self._running     = True

    def _build_env(self, role_id):
        self.env, self.contexts, self.gdata, self.sav = build_environment_for_role(role_id)
        self.ctx_names = list(self.contexts.keys())
        self.current_ctx_name = self.role.get("primary_context", self.ctx_names[0])
        self.current_ctx = self.contexts[self.current_ctx_name]

    def _build_agent(self, provider_name, model, host, timeout, retries, inject_memory):
        llm = make_provider(provider_name, model, host, timeout, retries)
        if not llm.available and not self.demo:
            warn(f"LLM '{provider_name}/{model}' not reachable — demo mode")
            self.demo = True
        self.llm = llm
        self.gw  = MepGateway(self.env)

        # Create a proxy element for the CLI agent
        class AgentElement(Element):
            async def on_impacted(self, r, f): pass
        self.agent_el = AgentElement("CLI-Agent", "AI")
        self.env_admit_sync(self.agent_el)

        for ctx in self.contexts.values(): ctx.include(self.agent_el.to_dict())

        self.session = self.gw.connect(f"cli-{self.role_id}")
        self.mep_agent = MepAgent(
            self.gw, self.llm, self.agent_el, self.session,
            demo=self.demo, aid="CLI",
            inject_memory=inject_memory, inject_sit_tone=True)

        # Wire tracker
        self.env.on_reaction(lambda r: (
            setattr(self.tracker, '_current_action', r.action_type) or
            self.tracker.record_reaction(r)))

    def env_admit_sync(self, el):
        # Synchronously admit: works both inside and outside event loop
        import concurrent.futures
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.env.admit(el))
        finally:
            loop.close()

    # ── Display helpers ───────────────────────────────────────────────────

    def show_header(self):
        hr(70,"╔")
        print(f"  {T.B}{T.CY}MEP Interactive CLI  —  {self.role['title']}{T.RST}")
        print(f"  {T.D}Provider: {'DEMO' if self.demo else self.llm.provider_name()}"
              f"  Role: {self.role_id}  "
              f"Session: {self.session.session_id[:8]}{T.RST}")
        hr(70,"╚")
        print(f"\n  {T.I}Persona:{T.RST} {self.role['persona'][:100]}...")
        print(f"\n  Type your goals naturally. Commands: /help /ctx /why /impact /quit\n")

    def show_context(self):
        ctx = self.current_ctx
        frame = {"actor_id": self.agent_el.element_id}
        available = ctx.get_available_actions(self.agent_el.to_dict(), frame)

        hr(60,"─")
        print(f"  {T.B}Context: {ctx.name} [{ctx.kind.value}]{T.RST}  depth={ctx.depth}")

        # Circumstances
        circ_evals = ctx.evaluate_circumstances(frame)
        holding = [c for c in circ_evals if c["holds"]]
        missing  = [c for c in circ_evals if not c["holds"] and c["role"]=="enabler"]
        if holding: dim("Active: " + "  ".join(f'✓{c["id"]}' for c in holding))
        if missing: dim("Missing:" + "  ".join(f'✗{c["id"]}' for c in missing))

        # Actions with harmony
        if available:
            print(f"\n  {T.B}Available actions (H=harmony):{T.RST}")
            for a, h in available:
                bar = "█"*max(1,int((h.score+1)/2*12))
                print(f"    [{bar:<12}] H={h.score:+.3f}  "
                      f"{T.CY}{a.type}{T.RST}  {T.D}{a.description}{T.RST}")
        else:
            warn("No actions available in this context")

        # Other contexts
        other = [n for n in self.ctx_names if n != self.current_ctx_name]
        if other: dim(f"Other contexts: {', '.join(other)}  (use /switch <name>)")
        hr(60,"─")

    def show_help(self):
        hr(60)
        print(f"  {T.B}MEP CLI Commands{T.RST}\n")
        cmds = [
            ("/help",           "Show this help"),
            ("/ctx",            "Show current context + available actions"),
            ("/switch <name>",  "Switch to another context (admin/acad/enrl/flight...)"),
            ("/why <action>",   "Explain why an action succeeded (causal trace)"),
            ("/whynot <action>","Explain why an action is blocked"),
            ("/impact",         "Show session impact matrix"),
            ("/causal",         "Export causal graph in DOT format"),
            ("/savoir",         "Show SAVOIR knowledge base (certainty layer)"),
            ("/env",            "Show environment snapshot"),
            ("/history",        "Show last 10 agent decisions"),
            ("/export",         "Export session to JSON"),
            ("/quit",           "End session"),
            ("",                ""),
            ("<any text>",      "Give the agent a goal in natural language"),
        ]
        for cmd, desc in cmds:
            if not cmd: print()
            else: print(f"  {T.CY}{cmd:<22}{T.RST} {desc}")
        hr(60)

    def show_impact(self):
        if self.tracker.record_count == 0:
            warn("No actions recorded yet"); return
        m = self.tracker.impact_matrix()
        hr(60,"─")
        print(f"  {T.B}Session Impact Matrix{T.RST}  ({self.tracker.record_count} actions)")
        print(m.to_table())
        vec = self.tracker.session_vector
        print(f"\n  φ_session ∈ ℝ^{len(vec)}: [{', '.join(f'{v:+.2f}' for v in vec[:6])}{'...' if len(vec)>6 else ''}]")
        print(f"\n  {T.B}Summary:{T.RST} {self.tracker.summary()}")
        hr(60,"─")

    def show_causal(self):
        if self.tracker.record_count == 0:
            warn("No causal data yet"); return
        dot = self.tracker.causal_graph().to_dot()
        hr(60,"─")
        print(f"  {T.B}Causal Graph (DOT format){T.RST}")
        print(f"  {T.D}(paste into https://dreampuf.github.io/GraphvizOnline/){T.RST}\n")
        print(textwrap.indent(dot, "  "))
        hr(60,"─")

    def show_savoir(self):
        snap = self.sav.snapshot()
        hr(60,"─")
        print(f"  {T.B}SAVOIR Knowledge Base{T.RST}")
        if snap["known_facts"]:
            print(f"\n  {T.G}KNOWN (certainty=1.0):{T.RST}")
            for k, v in snap["known_facts"].items():
                print(f"    ✓ {k} = {v['value']}  [{v['source']}]")
        if snap["probable_facts"]:
            print(f"\n  {T.Y}PROBABLE (certainty<1.0):{T.RST}")
            for k, v in snap["probable_facts"].items():
                print(f"    ~ {k} ≈ {v['value']}  [c={v['certainty']:.2f}]")
        dim(f"\n  State matrix: {snap['state_matrix']['shape']}"
            f"  avg_certainty={snap['state_matrix']['avg_certainty']:.2f}")
        hr(60,"─")

    def show_env(self):
        snap = self.env.snapshot()
        hr(60,"─")
        print(f"  {T.B}Environment: {snap['name']}{T.RST}")
        for k, v in snap.items():
            if k not in ("name","at","env_id"): print(f"    {k:<20}: {v}")
        # Show active data from contexts
        print(f"\n  {T.B}Context Data:{T.RST}")
        for name, ctx in self.contexts.items():
            filtered = {k:v for k,v in ctx.data.items() if not k.startswith("_")}
            if filtered: print(f"    [{name}] {filtered}")
        hr(60,"─")

    def show_history(self):
        if not self.history:
            warn("No history yet"); return
        hr(60,"─")
        print(f"  {T.B}Last {min(10,len(self.history))} Decisions{T.RST}")
        for h in self.history[-10:]:
            s=h.get("status","?"); icon="✓" if s=="success" else "✗"
            col=T.G if s=="success" else T.RD
            print(f"  {col}{icon}{T.RST} [{h.get('ctx','?')}] {h.get('action','?')}"
                  f"  {T.D}← {h.get('goal','?')[:50]}{T.RST}")
        hr(60,"─")

    def export_session(self):
        data = {
            "session_id": self.session.session_id,
            "role": self.role_id,
            "timestamp": time.time(),
            "history": self.history,
            "impact_matrix": self.tracker.impact_matrix().to_matrix_export(),
            "causal_graph_json": self.tracker.causal_graph().to_json(),
            "savoir_snapshot": self.sav.snapshot(),
            "env_snapshot": self.env.snapshot(),
            "session_vector": self.tracker.session_vector,
        }
        fname = f"mep_session_{self.session.session_id[:8]}.json"
        with open(fname,"w") as f: json.dump(data, f, indent=2, default=str)
        ok(f"Session exported to {fname}")

    # ── Main loop ─────────────────────────────────────────────────────────

    async def handle_command(self, cmd: str) -> bool:
        """Handle /command. Returns True to continue, False to quit."""
        parts = cmd.strip().split(maxsplit=1)
        c = parts[0].lower()
        arg = parts[1] if len(parts)>1 else ""

        if c == "/help":     self.show_help()
        elif c == "/ctx":    self.show_context()
        elif c == "/impact": self.show_impact()
        elif c == "/causal": self.show_causal()
        elif c == "/savoir": self.show_savoir()
        elif c == "/env":    self.show_env()
        elif c == "/history":self.show_history()
        elif c == "/export": self.export_session()
        elif c == "/quit":   return False

        elif c == "/switch":
            if arg in self.contexts:
                self.current_ctx_name = arg
                self.current_ctx      = self.contexts[arg]
                ok(f"Switched to context: {self.current_ctx.name}")
                self.show_context()
            else:
                err(f"Unknown context '{arg}'. Available: {', '.join(self.ctx_names)}")

        elif c == "/why":
            if arg:
                print(f"\n  {T.B}WHY '{arg}' succeeded:{T.RST}")
                print(textwrap.indent(self.env.explain_why(arg), "  "))
            else: warn("Usage: /why <action_type>")

        elif c == "/whynot":
            if arg:
                # Find the action
                frame = {"actor_id": self.agent_el.element_id}
                for entry in self.current_ctx._actions:
                    if entry["a"].type == arg:
                        reasons = self.env.explain_why_not(
                            self.agent_el, entry["a"], self.current_ctx)
                        print(f"\n  {T.B}WHY-NOT '{arg}':{T.RST}")
                        if reasons:
                            for r in reasons: err(r)
                        else:
                            ok(f"'{arg}' IS admissible in current context")
                        break
                else:
                    err(f"Action '{arg}' not found in '{self.current_ctx.name}'")
            else: warn("Usage: /whynot <action_type>")

        else:
            warn(f"Unknown command '{c}'. Type /help")

        return True

    async def handle_goal(self, goal: str):
        """Process a natural-language goal through the MEP agent."""
        print()
        from mep import MepParser
        from edp import compute_harmony

        # Show what the agent is about to receive
        env_snap = self.env.snapshot()
        sit_sense = SenseVector.technical("operational",0.5)
        if env_snap.get("situation","operational") == "degraded":
            sit_sense = SenseVector.causal("degraded",0.8)

        self.tracker.set_action_context("(pending)", self.current_ctx.name)
        self.mep_agent.inject_sit_tone = True

        reaction = await self.mep_agent.act(goal, self.current_ctx,
            sense=sit_sense)

        if reaction:
            self.history.append({
                "goal":   goal,
                "action": reaction.action_type,
                "status": reaction.status.value,
                "ctx":    self.current_ctx_name,
                "at":     time.time()
            })
            self.tracker.set_action_context(reaction.action_type, self.current_ctx.name)
        print()

    async def run(self):
        self.show_header()
        self.show_context()

        # Readline history
        try:
            readline.set_history_length(200)
            readline.read_history_file(".mep_cli_history")
        except: pass

        while self._running:
            try:
                # Rich prompt
                ctx_label = f"{T.CY}{self.current_ctx_name}{T.RST}"
                prompt = f"\n  {T.B}{T.BG_D} {self.role['title']} {T.RST} [{ctx_label}] ❯ "
                try:
                    user_input = input(prompt).strip()
                except (EOFError, KeyboardInterrupt):
                    print(); break

                if not user_input: continue

                readline.add_history(user_input)

                if user_input.startswith("/"):
                    if not await self.handle_command(user_input):
                        break
                else:
                    await self.handle_goal(user_input)

            except Exception as e:
                err(f"Error: {e}")

        # Save history
        try: readline.write_history_file(".mep_cli_history")
        except: pass

        # Exit summary
        print()
        hr()
        print(f"  {T.B}Session Complete{T.RST}")
        if self.tracker.record_count > 0:
            print(f"  {self.tracker.summary()}")
        hr()


# ─── ENTRY ────────────────────────────────────────────────────────────────────

async def main():
    ap = argparse.ArgumentParser(
        description="MEP Interactive CLI — AI Agent in EDP Environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
          Roles:
            university_admin  — Manage students, courses, enrollment (default)
            drone_pilot       — Control drone fleet operations
            hospital_admin    — Hospital operations coordination

          Examples:
            python mep_cli.py --demo
            python mep_cli.py --model gemma3:12b --role university_admin
            python mep_cli.py --provider openai --model gpt-4o-mini
        """))
    ap.add_argument("--demo",     action="store_true", help="Demo mode (no LLM)")
    ap.add_argument("--role",     default="university_admin",
                    choices=list(ROLES.keys()), help="Agent role")
    ap.add_argument("--provider", default="ollama",
                    choices=["ollama","openai","anthropic"])
    ap.add_argument("--model",    default="llama3",   help="Model name")
    ap.add_argument("--host",     default="http://localhost:11434")
    ap.add_argument("--timeout",  type=float, default=45.)
    ap.add_argument("--retries",  type=int,   default=3)
    ap.add_argument("--inject-memory", action="store_true",
                    help="Inject agent memory into prompts (large models)")
    args = ap.parse_args()

    cli = MepCLI(args.role, args.provider, args.model,
                  args.host, args.timeout, args.retries,
                  args.inject_memory, args.demo)
    await cli.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{T.Y}Session ended.{T.RST}")
