"""
examples.py  —  EDP + MEP + SAVOIR — Usage Examples
════════════════════════════════════════════════════
OneOrigine / ImperialSchool Research  —  I.S. License

A practical guide for users and developers.
Five complete examples from simple to advanced.
"""

from __future__ import annotations
import asyncio, math
from edp import (
    Environment, Context, Action, Reaction, Element, Circumstance,
    SenseVector, PhenomenonPattern, EnvironmentKind, ContextKind,
    ActionCategory, ImpactScope, Temporality, RawData,
    SENSE_NULL
)
from mep import MepGateway, MepSession
from savoir import Savoir, CertaintyLevel

# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 1: Minimum viable environment
# The simplest possible EDP environment — 5 lines of setup.
# ═══════════════════════════════════════════════════════════════════════════

async def example_1_minimal():
    """
    EXAMPLE 1 — Minimal EDP Environment

    Demonstrates:
    • Creating an environment and context
    • Registering one action
    • Admitting an element
    • Dispatching and receiving a reaction
    """
    print("\n══ EXAMPLE 1: Minimal Environment ══════════════════════════════")

    # 1. Build environment
    env = Environment("HelloEnv", EnvironmentKind.REACTIVE)

    # 2. Create a context (the semantic frame)
    ctx = env.create_context("Main", ContextKind.SEMANTIC,
        basis=SenseVector.normative("main context", 0.9))

    # 3. Register an action in the context
    async def greet(actor, payload, ctx, frame):
        name = payload.get("name", "World")
        return Reaction.ok("say.hello", f"Hello, {name}!",
            result={"greeting": f"Hello, {name}!"},
            impact=ImpactScope.on_actor())

    ctx.reg(Action("say.hello", ActionCategory.COMMAND, "Say hello",
                   SenseVector.social("greeting", 0.7), handler=greet))

    # 4. Create and admit an element
    class User(Element):
        def __init__(self, name): super().__init__(name, "User")
        async def on_impacted(self, r, frame): print(f"  {self.name} received: {r.message}")

    user = User("Alice")
    await env.admit(user)
    ctx.include(user.to_dict())

    # 5. Discover available actions
    frame = {"actor_id": user.element_id}
    available = ctx.get_available_actions(user.to_dict(), frame)
    print(f"\n  Available actions: {[a.type for a,_ in available]}")

    # 6. Dispatch
    reaction = await env.dispatch(user, "say.hello", {"name":"Alice"}, ctx)
    print(f"  Reaction: {reaction.line()}")
    print(f"  Result:   {reaction.result}")


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 2: Circumstance gating
# Showing the core EDP innovation: action visibility controlled by logic.
# ═══════════════════════════════════════════════════════════════════════════

async def example_2_circumstances():
    """
    EXAMPLE 2 — Circumstance Gating

    Demonstrates:
    • Circumstance as Boolean predicate
    • Action visibility (not blocking — visibility)
    • Algebraic composition: AND, OR, NOT
    • WHY-NOT query
    """
    print("\n══ EXAMPLE 2: Circumstance Gating ══════════════════════════════")

    env = Environment("Shop", EnvironmentKind.REACTIVE)

    # Circumstances (composable Boolean predicates)
    shop_open  = Circumstance.flag("shop.open",  "Shop is open",  "shopOpen",  True)
    user_adult = Circumstance.flag("user.adult", "User is adult", "isAdult",   True)
    has_funds  = Circumstance.when("has.funds",  "User has sufficient funds",
                    lambda ctx, frame: ctx.data.get("balance", 0) >= ctx.data.get("price", 0))

    # Compound circumstance: shop open AND adult AND has funds
    can_purchase = shop_open & user_adult & has_funds

    ctx = env.create_context("ShopFloor", ContextKind.SEMANTIC,
        basis=SenseVector.financial("commerce", 0.8),
        circumstances=[shop_open])
    ctx.set("shopOpen", True)
    ctx.set("balance",  50)
    ctx.set("price",    30)

    async def buy(actor, payload, ctx, frame):
        item = payload.get("item","item")
        return Reaction.ok("purchase.item", f"Purchased {item}",
            result={"item":item}, impact=ImpactScope.on_actor())

    ctx.reg(Action("purchase.item", ActionCategory.COMMAND, "Purchase an item",
                   SenseVector.financial("purchase", 0.9),
                   guards=[can_purchase], handler=buy))
    ctx.reg(Action("browse.catalog", ActionCategory.QUERY, "Browse catalog",
                   SenseVector.social("browse", 0.4), handler=buy))

    class Customer(Element):
        def __init__(self, name, adult): super().__init__(name, "Customer")
        async def on_impacted(self, r, frame): pass

    alice = Customer("Alice", adult=True)
    bob   = Customer("Bob (minor)", adult=False)
    await env.admit(alice); await env.admit(bob)
    ctx.include(alice.to_dict()); ctx.include(bob.to_dict())

    alice_data = {**alice.to_dict(), "isAdult": True}
    bob_data   = {**bob.to_dict(),   "isAdult": False}

    # Alice can see purchase (all conditions met)
    frame_a = {"actor_id": alice.element_id, "isAdult": True}
    ctx.data["isAdult"] = True  # simplified — normally in element
    avail_alice = ctx.get_available_actions(alice_data, frame_a)
    print(f"\n  Alice's available actions: {[a.type for a,_ in avail_alice]}")

    # WHY-NOT query (close shop)
    ctx.set("shopOpen", False)
    ctx.data["shopOpen"] = False
    avail_closed = ctx.get_available_actions(alice_data, frame_a)
    print(f"  Shop closed — available: {[a.type for a,_ in avail_closed]}")
    print(f"  → purchase.item INVISIBLE (not blocked — just not visible)")
    print(f"  → browse.catalog STILL VISIBLE (not gated by shop.open)")


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 3: Causal chains
# Actions spawning reactions spawning actions.
# ═══════════════════════════════════════════════════════════════════════════

async def example_3_causal_chains():
    """
    EXAMPLE 3 — Causal Chains

    Demonstrates:
    • Reaction spawning child actions (chain)
    • Causal graph WHY query
    • Chain depth limit
    • Causal delta Δ_t
    """
    print("\n══ EXAMPLE 3: Causal Chains ══════════════════════════════════════")

    env = Environment("Hospital", EnvironmentKind.REACTIVE)
    events = []
    env.on_reaction(lambda r: events.append(r.action_type))

    ctx = env.create_context("Emergency", ContextKind.CAUSAL,
        basis=SenseVector.causal("emergency ops", 0.95))

    async def triage(actor, payload, ctx, frame):
        severity = payload.get("severity","high")
        # Chain: triage → notify_doctor → prepare_OR
        chain = ["notify.doctor", "prepare.OR"] if severity=="critical" else ["notify.doctor"]
        return Reaction.ok("patient.triage", f"Triaged: severity={severity}",
            result={"severity":severity},
            impact=ImpactScope.on_env(),
            chain=chain)  # → spawns child actions

    async def notify(actor, payload, ctx, frame):
        return Reaction.ok("notify.doctor","Doctor notified",
            impact=ImpactScope.on_env(), chain=["equipment.ready"])

    async def prepare_or(actor, payload, ctx, frame):
        return Reaction.ok("prepare.OR","OR prepared",impact=ImpactScope.on_env())

    async def equip(actor, payload, ctx, frame):
        return Reaction.ok("equipment.ready","Equipment ready",impact=ImpactScope.on_env())

    for a in [
        Action("patient.triage",  ActionCategory.COMMAND, "Triage patient",
               SenseVector.causal("medical triage",0.95), handler=triage),
        Action("notify.doctor",   ActionCategory.SIGNAL,  "Notify on-call doctor",
               SenseVector.social("notification",0.6),   handler=notify),
        Action("prepare.OR",      ActionCategory.COMMAND, "Prepare operating room",
               SenseVector.technical("preparation",0.7), handler=prepare_or),
        Action("equipment.ready", ActionCategory.LIFECYCLE,"Mark equipment ready",
               SenseVector.technical("equipment",0.5),  handler=equip),
    ]:
        ctx.reg(a)

    class Staff(Element):
        async def on_impacted(self, r, f): pass

    staff = Staff("Nurse Chen", "Staff")
    await env.admit(staff); ctx.include(staff.to_dict())

    reaction = await env.dispatch(staff, "patient.triage",
        {"severity":"critical","patient_id":"P001"}, ctx)

    await asyncio.sleep(0.1)  # let chains complete

    print(f"\n  Initial reaction: {reaction.line()}")
    print(f"  Chain spawned:    {reaction.chain_actions}")
    print(f"  All reactions in order: {events}")
    print(f"\n  Causal trace:")
    print(env.explain_why("patient.triage"))


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 4: Phenomenon detection
# The environment observes its own reaction stream and raises alerts.
# ═══════════════════════════════════════════════════════════════════════════

async def example_4_phenomena():
    """
    EXAMPLE 4 — Emergent Phenomenon Detection

    Demonstrates:
    • PhenomenonPattern (sliding window)
    • Phenomenon as attractor in semantic space
    • Environment self-observation
    """
    print("\n══ EXAMPLE 4: Phenomenon Detection ══════════════════════════════")

    env = Environment("Exchange", EnvironmentKind.LIVING)
    detected = []
    env.on_phenomenon(lambda p: detected.append(p))

    # Pattern: 4+ rejections in 30s = "Fraud Cascade" phenomenon
    env.register_pattern(PhenomenonPattern(
        "FraudCascade", "rejected", threshold=4, window_s=30,
        attractor=SenseVector.emergent("fraud signal")))

    ctx = env.create_context("Trading", ContextKind.SEMANTIC,
        basis=SenseVector.financial("trading ops", 0.9))

    # Balance check circumstance
    has_funds = Circumstance.when("has.funds","Sufficient funds",
        lambda ctx, f: ctx.data.get("balance",0) > ctx.data.get("amount",0))
    ctx.set("balance", 10)

    async def trade(actor, payload, ctx, frame):
        amt = payload.get("amount", 0)
        if ctx.data.get("balance",0) < amt:
            return Reaction.reject("trade.execute","Insufficient funds")
        return Reaction.ok("trade.execute",f"Trade executed: ${amt}",
            result={"amount":amt}, impact=ImpactScope.on_actor())

    ctx.reg(Action("trade.execute", ActionCategory.COMMAND, "Execute trade",
                   SenseVector.financial("trade",0.9), handler=trade))

    class Trader(Element):
        async def on_impacted(self, r, f): pass

    trader = Trader("Bot-X", "Trader")
    await env.admit(trader); ctx.include(trader.to_dict())

    # Trigger 5 failed trades → phenomenon emerges
    print(f"\n  Triggering 5 failed trades...")
    for i in range(5):
        r = await env.dispatch(trader,"trade.execute",{"amount":1000},ctx)
        print(f"  Trade {i+1}: {r.status.value}")

    if detected:
        p = detected[0]
        print(f"\n  ★ Phenomenon: '{p.name}'")
        print(f"    Magnitude: {p.magnitude:.0%}")
        print(f"    This was not coded per-domain. It emerged from the reaction stream.")
    else:
        print(f"\n  (Phenomenon detection queued — check after brief delay)")


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 5: SAVOIR — certainty vs probability
# The core innovation: KNOWN facts vs ESTIMATED facts.
# ═══════════════════════════════════════════════════════════════════════════

async def example_5_savoir():
    """
    EXAMPLE 5 — SAVOIR: Certainty Layer

    Demonstrates the core problem SAVOIR solves:
    LLMs probabilize everything — SAVOIR separates KNOWN from ESTIMATED.
    """
    print("\n══ EXAMPLE 5: SAVOIR — Certainty vs Probability ═════════════════")

    savoir = Savoir(
        element_ids=["robot_arm", "cup_A"],
        property_dims=["pos_x","pos_y","pos_z"])

    # Initial state — robot knows its arm position (sensors)
    savoir.assert_known("robot_arm.pos_x", 0.0, "joint_encoder")
    savoir.assert_known("robot_arm.pos_y", 0.0, "joint_encoder")
    savoir.assert_known("robot_arm.pos_z", 0.5, "joint_encoder")

    # Cup position — initially estimated (last seen)
    savoir.assert_estimated("cup_A.pos_x",  1.2, "last_observation")
    savoir.assert_estimated("cup_A.pos_y",  0.5, "last_observation")
    savoir.assert_estimated("cup_A.pos_z",  0.8, "last_observation")

    print("\n  BEFORE manipulation:")
    print(savoir.to_llm_context())
    print(f"\n  Is cup_A.pos_x CERTAIN?  {savoir.is_certain('cup_A.pos_x')}")
    print(f"  Certainty of cup_A.pos_x: {savoir.certainty_of('cup_A.pos_x'):.2f}")

    # Register the "move_cup" action transition
    savoir.transition_matrix.register("robot.move_cup", {
        "pos_x": 0., "pos_y": 0., "pos_z": 0.  # actual values from payload
    })

    # Robot executes: move cup from (1.2,0.5,0.8) to (2.0,0.5,0.8)
    payload = {"d_pos_x": 0.8, "d_pos_y": 0.0, "d_pos_z": 0.0}
    savoir.record_action_outcome("robot.move_cup","cup_A",payload,CertaintyLevel.KNOWN)

    print("\n  AFTER manipulation (robot KNOWS where cup is):")
    print(savoir.to_llm_context())
    print(f"\n  Is cup_A.pos_x CERTAIN?  {savoir.is_certain('cup_A.pos_x')}")
    print(f"  Certainty of cup_A.pos_x: {savoir.certainty_of('cup_A.pos_x'):.2f}")
    print(f"\n  The robot doesn't re-estimate. It KNOWS it moved the cup to x=2.0.")
    print(f"  An LLM querying SAVOIR will see: '✓ cup_A.pos_x = 2.0 [KNOWN]'")
    print(f"  Not: 'The cup is probably around x=2.0 (confidence: 0.83)'")

    # Demonstrate degradation
    print(f"\n  Simulating time passage (certainty degrades for ESTIMATED facts)...")
    for _ in range(3): savoir.degrade_over_time()
    snap = savoir.snapshot()
    print(f"  Known facts:    {snap['total_known']}")
    print(f"  Probable facts: {snap['total_probable']}")


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE 6: Harmony-based action selection
# The mathematical foundation: a* = argmax H(a,c,s,r)
# ═══════════════════════════════════════════════════════════════════════════

async def example_6_harmony():
    """
    EXAMPLE 6 — Harmony-Based Action Selection

    Shows: a* = argmax_{a∈Avail} H(a,c,s,r)
    The AI receives actions ranked by semantic fit, not just availability.
    """
    print("\n══ EXAMPLE 6: Harmony a* = argmax H(A,C,S) ══════════════════════")

    env = Environment("University", EnvironmentKind.REACTIVE)

    # Academic context: basis on normative axis
    acad_ctx = env.create_context("Academic", ContextKind.SEMANTIC,
        basis=SenseVector.normative("academic operations", 0.95))

    # Financial context: basis on financial axis
    fin_ctx = env.create_context("Financial", ContextKind.SEMANTIC,
        basis=SenseVector.financial("financial operations", 0.90))

    # Same actions available in both contexts
    actions = [
        Action("grade.record",    ActionCategory.COMMAND, "Record student grade",
               SenseVector.normative("grading",0.95)),
        Action("payment.process", ActionCategory.COMMAND, "Process tuition payment",
               SenseVector.financial("payment",0.90)),
        Action("student.enroll",  ActionCategory.COMMAND, "Enroll student",
               SenseVector.temporal("enrollment",0.85)),
        Action("system.snapshot", ActionCategory.QUERY,   "System snapshot",
               SenseVector.technical("snapshot",0.30)),
    ]
    for a in actions:
        acad_ctx.reg(a)
        fin_ctx.reg(a)

    class Professor(Element):
        async def on_impacted(self, r, f): pass

    prof = Professor("Dr. Smith", "Faculty")
    await env.admit(prof)
    acad_ctx.include(prof.to_dict()); fin_ctx.include(prof.to_dict())
    frame = {"actor_id": prof.element_id}

    print("\n  ACADEMIC context (basis: normative) — action harmony ranking:")
    for a, h in acad_ctx.get_available_actions(prof.to_dict(), frame):
        bar = "█"*max(1,int((h.score+1)/2*16))
        print(f"    [{bar:<16}] H={h.score:+.3f}  {a.type}")

    print("\n  FINANCIAL context (basis: financial) — same actions, different ranking:")
    for a, h in fin_ctx.get_available_actions(prof.to_dict(), frame):
        bar = "█"*max(1,int((h.score+1)/2*16))
        print(f"    [{bar:<16}] H={h.score:+.3f}  {a.type}")

    print("\n  Same actions. Same actor. Different context → different natural order.")
    print("  The environment shapes what is appropriate, not the actor's hardcoded logic.")


# ─── Run all examples ─────────────────────────────────────────────────────────

async def main():
    print("═"*65)
    print("  EDP + MEP + SAVOIR — Usage Examples")
    print("  OneOrigine / ImperialSchool Research")
    print("═"*65)

    await example_1_minimal()
    await example_2_circumstances()
    await example_3_causal_chains()
    await example_4_phenomena()
    await example_5_savoir()
    await example_6_harmony()

    print("\n" + "═"*65)
    print("  All examples complete.")
    print("  Key takeaways:")
    print("  1. Actions are discovered from context — not hardcoded in actors")
    print("  2. Circumstances make actions invisible — not just blocked")
    print("  3. Causal chains trace automatically — WHY is always answerable")
    print("  4. Phenomena emerge from reaction streams — no per-domain coding")
    print("  5. SAVOIR separates KNOWN from ESTIMATED — no probabilistic noise")
    print("  6. Harmony H(A,C,S) ranks actions by semantic fit — not insertion order")
    print("═"*65 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
