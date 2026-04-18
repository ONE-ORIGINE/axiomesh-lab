"""
drone_edp.py  —  EDP + MEP + SAVOIR for Autonomous Drones
════════════════════════════════════════════════════════════════════════════
OneOrigine / ImperialSchool Research  —  I.S. License

Language choice rationale:
  Python chosen over C/C++ because:
    • MAVLink (drone protocol) → Python SDK (pymavlink, dronekit)
    • ROS2 → Python rclpy nodes
    • Most drone ML pipelines (YOLO, depth estimation) → Python
    • C/C++ version would be production-embedded; Python is mission-planning layer
    • The EDP pattern is the coordination layer, not the real-time control loop
    (Real-time motor PID remains in C on flight controller — this is the mind, not the muscles)

Architecture:
  ┌────────────────────────────────────────────────────────┐
  │  Mission AI (LLM via MEP)                              │
  │  Receives: ContextEnvelope with drone state + harmony  │
  ├────────────────────────────────────────────────────────┤
  │  SAVOIR — certainty layer                              │
  │  KNOWS: GPS position (0.97), battery (1.0), velocity   │
  │  ESTIMATES: target position (0.70), wind speed (0.60)  │
  ├────────────────────────────────────────────────────────┤
  │  EDP Environment — DroneSwarm                          │
  │  Contexts: Flight / Navigation / Emergency / Scan      │
  │  Circumstances: battery>15%, gps_lock, no_collision    │
  │  Actions: takeoff, move, scan, land, return_home       │
  │  Reactions: position_updated, obstacle_detected, ...   │
  │  Phenomena: NavigationFailureCascade, BatteryCluster   │
  └────────────────────────────────────────────────────────┘
  │  MAVLink / ROS2 (not in this file — this is the mind)  │
  └────────────────────────────────────────────────────────┘

Semantic adaptation for drones:
  • Causal axis   → propulsion causality (thrust → motion)
  • Temporal axis → waypoint timing, ETAs, battery time-to-empty
  • Spatial axis  → PRIMARY axis for drones (position, vector, zone)
  • Normative axis → mission rules, airspace regulations, no-fly zones
  • Technical axis → telemetry, sensors, actuator status
  • Emergent axis → swarm emergent behaviors, formation patterns
"""

from __future__ import annotations

import asyncio, math, time, uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from edp import (
    Environment, Context, Action, Reaction, Element, Circumstance,
    SenseVector, HarmonyProfile, PhenomenonPattern,
    EnvironmentKind, ContextKind, ActionCategory,
    ReactionStatus, ImpactScope, Temporality,
    RawData, CausalGraph, SENSE_NULL
)
from mep import MepGateway, MepSession, ContextEnvelope
from savoir import Savoir, CertaintyLevel, EnvironmentalStateMatrix

# ─── Drone-specific semantic axes ─────────────────────────────────────────────

class DroneS:
    """Drone-adapted SenseVector factories.
    Spatial is the primary axis for drones (unlike software systems where normative dominates).
    """
    @staticmethod
    def spatial(m: str, g: float=1.0): return SenseVector.spatial(m, g)
    @staticmethod
    def navigation(m: str, g: float=1.0): return SenseVector.temporal(m, g)
    @staticmethod
    def emergency(m: str, g: float=1.0): return SenseVector.causal(m, g)
    @staticmethod
    def mission(m: str, g: float=1.0): return SenseVector.normative(m, g)
    @staticmethod
    def telemetry(m: str, g: float=1.0): return SenseVector.technical(m, g)
    @staticmethod
    def swarm(m: str, g: float=1.0): return SenseVector.social(m, g)
    @staticmethod
    def formation(m: str, g: float=1.0): return SenseVector.emergent(m, g)


# ─── Drone state ──────────────────────────────────────────────────────────────

@dataclass
class DroneState:
    """Physical state of one drone — updated from sensors via SAVOIR."""
    x: float = 0.0; y: float = 0.0; z: float = 0.0   # position (m)
    vx: float = 0.0; vy: float = 0.0; vz: float = 0.0 # velocity (m/s)
    yaw: float = 0.0                                    # heading (deg)
    battery_pct: float = 100.0                          # battery %
    battery_time_s: float = 1200.0                      # estimated seconds remaining
    gps_sats: int = 8                                   # GPS satellite count
    gps_hdop: float = 1.2                               # GPS dilution (lower=better)
    armed: bool = False
    airborne: bool = False
    mode: str = "STABILIZE"

    @property
    def position(self) -> Tuple[float, float, float]: return (self.x, self.y, self.z)

    @property
    def gps_lock(self) -> bool: return self.gps_sats >= 6 and self.gps_hdop < 2.0

    @property
    def battery_critical(self) -> bool: return self.battery_pct < 15.0

    @property
    def battery_low(self) -> bool: return self.battery_pct < 25.0

    def update_from_savoir(self, savoir: Savoir, drone_id: str):
        """Sync state from SAVOIR knowledge base."""
        for attr, key in [("x","pos_x"),("y","pos_y"),("z","pos_z"),
                          ("battery_pct","battery"),("vx","vel_x"),
                          ("vy","vel_y"),("vz","vel_z")]:
            val = savoir.value_of(f"{drone_id}.{key}")
            if val is not None: setattr(self, attr, float(val))


# ─── Drone Element ────────────────────────────────────────────────────────────

class Drone(Element):
    """
    A drone as an EDP Element.

    Key insight: the drone's position IS known after each movement
    (actuator feedback + IMU integration) — SAVOIR holds this at certainty=KNOWN.
    The LLM does not re-estimate position. It queries SAVOIR.
    """

    def __init__(self, name: str, drone_id: str):
        super().__init__(name, "Drone",
                         sense=DroneS.spatial(f"drone_{name}", 0.8))
        self.drone_id = drone_id
        self.state    = DroneState()
        self.savoir   = Savoir(
            element_ids=[drone_id],
            property_dims=["pos_x","pos_y","pos_z","battery","vel_x","vel_y","vel_z"])

        # Register reaction transitions (M_R)
        self.savoir.transition_matrix.register("drone.move", {
            "pos_x": 0., "pos_y": 0., "pos_z": 0.,  # filled from payload
            "battery": -0.5,  # expected battery cost per move
        })
        self.savoir.transition_matrix.register("drone.takeoff", {
            "pos_z": 5.0,   # default takeoff altitude
            "battery": -2.0,
        })
        self.savoir.transition_matrix.register("drone.land", {
            "pos_z": 0.0,
            "battery": -1.0,
        })

        # Initial known state
        self._seed_savoir()

        # Property bridge for EDP circumstance evaluation
        self.set_stable("drone_id",  drone_id)
        self.set_stable("drone_name", name)
        self.set_dynamic("battery",  100.0)
        self.set_dynamic("gps_lock", True)
        self.set_dynamic("airborne", False)

    def _seed_savoir(self):
        """Seed initial SAVOIR facts at KNOWN certainty."""
        sid = self.drone_id
        self.savoir.assert_known(f"{sid}.pos_x",    0.0,   "init")
        self.savoir.assert_known(f"{sid}.pos_y",    0.0,   "init")
        self.savoir.assert_known(f"{sid}.pos_z",    0.0,   "init")
        self.savoir.assert_known(f"{sid}.battery",  100.0, "sensor")
        self.savoir.assert_known(f"{sid}.vel_x",    0.0,   "imu")
        self.savoir.assert_known(f"{sid}.vel_y",    0.0,   "imu")
        self.savoir.assert_known(f"{sid}.vel_z",    0.0,   "imu")
        self.savoir.assert_known(f"{sid}.armed",    False, "fc")
        self.savoir.assert_known(f"{sid}.airborne", False, "fc")

    def sensor_update(self, pos: Tuple[float,float,float],
                       battery: float, gps_lock: bool):
        """
        Update state from physical sensors — this is KNOWN, not estimated.
        The drone KNOWS its position from GPS+IMU.
        """
        sid = self.drone_id
        self.state.x, self.state.y, self.state.z = pos
        self.state.battery_pct = battery
        self.set_dynamic("battery",  battery)
        self.set_dynamic("gps_lock", gps_lock)

        # Update SAVOIR with KNOWN certainty (sensor-verified)
        cert = CertaintyLevel.KNOWN if gps_lock else CertaintyLevel.PROBABLE
        self.savoir.assert_known(f"{sid}.pos_x", pos[0], "gps+imu")
        self.savoir.assert_known(f"{sid}.pos_y", pos[1], "gps+imu")
        self.savoir.assert_known(f"{sid}.pos_z", pos[2], "gps+imu")
        self.savoir.assert_known(f"{sid}.battery", battery, "sensor")
        self.savoir.state_matrix.set(sid, "pos_x", pos[0], cert.value)
        self.savoir.state_matrix.set(sid, "pos_y", pos[1], cert.value)
        self.savoir.state_matrix.set(sid, "pos_z", pos[2], cert.value)
        self.savoir.state_matrix.set(sid, "battery", battery, 1.0)

    async def on_impacted(self, reaction: Reaction, frame: Dict):
        at = reaction.action_type
        sid = self.drone_id
        payload = frame.get("payload", {})

        if at == "drone.move" and reaction.is_success:
            # KNOWN outcome — we commanded this, IMU confirms it
            self.savoir.record_action_outcome(
                "drone.move", sid, payload, CertaintyLevel.KNOWN)
            # Update DroneState
            self.state.x += float(payload.get("dx", 0))
            self.state.y += float(payload.get("dy", 0))
            self.state.z += float(payload.get("dz", 0))
            self.state.battery_pct -= 0.5
            self.set_dynamic("battery", self.state.battery_pct)

        elif at == "drone.takeoff" and reaction.is_success:
            self.state.z  = float(payload.get("altitude", 5.0))
            self.state.armed    = True
            self.state.airborne = True
            self.savoir.assert_known(f"{sid}.pos_z",   self.state.z, "fc")
            self.savoir.assert_known(f"{sid}.armed",   True,         "fc")
            self.savoir.assert_known(f"{sid}.airborne",True,         "fc")
            self.set_dynamic("airborne", True)

        elif at == "drone.land" and reaction.is_success:
            self.state.z        = 0.0
            self.state.airborne = False
            self.savoir.assert_known(f"{sid}.pos_z",    0.0,  "fc")
            self.savoir.assert_known(f"{sid}.airborne", False, "fc")
            self.set_dynamic("airborne", False)


# ─── Swarm Environment ────────────────────────────────────────────────────────

class DroneSwarmEnv(Environment):
    """
    EDP Environment for a drone swarm.

    Contexts:
      • Pre-flight     [normative]  — mission planning, checklist
      • Flight         [spatial]    — primary operational context
      • Navigation     [temporal]   — waypoint following
      • Emergency      [causal]     — immediate threat response
      • Scan           [spatial]    — sensor/camera operations
      • Return         [temporal]   — RTH (Return to Home)

    Circumstances:
      • battery_safe         → battery > 15% (hard gate — ALL flight actions)
      • gps_lock             → GPS lock confirmed (hard gate — navigation)
      • airborne             → drone is in the air
      • no_critical_alarm    → no emergency circumstances active
      • mission_active       → mission is running

    The battery circumstance is a perfect example of EDP gates:
    When battery < 15%, ALL flight actions DISAPPEAR from available actions.
    The drone cannot attempt a maneuver — it's not blocked, it's invisible.
    """

    def __init__(self, drones: List[Drone]):
        super().__init__("DroneSwarm", EnvironmentKind.LIVING)
        self.drones = {d.drone_id: d for d in drones}
        self._build_contexts()
        self._build_phenomena()

    def _build_contexts(self):
        # ── Circumstances ─────────────────────────────────────────────────
        def battery_ok(ctx, frame):
            did = frame.get("actor_id","")
            d = self.drones.get(did)
            return d.state.battery_pct > 15.0 if d else True

        def gps_ok(ctx, frame):
            did = frame.get("actor_id","")
            d = self.drones.get(did)
            return d.state.gps_lock if d else True

        def is_airborne(ctx, frame):
            did = frame.get("actor_id","")
            d = self.drones.get(did)
            return d.state.airborne if d else False

        def not_airborne(ctx, frame):
            return not is_airborne(ctx, frame)

        C_BATTERY  = Circumstance.when("battery.safe","Battery > 15%", battery_ok)
        C_GPS      = Circumstance.when("gps.lock",    "GPS lock active", gps_ok)
        C_AIRBORNE = Circumstance.when("is.airborne", "Drone is airborne", is_airborne)
        C_GROUND   = Circumstance.when("on.ground",   "Drone on ground",  not_airborne)

        # Emergency compound: battery critical OR GPS lost
        C_EMERGENCY = (~C_BATTERY) | (~C_GPS)

        # ── Contexts ──────────────────────────────────────────────────────

        self.preflight_ctx = self.create_context(
            "PreFlight", ContextKind.GLOBAL,
            basis=DroneS.mission("pre-flight operations", 0.8),
            circumstances=[C_GROUND])

        self.flight_ctx = self.create_context(
            "Flight", ContextKind.SPATIAL,
            basis=DroneS.spatial("primary flight ops", 0.95),
            circumstances=[C_BATTERY, C_GPS, C_AIRBORNE])

        self.navigation_ctx = self.create_context(
            "Navigation", ContextKind.TEMPORAL,
            basis=DroneS.navigation("waypoint navigation", 0.9),
            circumstances=[C_BATTERY, C_GPS, C_AIRBORNE])

        self.emergency_ctx = self.create_context(
            "Emergency", ContextKind.CAUSAL,
            basis=DroneS.emergency("emergency response", 0.99),
            circumstances=[])  # ALWAYS available — no gates on emergency

        self.scan_ctx = self.create_context(
            "Scan", ContextKind.SPATIAL,
            basis=DroneS.spatial("scanning operations", 0.85),
            circumstances=[C_BATTERY, C_GPS, C_AIRBORNE])

        # ── Register Actions ───────────────────────────────────────────────

        self._register_flight_actions()
        self._register_navigation_actions()
        self._register_emergency_actions()
        self._register_scan_actions()
        self._register_preflight_actions()

    def _register_flight_actions(self):
        async def h_takeoff(actor, payload, ctx, frame):
            altitude = float(payload.get("altitude", 5.0))
            d = self.drones.get(actor.element_id)
            if not d: return Reaction.reject("drone.takeoff","Drone not found")
            if d.state.airborne: return Reaction.reject("drone.takeoff","Already airborne")
            d.state.mode = "GUIDED"
            return Reaction.ok("drone.takeoff",
                f"{d.name} taking off to {altitude}m",
                result={"altitude":altitude,"drone":d.name},
                sense=DroneS.spatial("ascending", 0.8),
                impact=ImpactScope.on_actor(1.0),
                temporality=Temporality.deferred(int(altitude*500)))  # ~0.5s/m

        async def h_move(actor, payload, ctx, frame):
            d = self.drones.get(actor.element_id)
            if not d: return Reaction.reject("drone.move","Drone not found")
            dx=float(payload.get("dx",0)); dy=float(payload.get("dy",0))
            dz=float(payload.get("dz",0)); speed=float(payload.get("speed",2.0))
            dist = math.sqrt(dx**2+dy**2+dz**2)
            eta  = dist/speed if speed > 0 else 0
            # SAVOIR: after this action, position is KNOWN
            prev = SenseVector.spatial(f"pos@{d.state.x:.1f},{d.state.y:.1f}", 0.5)
            npos = SenseVector.spatial(f"pos@{d.state.x+dx:.1f},{d.state.y+dy:.1f}", 0.95)
            delta = prev.delta(npos)
            r = Reaction.ok("drone.move",
                f"{d.name} → Δ({dx:.1f},{dy:.1f},{dz:.1f}) dist={dist:.1f}m ETA={eta:.1f}s",
                result={"dx":dx,"dy":dy,"dz":dz,"eta_s":round(eta,2),"drone":d.name},
                sense=DroneS.spatial("translating", 0.9),
                impact=ImpactScope.on_actor(0.8))
            r.causal_delta = delta  # Δ_t — position change
            return r

        async def h_land(actor, payload, ctx, frame):
            d = self.drones.get(actor.element_id)
            if not d: return Reaction.reject("drone.land","Drone not found")
            return Reaction.ok("drone.land",
                f"{d.name} landing",
                sense=DroneS.spatial("descending", 0.7),
                impact=ImpactScope.on_actor(1.0),
                temporality=Temporality.deferred(3000))

        async def h_hover(actor, payload, ctx, frame):
            d = self.drones.get(actor.element_id)
            if not d: return Reaction.reject("drone.hover","Drone not found")
            return Reaction.ok("drone.hover",
                f"{d.name} holding position at z={d.state.z:.1f}m",
                result={"position":d.state.position},
                sense=DroneS.spatial("stable hover", 0.6),
                impact=ImpactScope.on_actor(0.3))

        for a in [
            Action("drone.takeoff", ActionCategory.COMMAND, "Take off to altitude",
                   DroneS.spatial("takeoff operation", 0.85), handler=h_takeoff),
            Action("drone.move",    ActionCategory.COMMAND, "Move to relative position",
                   DroneS.spatial("translational movement", 0.95), handler=h_move),
            Action("drone.land",    ActionCategory.COMMAND, "Land at current position",
                   DroneS.spatial("landing operation", 0.75), handler=h_land),
            Action("drone.hover",   ActionCategory.COMMAND, "Hold current position",
                   DroneS.spatial("station keeping", 0.60), handler=h_hover),
        ]:
            self.flight_ctx.reg(a)
            if a.type != "drone.takeoff": self.navigation_ctx.reg(a)

        self.preflight_ctx.reg(
            Action("drone.takeoff", ActionCategory.COMMAND, "Take off to altitude",
                   DroneS.spatial("takeoff", 0.85), handler=h_takeoff))

    def _register_navigation_actions(self):
        async def h_goto(actor, payload, ctx, frame):
            d = self.drones.get(actor.element_id)
            if not d: return Reaction.reject("drone.goto","Drone not found")
            tx=float(payload.get("x",0)); ty=float(payload.get("y",0))
            tz=float(payload.get("z", d.state.z))
            dist=math.sqrt((tx-d.state.x)**2+(ty-d.state.y)**2+(tz-d.state.z)**2)
            return Reaction.ok("drone.goto",
                f"{d.name} navigating to ({tx:.1f},{ty:.1f},{tz:.1f}) dist={dist:.1f}m",
                result={"target":(tx,ty,tz),"dist":round(dist,2)},
                sense=DroneS.navigation("waypoint navigation", 0.95),
                impact=ImpactScope.on_actor(0.9))

        async def h_orbit(actor, payload, ctx, frame):
            d = self.drones.get(actor.element_id)
            if not d: return Reaction.reject("drone.orbit","Drone not found")
            cx=float(payload.get("cx",0)); cy=float(payload.get("cy",0))
            r=float(payload.get("radius",10.0))
            return Reaction.ok("drone.orbit",
                f"{d.name} orbiting ({cx:.1f},{cy:.1f}) r={r:.1f}m",
                sense=DroneS.navigation("circular orbit", 0.8),
                impact=ImpactScope.on_actor(0.5))

        self.navigation_ctx.reg(
            Action("drone.goto",  ActionCategory.COMMAND, "Navigate to absolute position",
                   DroneS.navigation("waypoint navigation", 0.95), handler=h_goto))
        self.navigation_ctx.reg(
            Action("drone.orbit", ActionCategory.COMMAND, "Orbit a point of interest",
                   DroneS.navigation("orbit pattern", 0.80), handler=h_orbit))

    def _register_emergency_actions(self):
        async def h_rth(actor, payload, ctx, frame):
            d = self.drones.get(actor.element_id)
            if not d: return Reaction.reject("drone.rth","Drone not found")
            return Reaction.ok("drone.rth",
                f"{d.name} returning to home — EMERGENCY RTH",
                sense=DroneS.emergency("emergency return", 1.0),
                impact=ImpactScope.on_actor(1.0))

        async def h_estop(actor, payload, ctx, frame):
            d = self.drones.get(actor.element_id)
            if not d: return Reaction.reject("drone.emergency_stop","Drone not found")
            d.state.armed = False
            return Reaction.ok("drone.emergency_stop",
                f"{d.name} EMERGENCY STOP — motors cut",
                sense=DroneS.emergency("immediate stop", 1.0),
                impact=ImpactScope.on_actor(1.0),
                temporality=Temporality.immediate())

        for a in [
            Action("drone.rth",           ActionCategory.SIGNAL,
                   "Return to home immediately",
                   DroneS.emergency("emergency RTH", 1.0),   handler=h_rth),
            Action("drone.emergency_stop",ActionCategory.SIGNAL,
                   "Cut motors — emergency stop",
                   DroneS.emergency("motor cutoff", 0.99),   handler=h_estop),
        ]:
            for ctx in [self.emergency_ctx, self.flight_ctx, self.navigation_ctx]:
                ctx.reg(a)

    def _register_scan_actions(self):
        async def h_scan(actor, payload, ctx, frame):
            d = self.drones.get(actor.element_id)
            if not d: return Reaction.reject("drone.scan","Drone not found")
            scan_type = payload.get("type","visual")
            area = payload.get("area","current_zone")
            return Reaction.ok("drone.scan",
                f"{d.name} scanning [{scan_type}] area={area}",
                result={"drone":d.name,"scan_type":scan_type,"area":area},
                sense=DroneS.telemetry("sensor scan", 0.7),
                impact=ImpactScope.on_env(0.4))

        async def h_photo(actor, payload, ctx, frame):
            d = self.drones.get(actor.element_id)
            if not d: return Reaction.reject("drone.photo","Drone not found")
            return Reaction.ok("drone.photo",
                f"{d.name} capturing at ({d.state.x:.1f},{d.state.y:.1f},{d.state.z:.1f})",
                result={"position":d.state.position},
                sense=DroneS.telemetry("photo capture", 0.5),
                impact=ImpactScope.on_env(0.2))

        self.scan_ctx.reg(Action("drone.scan",  ActionCategory.QUERY,
            "Scan area with sensors",  DroneS.telemetry("area scan",0.7),   handler=h_scan))
        self.scan_ctx.reg(Action("drone.photo", ActionCategory.QUERY,
            "Capture photo/video",     DroneS.telemetry("photo capture",0.5), handler=h_photo))

    def _register_preflight_actions(self):
        async def h_arm(actor, payload, ctx, frame):
            d = self.drones.get(actor.element_id)
            if not d: return Reaction.reject("drone.arm","Drone not found")
            d.state.armed = True
            d.savoir.assert_known(f"{actor.element_id}.armed", True, "fc")
            return Reaction.ok("drone.arm",
                f"{d.name} armed — ready for flight",
                sense=DroneS.mission("pre-flight arm", 0.7),
                impact=ImpactScope.on_actor(1.0))

        async def h_preflight_check(actor, payload, ctx, frame):
            d = self.drones.get(actor.element_id)
            if not d: return Reaction.reject("drone.preflight","Drone not found")
            checks = {
                "battery": d.state.battery_pct > 50,
                "gps":     d.state.gps_lock,
                "sensors": True,  # assume OK
            }
            all_ok = all(checks.values())
            return Reaction.ok("drone.preflight_check",
                f"Preflight {'PASS' if all_ok else 'FAIL'}: {checks}",
                result=checks, sense=DroneS.mission("preflight", 0.9),
                impact=ImpactScope.on_actor(0.2))

        self.preflight_ctx.reg(Action("drone.arm", ActionCategory.COMMAND,
            "Arm motors", DroneS.mission("arm", 0.7), handler=h_arm))
        self.preflight_ctx.reg(Action("drone.preflight_check", ActionCategory.QUERY,
            "Run preflight checklist", DroneS.mission("preflight check", 0.9),
            handler=h_preflight_check))

    def _build_phenomena(self):
        """Register emergent pattern detectors for swarm."""
        # Navigation failures cascading across multiple drones
        self.register_pattern(PhenomenonPattern(
            "NavigationFailureCascade", "rejected", threshold=3, window_s=120,
            attractor=SenseVector.emergent("nav cascade failure")))
        # Battery drain cluster — multiple drones going critical
        self.register_pattern(PhenomenonPattern(
            "BatteryCluster", "rejected", threshold=2, window_s=60,
            attractor=SenseVector.emergent("battery emergency cluster")))


# ─── Context Topology for Drones ──────────────────────────────────────────────

def drone_context_topology(env: DroneSwarmEnv):
    """
    Show semantic distances between drone contexts.
    Emergency context is near Causal axis.
    Navigation context is near Temporal axis.
    These distances guide context selection when the AI must choose.
    """
    ctxs = [env.preflight_ctx, env.flight_ctx, env.navigation_ctx,
            env.emergency_ctx, env.scan_ctx]
    print("\n  Drone Context Topology (angular distance):")
    print(f"  {'Context A':<18} ↔ {'Context B':<18} {'Dist':>6}  Proximity")
    print("  " + "─"*58)
    for i,a in enumerate(ctxs):
        for b in ctxs[i+1:]:
            d=a.basis.angular_distance(b.basis)
            bar="█"*int((1-d)*18)
            print(f"  {a.name:<18} ↔ {b.name:<18} {d:>6.3f}  {bar}")


# ─── Swarm-level SAVOIR ───────────────────────────────────────────────────────

class SwarmSavoir:
    """
    Shared certainty layer for a drone swarm.
    Tracks: which drone knows where, inter-drone distances, shared observations.
    Enables cross-drone phenomenon detection.
    """

    def __init__(self, drones: List[Drone]):
        self.drones  = {d.drone_id: d for d in drones}
        # Shared state matrix: all drones × [pos_x, pos_y, pos_z, battery]
        self.shared = EnvironmentalStateMatrix(
            [d.drone_id for d in drones],
            ["pos_x","pos_y","pos_z","battery"])

    def sync_from_drones(self):
        """Pull KNOWN positions from each drone's individual SAVOIR."""
        for did, drone in self.drones.items():
            for dim in ["pos_x","pos_y","pos_z","battery"]:
                val = drone.savoir.value_of(f"{did}.{dim}", 0.)
                cert = drone.savoir.certainty_of(f"{did}.{dim}")
                self.shared.set(did, dim, float(val), cert)

    def inter_drone_distance(self, id_a: str, id_b: str) -> Optional[float]:
        """Compute distance between two drones — only if positions are known."""
        a = self.drones.get(id_a); b = self.drones.get(id_b)
        if not a or not b: return None
        # Only compute if positions are at least PROBABLE certainty
        if a.savoir.certainty_of(f"{id_a}.pos_x") < 0.70: return None
        if b.savoir.certainty_of(f"{id_b}.pos_x") < 0.70: return None
        dx = a.state.x - b.state.x; dy = a.state.y - b.state.y
        dz = a.state.z - b.state.z
        return math.sqrt(dx**2+dy**2+dz**2)

    def env_embedding(self) -> List[float]:
        """
        φ_swarm = flatten(M_Σ) — the swarm state as a flat embedding vector.
        This is the persistent numerical memory of swarm state.
        Can be used as input to an ML model or a semantic similarity search.
        """
        self.sync_from_drones()
        return self.shared.flatten()

    def savoir_summary(self) -> str:
        self.sync_from_drones()
        lines = ["SWARM SAVOIR:", ""]
        for did, drone in self.drones.items():
            pos = f"({drone.state.x:.1f},{drone.state.y:.1f},{drone.state.z:.1f})"
            cert= drone.savoir.certainty_of(f"{did}.pos_x")
            bat = drone.savoir.value_of(f"{did}.battery",0)
            known = drone.savoir.known_count
            lines.append(f"  {drone.name:<15} pos={pos}  batt={bat:.0f}%"
                         f"  pos_cert={cert:.2f}  known_facts={known}")
        emb = self.env_embedding()
        lines.append(f"\n  Swarm embedding: φ_swarm ∈ ℝ^{len(emb)}"
                     f"  avg_certainty={self.shared.average_certainty():.2f}")
        return "\n".join(lines)


# ─── Demonstration ────────────────────────────────────────────────────────────

async def drone_demo():
    """
    Full demonstration of EDP + SAVOIR for drone swarm.
    """
    print("\n" + "═"*65)
    print("  EDP v4.1 + SAVOIR — Drone Swarm Demo")
    print("═"*65 + "\n")

    # Create swarm
    drones = [Drone(f"ALPHA-{i}", f"drone_{i}") for i in range(3)]
    env    = DroneSwarmEnv(drones)
    gw     = MepGateway(env)
    swarm_sav = SwarmSavoir(drones)

    # Admit drones
    for d in drones:
        await env.admit(d)
        env.preflight_ctx.include(d.to_dict())
        env.flight_ctx.include(d.to_dict())
        env.navigation_ctx.include(d.to_dict())
        env.emergency_ctx.include(d.to_dict())
        env.scan_ctx.include(d.to_dict())

    env.on_reaction(lambda r: print(f"  [RXN] {r.line()}"))
    env.on_phenomenon(lambda p: print(f"  [PHN] ⚡ {p.name} mag={p.magnitude:.0%}"))

    # ── 1. SAVOIR initial state ────────────────────────────────────────────
    print("── 1. Initial SAVOIR Knowledge ───────────────────────────────────")
    print(swarm_sav.savoir_summary())

    # ── 2. Context topology ────────────────────────────────────────────────
    print("\n── 2. Context Topology ───────────────────────────────────────────")
    drone_context_topology(env)

    # ── 3. Preflight — context-gated actions ──────────────────────────────
    print("\n── 3. Preflight Actions (alpha drone) ────────────────────────────")
    alpha = drones[0]
    session = gw.connect("mission-ai")

    # Show available actions
    frame = {"actor_id": alpha.element_id}
    available = env.preflight_ctx.get_available_actions(alpha.to_dict(), frame)
    print(f"\n  Pre-flight context actions (harmony-ranked):")
    for a, h in available:
        print(f"    H={h.score:+.3f}  {a.type:<30} {a.description}")

    # Preflight check
    await env.dispatch(alpha, "drone.preflight_check", {}, env.preflight_ctx)

    # Arm + takeoff
    await env.dispatch(alpha, "drone.arm", {}, env.preflight_ctx)
    await env.dispatch(alpha, "drone.takeoff", {"altitude": 10.0}, env.preflight_ctx)

    # ── 4. SAVOIR after takeoff — positions are KNOWN ─────────────────────
    print("\n── 4. SAVOIR After Takeoff ───────────────────────────────────────")
    print(alpha.savoir.to_llm_context())

    # ── 5. Flight with position tracking ─────────────────────────────────
    print("\n── 5. Flight Operations + Causal Δ Tracking ─────────────────────")
    moves = [(10,0,0,"North"), (0,10,0,"East"), (-5,-5,2,"NW+climb")]
    for dx,dy,dz,label in moves:
        await env.dispatch(alpha, "drone.move",
            {"dx":dx,"dy":dy,"dz":dz,"speed":3.0}, env.flight_ctx)
        print(f"  Position after {label}: "
              f"({alpha.state.x:.1f},{alpha.state.y:.1f},{alpha.state.z:.1f})"
              f"  pos_cert={alpha.savoir.certainty_of(f'{alpha.drone_id}.pos_x'):.2f}")

    # ── 6. WHY-NOT: battery gate ───────────────────────────────────────────
    print("\n── 6. WHY-NOT: Battery Circumstance Gate ─────────────────────────")
    # Simulate low battery
    alpha.sensor_update((alpha.state.x,alpha.state.y,alpha.state.z),
                         12.0, True)  # battery at 12% < 15% threshold
    env_blocked = gw.build_envelope(session, alpha, env.flight_ctx)
    print(f"  Available in Flight ctx when battery=12%: {len(env_blocked.actions)} actions")
    if not env_blocked.actions:
        print(f"  → {alpha.name} battery=12% → ALL flight actions INVISIBLE")
        print(f"  → WHY-NOT: battery.safe circumstance evaluates to FALSE")

    # Emergency still available (no gates on emergency)
    env_emerg = gw.build_envelope(session, alpha, env.emergency_ctx)
    print(f"  Emergency context actions (always available): {len(env_emerg.actions)}")
    for a in env_emerg.actions:
        print(f"    ⚡ {a['type']}")

    # ── 7. Swarm SAVOIR embedding ──────────────────────────────────────────
    print("\n── 7. Swarm Environmental Embedding ─────────────────────────────")
    swarm_sav.sync_from_drones()
    phi_env = swarm_sav.env_embedding()
    print(f"  φ_swarm ∈ ℝ^{len(phi_env)}"
          f"  (3 drones × 4 properties × 2 [value+certainty])")
    print(f"  avg_certainty = {swarm_sav.shared.average_certainty():.3f}")
    print(f"\n  This vector IS the persistent numerical memory of swarm state.")
    print(f"  No LLM re-estimation needed. Query this vector for any position.")

    # ── 8. Context-based action selection (topology) ──────────────────────
    print("\n── 8. Nearest Context to 'scan and photograph' goal ──────────────")
    goal_sense = SenseVector.technical("scan photograph", 0.7)
    # Find nearest context
    best_ctx = min([env.preflight_ctx, env.flight_ctx, env.navigation_ctx,
                    env.emergency_ctx, env.scan_ctx],
                   key=lambda c: goal_sense.angular_distance(c.basis))
    print(f"  Goal: 'scan and photograph area'")
    print(f"  → Best context: {best_ctx.name}  "
          f"dist={goal_sense.angular_distance(best_ctx.basis):.3f}")

    print("\n" + "═"*65)
    print("  Drone EDP demo complete.")
    print("  Key: positions KNOWN via SAVOIR after each action.")
    print("  No re-estimation. No probabilistic position guessing.")
    print("═"*65 + "\n")


if __name__ == "__main__":
    asyncio.run(drone_demo())
