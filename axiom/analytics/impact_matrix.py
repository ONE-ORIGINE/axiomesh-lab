"""
impact_matrix.py  —  Causal Impact Matrix & Session Analytics
════════════════════════════════════════════════════════════════════════════
OneOrigine / ImperialSchool Research  —  I.S. License

WHAT THIS IS
────────────
An optional analytics layer over any EDP environment session.

When enabled, it records every (action, reaction) pair with:
  • An impact score ∈ [-1, +1]  (positive = beneficial, negative = harmful)
  • The context in which it occurred
  • The chain depth
  • The causal delta Δ_t (if available from SAVOIR/EDP)

At end of session (or on demand), it produces:
  • Impact Matrix:  M_I[action][reaction] = mean_impact
  • Action Profile: per-action stats (count, mean_impact, success_rate)
  • Session Vector: φ_session ∈ ℝ^N — the session state as an embedding
  • Causal Export:  graph edges (action→reaction) for visualization

NOT RIGID: this is purely observational. It never blocks actions.
It provides intelligence AFTER the fact, enabling the agent to
learn better strategies over multiple sessions.
"""

from __future__ import annotations

import json, math, time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from edp import Reaction, ReactionStatus, SenseVector, SENSE_NULL

__version__ = "1.0.0"
__author__  = "OneOrigine"
__license__ = "I.S."


# ─── Impact Scoring ───────────────────────────────────────────────────────────

def default_impact_score(reaction: Reaction) -> float:
    """
    Default impact scorer: maps reaction status + sense to [-1, +1].

    Override this per domain:
      medical:   a failed medication reaction = -1.0
      drone:     a successful scan = +0.7
      finance:   a rejected trade = -0.3 (not catastrophic)
    """
    base = {
        ReactionStatus.SUCCESS:    +0.6,
        ReactionStatus.DEFERRED:   +0.2,
        ReactionStatus.PARTIAL:    +0.1,
        ReactionStatus.REJECTED:   -0.4,
        ReactionStatus.ERROR:      -0.8,
        ReactionStatus.CHAIN_MAX:  -0.6,
    }.get(reaction.status, 0.0)

    # Sense vector magnitude adjusts impact direction
    if hasattr(reaction, 'sense') and reaction.sense != SENSE_NULL:
        sense_factor = reaction.sense.magnitude
        # Causal axis (index 0) amplifies negative impact (cascade risk)
        causal = reaction.sense.v[0] if reaction.sense.v else 0.
        if base < 0 and causal > 0.5: base *= 1.3  # worse if causal chain
        base *= max(0.5, sense_factor)

    return max(-1.0, min(1.0, base))


# ─── Records ──────────────────────────────────────────────────────────────────

@dataclass
class ImpactRecord:
    """One action→reaction observation."""
    session_id    : str
    action_type   : str
    reaction_type : str
    status        : str
    impact_score  : float   # ∈ [-1, +1]
    context_name  : str
    chain_depth   : int
    timestamp     : float = field(default_factory=time.time)
    causal_delta  : Optional[float] = None  # |Δ_t| from SAVOIR if available


# ─── Impact Matrix ────────────────────────────────────────────────────────────

class ImpactMatrix:
    """
    M_I[action][reaction] = mean_impact_score

    Computed from all ImpactRecords in a session (or across sessions).
    Enables the agent to answer:
      "Which actions produce the most positive reactions in which contexts?"
      "Which action-reaction pairs are systematically negative?"

    NOT a rigid gate. A learning surface.
    """

    def __init__(self, records: List[ImpactRecord]):
        self._records = records
        self._matrix: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list))
        self._compute()

    def _compute(self):
        for r in self._records:
            self._matrix[r.action_type][r.reaction_type].append(r.impact_score)

    def mean_impact(self, action_type: str, reaction_type: str = "*") -> float:
        """Mean impact of action→reaction. reaction_type='*' = all reactions."""
        if reaction_type == "*":
            scores = [s for rxn_scores in self._matrix.get(action_type,{}).values()
                      for s in rxn_scores]
        else:
            scores = self._matrix.get(action_type,{}).get(reaction_type,[])
        return sum(scores)/len(scores) if scores else 0.

    def action_profile(self, action_type: str) -> Dict:
        rxns = self._matrix.get(action_type, {})
        all_scores = [s for scores in rxns.values() for s in scores]
        success_scores = [s for scores in rxns.values() for s in scores if s > 0]
        return {
            "action_type":   action_type,
            "total_count":   len(all_scores),
            "mean_impact":   round(sum(all_scores)/len(all_scores), 3) if all_scores else 0.,
            "success_rate":  round(len(success_scores)/len(all_scores), 3) if all_scores else 0.,
            "max_positive":  round(max(all_scores), 3) if all_scores else 0.,
            "min_negative":  round(min(all_scores), 3) if all_scores else 0.,
            "reactions":     {rxn: round(sum(s)/len(s),3) for rxn,s in rxns.items()},
        }

    def top_actions(self, n: int = 5) -> List[Tuple[str, float]]:
        """Top N actions by mean positive impact."""
        profiles = [(a, self.mean_impact(a)) for a in self._matrix]
        return sorted(profiles, key=lambda x: -x[1])[:n]

    def worst_actions(self, n: int = 5) -> List[Tuple[str, float]]:
        """Bottom N actions by mean impact."""
        profiles = [(a, self.mean_impact(a)) for a in self._matrix]
        return sorted(profiles, key=lambda x: x[1])[:n]

    def to_table(self, max_rows: int = 10) -> str:
        """Render as ASCII table."""
        all_actions = sorted(self._matrix.keys())[:max_rows]
        lines = [
            f"\n  {'Action':<32} {'Count':>6} {'Mean Impact':>12} {'Success%':>9} {'Best Rxn'}",
            "  " + "─"*75
        ]
        for a in all_actions:
            p = self.action_profile(a)
            bar_val = p["mean_impact"]
            bar_len = int(abs(bar_val) * 8)
            bar     = ("█"*bar_len) if bar_val >= 0 else ("▓"*bar_len)
            sign    = "+" if bar_val >= 0 else "-"
            best_rxn = max(p["reactions"], key=p["reactions"].get, default="—") if p["reactions"] else "—"
            lines.append(
                f"  {a:<32} {p['total_count']:>6} "
                f"  {sign}{abs(bar_val):.3f} {bar:<8} "
                f"{p['success_rate']*100:>7.1f}%  {best_rxn}")
        return "\n".join(lines)

    def to_matrix_export(self) -> Dict:
        """Export as JSON-serializable structure."""
        actions   = sorted(self._matrix.keys())
        reactions = sorted({rxn for rxn_map in self._matrix.values()
                            for rxn in rxn_map.keys()})
        matrix = [[round(self.mean_impact(a,r),4) for r in reactions] for a in actions]
        return {
            "rows": actions,
            "cols": reactions,
            "matrix": matrix,
            "description": "M_I[action][reaction] = mean_impact_score ∈ [-1,+1]"
        }

    def session_vector(self) -> List[float]:
        """
        φ_session ∈ ℝ^N — the session as an embedding vector.
        Each dimension = mean impact of one (action,reaction) pair.
        Enables session similarity, strategy learning, anomaly detection.
        """
        all_pairs = sorted(
            (a, r)
            for a, rxn_map in self._matrix.items()
            for r in rxn_map.keys())
        return [self.mean_impact(a,r) for a,r in all_pairs]


# ─── Causal Graph Export ──────────────────────────────────────────────────────

@dataclass
class CausalEdge:
    """One edge in the causal graph: action → reaction."""
    source      : str      # action_type
    target      : str      # reaction_type
    weight      : float    # mean_impact_score
    count       : int
    context     : str
    chain_depth : int

class CausalGraphExport:
    """
    Exports the session's causal graph as:
      • Edge list (action→reaction with weight)
      • Adjacency matrix
      • DOT format (for Graphviz)
      • JSON for D3.js or similar
    """

    def __init__(self, records: List[ImpactRecord]):
        self._records = records

    def edges(self) -> List[CausalEdge]:
        groups: Dict[Tuple[str,str,str], List[float]] = defaultdict(list)
        depths: Dict[Tuple[str,str,str], int] = {}
        for r in self._records:
            key = (r.action_type, r.reaction_type, r.context_name)
            groups[key].append(r.impact_score)
            depths[key] = r.chain_depth
        return [
            CausalEdge(k[0], k[1],
                       round(sum(s)/len(s),3), len(s), k[2], depths.get(k,0))
            for k, s in groups.items()
        ]

    def to_dot(self) -> str:
        """Export as Graphviz DOT format."""
        lines = ['digraph CausalGraph {', '  rankdir=LR;',
                 '  node [shape=box fontsize=10];']
        for e in self.edges():
            color = "green" if e.weight > 0 else "red"
            label = f"{e.weight:+.2f} ({e.count}x)"
            lines.append(f'  "{e.source}" -> "{e.target}" '
                         f'[label="{label}" color={color} weight={abs(e.weight)}];')
        lines.append("}")
        return "\n".join(lines)

    def to_json(self) -> str:
        """Export as JSON for D3.js or similar visualization."""
        edges = self.edges()
        nodes = list({e.source for e in edges} | {e.target for e in edges})
        return json.dumps({
            "nodes": [{"id":n,"group": 1 if n.endswith(".record") else 2} for n in nodes],
            "links": [{"source":e.source,"target":e.target,
                       "value":e.weight,"count":e.count} for e in edges]
        }, indent=2)


# ─── Session Tracker ──────────────────────────────────────────────────────────

class SessionTracker:
    """
    Attach to any EDP environment to track impact over a session.

    Usage:
      env = Environment("MyEnv", ...)
      tracker = SessionTracker("session-001")
      env.on_reaction(tracker.record_reaction)

      # After session:
      matrix = tracker.impact_matrix()
      print(matrix.to_table())
    """

    def __init__(self, session_id: str = "",
                  scorer: "Callable[[Reaction], float]" = None):
        self.session_id = session_id or str(__import__("uuid").uuid4())[:8]
        self._scorer    = scorer or default_impact_score
        self._records:  List[ImpactRecord] = []
        self._current_action: str = ""
        self._current_context: str = ""

    def set_action_context(self, action_type: str, context_name: str):
        """Call before dispatch to annotate the reaction."""
        self._current_action  = action_type
        self._current_context = context_name

    def record_reaction(self, reaction: Reaction):
        """Attach this to env.on_reaction(...)"""
        score  = self._scorer(reaction)
        delta  = reaction.causal_delta.magnitude if reaction.causal_delta else None
        record = ImpactRecord(
            session_id   = self.session_id,
            action_type  = self._current_action or reaction.action_type,
            reaction_type= reaction.action_type,
            status       = reaction.status.value,
            impact_score = score,
            context_name = self._current_context,
            chain_depth  = reaction.chain_depth,
            causal_delta = delta)
        self._records.append(record)

    def impact_matrix(self) -> ImpactMatrix:
        return ImpactMatrix(self._records)

    def causal_graph(self) -> CausalGraphExport:
        return CausalGraphExport(self._records)

    def summary(self) -> str:
        if not self._records: return "No records in session."
        m = self.impact_matrix()
        total = len(self._records)
        positive = sum(1 for r in self._records if r.impact_score > 0)
        mean_imp = sum(r.impact_score for r in self._records) / total
        top = m.top_actions(3)
        return (f"Session '{self.session_id}': {total} actions  "
                f"mean_impact={mean_imp:+.3f}  "
                f"positive_rate={positive/total:.0%}\n"
                f"Top actions: {', '.join(f'{a}({s:+.2f})' for a,s in top)}")

    @property
    def record_count(self) -> int: return len(self._records)
    @property
    def session_vector(self) -> List[float]: return self.impact_matrix().session_vector()
