"""Domain knowledge: features, intermediate concepts, and Horn-style rules.

Two benchmark domains are defined. Each domain has
  * observed features (some continuous, some binary),
  * latent intermediate concepts that are *not* observed at test time,
  * a rule base in which the label is the disjunction of a small number of
    conjunctive rules (a Horn-like fragment).

`TRUE_KB` is the rule base that *generates* the data.  `EXPERT_KB` is a
deliberately imperfect hand-written knowledge base (thresholds are rounded to
human-plausible values) used by the symbolic expert system and by the
neuro-symbolic models.  Keeping the two apart lets us measure how well each
method copes with approximate expert knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal as TLiteral

Op = TLiteral[">=", "<=", "=="]


@dataclass(frozen=True)
class Literal:
    var: str
    op: Op
    thr: float
    kind: TLiteral["feature", "concept"] = "feature"

    def __str__(self) -> str:
        sym = {"<=": "<=", ">=": ">=", "==": "="}[self.op]
        return f"{self.var}{sym}{self.thr:g}"


@dataclass(frozen=True)
class Rule:
    head: str
    body: tuple[Literal, ...]
    cf: float = 1.0

    def __str__(self) -> str:
        return f"IF {' AND '.join(map(str, self.body))} THEN {self.head} (cf={self.cf:g})"


@dataclass(frozen=True)
class Domain:
    name: str
    features: tuple[str, ...]
    concepts: tuple[str, ...]
    binary_features: tuple[str, ...]
    true_rules: tuple[Rule, ...]
    expert_rules: tuple[Rule, ...]
    label: str = "risk"
    # how strongly each concept / the label is realised before label noise
    label_noise: float = 0.04
    meta: dict = field(default_factory=dict)


def _f(var, op, thr):
    return Literal(var, op, thr, "feature")


def _c(var, op, thr):
    return Literal(var, op, thr, "concept")


# ---------------------------------------------------------------------------
# Domain 1 - clinical triage (heart-risk style, 8 features, 2 concepts)
# ---------------------------------------------------------------------------
MEDICAL = Domain(
    name="medical",
    features=(
        "age", "systolic_bp", "cholesterol", "max_hr",
        "st_depression", "bmi", "smoking", "diabetes",
    ),
    concepts=("hypertension", "dyslipidemia"),
    binary_features=("smoking", "diabetes"),
    true_rules=(
        Rule("hypertension", (_f("age", ">=", 0.60), _f("systolic_bp", ">=", 0.60))),
        Rule("dyslipidemia", (_f("cholesterol", ">=", 0.62), _f("smoking", "==", 1.0))),
        Rule("risk", (_c("hypertension", ">=", 0.5), _f("diabetes", "==", 1.0))),
        Rule("risk", (_c("dyslipidemia", ">=", 0.5), _f("age", ">=", 0.55))),
        Rule("risk", (_f("st_depression", ">=", 0.60), _f("max_hr", "<=", 0.42))),
    ),
    expert_rules=(
        Rule("hypertension", (_f("age", ">=", 0.55), _f("systolic_bp", ">=", 0.60)), cf=0.9),
        Rule("dyslipidemia", (_f("cholesterol", ">=", 0.60), _f("smoking", "==", 1.0)), cf=0.9),
        Rule("risk", (_c("hypertension", ">=", 0.5), _f("diabetes", "==", 1.0)), cf=0.95),
        Rule("risk", (_c("dyslipidemia", ">=", 0.5), _f("age", ">=", 0.50)), cf=0.85),
        Rule("risk", (_f("st_depression", ">=", 0.55), _f("max_hr", "<=", 0.45)), cf=0.9),
    ),
    meta={"label": "high cardiac risk", "application": "primary-care triage"},
)

# ---------------------------------------------------------------------------
# Domain 2 - industrial process fault diagnosis (6 sensors, 2 concepts)
# ---------------------------------------------------------------------------
INDUSTRIAL = Domain(
    name="industrial",
    features=("temperature", "pressure", "vibration", "flow", "current", "rpm"),
    concepts=("overheat", "bearing_wear"),
    binary_features=(),
    true_rules=(
        Rule("overheat", (_f("temperature", ">=", 0.70), _f("pressure", ">=", 0.65))),
        Rule("bearing_wear", (_f("vibration", ">=", 0.60), _f("rpm", ">=", 0.70))),
        Rule("risk", (_c("overheat", ">=", 0.5), _f("vibration", ">=", 0.50))),
        Rule("risk", (_c("bearing_wear", ">=", 0.5), _f("current", ">=", 0.60))),
        Rule("risk", (_f("flow", "<=", 0.30), _f("pressure", ">=", 0.70))),
    ),
    expert_rules=(
        Rule("overheat", (_f("temperature", ">=", 0.68), _f("pressure", ">=", 0.62)), cf=0.9),
        Rule("bearing_wear", (_f("vibration", ">=", 0.58), _f("rpm", ">=", 0.68)), cf=0.9),
        Rule("risk", (_c("overheat", ">=", 0.5), _f("vibration", ">=", 0.48)), cf=0.9),
        Rule("risk", (_c("bearing_wear", ">=", 0.5), _f("current", ">=", 0.58)), cf=0.9),
        Rule("risk", (_f("flow", "<=", 0.32), _f("pressure", ">=", 0.68)), cf=0.85),
    ),
    meta={"label": "fault", "application": "predictive maintenance"},
)

DOMAINS: dict[str, Domain] = {d.name: d for d in (MEDICAL, INDUSTRIAL)}


def rule_text(domain: Domain, source: str = "expert") -> str:
    rules = domain.expert_rules if source == "expert" else domain.true_rules
    return "\n".join(str(r) for r in rules)
