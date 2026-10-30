"""Neural, semantic-loss, and proposed neuro-symbolic models (PyTorch, CPU).

Three families are provided behind one interface so the experiment runner can
treat them uniformly:

* ``MLPModel``        - plain deep tabular classifier.
* ``SemanticLossModel`` - differentiable-logic semantic loss (Xu et al., 2018)
                          over the rule base; knowledge is used as a soft
                          consistency penalty on the output only.
* ``NEXS`` (proposed)  - Neuro-symbolic EXpert System: a *differentiable rule
                          layer* whose per-rule confidences are injected with an
                          expert prior, fused additively with a neural head.

All rule reasoning uses the product t-norm for conjunction and the
probabilistic sum for disjunction, which keeps the layer differentiable and
lets gradients flow from the label back into the literals.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .knowledge import Domain, Literal, Rule

TRAIN_DEFAULTS = dict(epochs=300, lr=2e-3, weight_decay=1e-4, patience=40,
                      seed=0, batch_size=256)


# --------------------------------------------------------------------------- #
# differentiable logic primitives
# --------------------------------------------------------------------------- #
def _membership(X: torch.Tensor, lit: Literal, concepts: torch.Tensor | None,
                col: dict[str, int], tau: float = 0.10,
                ccol: dict[str, int] | None = None) -> torch.Tensor:
    if lit.kind == "concept":
        assert concepts is not None and ccol is not None
        return concepts[:, ccol[lit.var]]
    v = X[:, col[lit.var]]
    z = (v - lit.thr) / tau
    if lit.op == "<=":
        z = -z
    if lit.op == "==":
        z = 1.0 - 2.0 * (v - lit.thr).abs() / tau
    return torch.sigmoid(z)


def _body(X, concepts, rule: Rule, col, tau, ccol=None) -> torch.Tensor:
    m = torch.ones(X.shape[0], device=X.device)
    for lit in rule.body:
        m = m * _membership(X, lit, concepts, col, tau, ccol)
    return m


def _disjunction(mats: list[torch.Tensor]) -> torch.Tensor:
    if not mats:
        return torch.zeros(1)
    keep = torch.ones_like(mats[0])
    for m in mats:
        keep = keep * (1.0 - m)
    return 1.0 - keep


# --------------------------------------------------------------------------- #
# plain MLP
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    def __init__(self, d_in: int, hidden=(128, 64), dropout=0.1):
        super().__init__()
        layers: list[nn.Module] = []
        prev = d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# --------------------------------------------------------------------------- #
# model wrappers with a common fit/predict interface
# --------------------------------------------------------------------------- #
class _TorchClassifier:
    """Shared training loop; subclasses implement ``_loss`` and ``forward``."""

    name = "torch"

    def __init__(self, domain: Domain, **kw):
        self.domain = domain
        self.kw = {**TRAIN_DEFAULTS, **kw}
        self.col = {f: i for i, f in enumerate(domain.features)}
        self.ccol = {c: i for i, c in enumerate(domain.concepts)}
        torch.manual_seed(self.kw["seed"])
        np.random.seed(self.kw["seed"])

    # subclasses build self.model and optionally self.extra_params
    def _build(self, d_in: int) -> None:
        raise NotImplementedError

    def _loss(self, logits, y, x, extra):
        return nn.functional.binary_cross_entropy_with_logits(logits, y)

    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None) -> "TorchClassifier":
        self._build(X.shape[1])
        self.model = getattr(self, "model", None) or self.model
        params = list(self.model.parameters())
        if getattr(self, "extra_params", None):
            params += list(self.extra_params())
        opt = torch.optim.AdamW(params, lr=self.kw["lr"], weight_decay=self.kw["weight_decay"])
        xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        xv = torch.tensor(X_val, dtype=torch.float32) if X_val is not None else None
        yv = torch.tensor(y_val, dtype=torch.float32) if y_val is not None else None
        bs = min(self.kw["batch_size"], len(xt))
        g = torch.Generator().manual_seed(self.kw["seed"])

        best, best_state, wait = np.inf, None, 0
        for _ in range(self.kw["epochs"]):
            self.model.train()
            perm = torch.randperm(len(xt), generator=g)
            for i in range(0, len(xt), bs):
                idx = perm[i:i + bs]
                xb, yb = xt[idx], yt[idx]
                opt.zero_grad()
                loss = self._loss(self.model(xb), yb, xb, self._extra(xb))
                loss.backward()
                opt.step()
            if xv is not None:
                self.model.eval()
                with torch.no_grad():
                    vl = self._loss(self.model(xv), yv, xv, self._extra(xv)).item()
                if vl < best - 1e-5:
                    best, wait = vl, 0
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    wait += 1
                    if wait >= self.kw["patience"]:
                        break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def _extra(self, x):
        return None

    def _fuse(self, logits, p_sat):
        sym = torch.clamp(self.alpha, 0.0, 10.0) * torch.logit(p_sat.clamp(1e-6, 1 - 1e-6))
        return sym if self.alpha_only else logits + sym

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X, dtype=torch.float32))
        return torch.sigmoid(logits).numpy()

    def predict(self, X: np.ndarray, thr: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= thr).astype(int)

    def rule_confidence(self) -> np.ndarray | None:
        return None


class MLPModel(_TorchClassifier):
    name = "MLP"

    def _build(self, d_in: int) -> None:
        self.model = MLP(d_in)


class SemanticLossModel(_TorchClassifier):
    """Semantic loss (Xu et al. 2018) + plain MLP head."""

    name = "SemanticLoss"

    def _build(self, d_in: int) -> None:
        self.model = MLP(d_in)
        self.lam = self.kw.get("lam", 1.0)

    def _label_prob(self, x, extra):
        return extra

    def _extra(self, x):
        return self._rule_prob(x, None)[1]

    def _rule_prob(self, x, concept_override):
        """Return (concept_probs [n,C], label_prob [n])."""
        n = x.shape[0]
        cprobs = []
        for c in self.domain.concepts:
            rules = [r for r in self.domain.expert_rules if r.head == c]
            cprobs.append(_disjunction([_body(x, None, r, self.col, 0.1, self.ccol) for r in rules]))
        concepts = torch.stack(cprobs, dim=1) if cprobs else torch.zeros(n, 0)
        lrules = [r for r in self.domain.expert_rules if r.head == self.domain.label]
        lprob = _disjunction([_body(x, concepts, r, self.col, 0.1, self.ccol) for r in lrules])
        return concepts, lprob

    def _loss(self, logits, y, x, extra):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, y)
        p = extra.clamp(1e-6, 1 - 1e-6)
        sem = -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()
        return bce + self.lam * sem


class NEXS(_TorchClassifier):
    """Proposed Neuro-symbolic EXpert System.

    Components
    ----------
    * differentiable rule layer producing a label probability ``p_sat``,
    * a per-rule confidence vector initialised from the expert ``cf`` values and
      regularised towards them (expert prior),
    * a neural head fused additively: ``z = f(x) + alpha * logit(p_sat)``.

    Ablations are obtained by zeroing ``lam_prior`` / ``lam_sem`` / ``alpha``.
    """

    name = "NEXS"

    def _build(self, d_in: int) -> None:
        self.model = MLP(d_in, hidden=self.kw.get("hidden", (128, 64)))
        rules = list(self.domain.expert_rules)
        self.rules = rules
        init = torch.tensor([r.cf for r in rules], dtype=torch.float32)
        self._cf = nn.Parameter(init.clone())
        self._prior = init.clone()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.lam_sem = self.kw.get("lam_sem", 0.5)
        self.lam_prior = self.kw.get("lam_prior", 1.0)
        self.alpha_only = self.kw.get("alpha_only", False)

    def extra_params(self):
        return [self._cf, self.alpha]

    def _cf_for(self, head: str) -> torch.Tensor:
        idx = [i for i, r in enumerate(self.rules) if r.head == head]
        return self._cf[idx]

    def _layer(self, x):
        concepts = []
        for c in self.domain.concepts:
            idx = [i for i, r in enumerate(self.rules) if r.head == c]
            cf = self._cf[idx]
            bodies = [_body(x, None, self.rules[i], self.col, 0.1, self.ccol) for i in idx]
            mats = [torch.sigmoid(cf[j]) * b for j, b in enumerate(bodies)]
            concepts.append(_disjunction(mats))
        cprob = torch.stack(concepts, dim=1) if concepts else torch.zeros(x.shape[0], 0)
        idx = [i for i, r in enumerate(self.rules) if r.head == self.domain.label]
        cf = self._cf[idx]
        mats = []
        for j, i in enumerate(idx):
            b = _body(x, cprob, self.rules[i], self.col, 0.1, self.ccol)
            mats.append(torch.sigmoid(cf[j]) * b)
        return cprob, _disjunction(mats)

    def _extra(self, x):
        return self._layer(x)[1]

    def _loss(self, logits, y, x, extra):
        cprob, p_sat = self._layer(x)
        z = self._fuse(logits, p_sat)
        task = nn.functional.binary_cross_entropy_with_logits(z, y)
        p = p_sat.clamp(1e-6, 1 - 1e-6)
        sem = -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()
        prior = ((self._cf - self._prior) ** 2).mean()
        return task + self.lam_sem * sem + self.lam_prior * prior

    def _fuse(self, logits, p_sat):
        sym = torch.clamp(self.alpha, 0.0, 10.0) * torch.logit(p_sat.clamp(1e-6, 1 - 1e-6))
        return sym if self.alpha_only else logits + sym

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32)
            logits = self.model(x)
            _, p_sat = self._layer(x)
            z = self._fuse(logits, p_sat)
        return torch.sigmoid(z).numpy()

    def rule_confidence(self) -> np.ndarray:
        return torch.sigmoid(self._cf).detach().numpy()
