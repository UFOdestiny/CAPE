"""CAPE — Coherent Answer Posterior Estimation (paper: "CAPE: Joint Outcome Estimation").

Answer model. For a question with K responses grouped into answer clusters c = 1..m, an answer
model is a conditional logit over "cluster c holds the correct answer" and an outside option ∅
("no sampled answer is correct"):

    P(c) = exp(u_c) / (exp(u_∅) + Σ_c' exp(u_c'))
    u_c = log(n_c / K) + mean_{i ∈ c} g(x_i)
    u_∅ = θ_0

the Bayes form under equicorrelated evidence (a constant outside-option weight), with the evidence of a
cluster averaged over its members. g(x) = ψ_f s(x): s is the hidden-state probe logit (cross-fitted logistic
regression on whitened principal components of the hidden states at the final answer token, middle layer, and at
the end of the reasoning block, three-quarter depth), and ψ_f its weight for the question's answer format f (math,
multiple choice, short answer), since the probe separates correct from incorrect responses unequally across formats.
A response's confidence is the posterior of its own cluster; responses without an extractable answer get 0. The
forecast is coherent by construction, contains self-consistency (g ≡ 0, no ∅) as a special case, and keeps
explicit mass for "all sampled answers are wrong". Parameters are fitted by maximum likelihood of the correct
cluster (or ∅) on the training pool; the L2 strength by validation NLL.

Optional shift gate (ablation). With ``evidence="gated"`` CAPE also fits a trace model without the probe and pools
the two posteriors log-linearly per question,

    log P(o) = (1 - w) log P_trace(o) + w log P_probe(o) - log Z,   o ∈ {clusters, ∅},
    w = 1[answer format occurs in the pool] · min(1, p / α),

where p is the conformal p-value of the question's hidden-state residual among the validation questions. On the
benchmark every answer format occurs in the pool and the gate changes the results by less than 0.01, so the method
uses the probe model alone.

The constructor switches (learned α, learned within-cluster correlation r, no ∅, learned ∅ features, no votes,
difficulty, judge, objective, trace statistics, evidence, gate) exist only for the ablations reported in the paper;
the defaults are the method.
"""
from __future__ import annotations

import itertools

import numpy as np

from cape.features import FEATURES, scalar_features
from uqbench.estimator import Estimator
from uqbench.records import labels_of

L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1)
EPOCHS, LR, SEED = 400, 0.05, 0

PROBE_COMPONENTS, PROBE_FOLDS, PROBE_C = 256, 5, (1e-3, 1e-2, 1e-1, 1.0)
# probe sites (replayed position, relative depth): position 0 = last token of the final answer, 1 = end of the
# reasoning block (see uqbench.pipeline.replay)
PROBE_SITES = ((0, 0.5), (1, 0.75))
FORMATS = ("math", "mc", "qa")   # answer formats with their own evidence weight ψ_f
SUPPORT_ALPHA = 0.05
_PROBE_IDS = itertools.count()   # unique per probe; id() can be reused once a probe is garbage-collected


def judge_evidence(q) -> tuple[np.ndarray, float]:
    """Answer-set judge (ablation; its verdict is attached as ``q.judge``): normalised log-probability of
    each response's own answer cluster (floor -20 for unanswered or unlisted) and of "none correct"."""
    j = getattr(q, "judge", None)
    if not j:
        return np.full(q.k, -20.0), 0.0
    lp = np.array(j["logp"] + [j["none_logp"]], dtype=np.float64)
    lp = lp - np.logaddexp.reduce(lp)
    own = dict(zip(j["clusters"], lp[:-1], strict=True))
    per = [own.get(c, -20.0) if s["pred"] is not None else -20.0 for c, s in zip(q.clusters, q.samples, strict=True)]
    return np.maximum(np.array(per), -20.0), float(lp[-1])


def probe_features(q) -> np.ndarray:
    """Hidden states at the probe sites (``PROBE_SITES``), concatenated."""
    layers = q.replay[0]["hidden"].shape[1]
    picks = [(p, round(depth * (layers - 1))) for p, depth in PROBE_SITES]
    return np.stack([np.concatenate([r["hidden"][p, j] for p, j in picks]) for r in q.replay]).astype(np.float32)


class HiddenProbe:
    """Logistic regression on whitened principal components of the hidden states at the probe sites.

    Training responses get cross-fitted (out-of-fold) logits, so the answer model that consumes the
    probe sees the same kind of score at training and test time. ``p_value`` is the conformal
    p-value of a question's residual (median over its responses of the distance to the principal
    subspace) among the validation questions."""

    def fit(self, train, val) -> HiddenProbe:
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression

        self.token = next(_PROBE_IDS)

        h = np.concatenate([probe_features(q) for q in train])
        y = labels_of(train)
        self.mu = h.mean(0)
        n_pc = min(PROBE_COMPONENTS, h.shape[1], h.shape[0] - 1)
        pca = PCA(n_pc, svd_solver="randomized", random_state=SEED).fit(h - self.mu)
        self.basis, self.scale = pca.components_, np.sqrt(pca.explained_variance_) + 1e-6
        z = self._whiten(h)
        fold = np.repeat(np.arange(len(train)), [q.k for q in train]) % PROBE_FOLDS
        best = None
        for c in PROBE_C:
            oof = np.zeros(len(y))
            for f in range(PROBE_FOLDS):
                lr = LogisticRegression(C=c, max_iter=3000).fit(z[fold != f], y[fold != f])
                oof[fold == f] = lr.decision_function(z[fold == f])
            p = np.clip(1 / (1 + np.exp(-oof)), 1e-9, 1 - 1e-9)
            nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
            if best is None or nll < best[0]:
                best = (nll, c, oof)
        self.nll, self.C, oof = best
        self.model = LogisticRegression(C=self.C, max_iter=3000).fit(z, y)
        samples = [s for q in train for s in q.samples]
        # keyed by object identity; the stored reference guards against a reused id()
        self.out_of_fold = {id(s): (s, v) for s, v in zip(samples, oof, strict=True)}
        self.calibration = np.array([self.residual(q) for q in val])
        return self

    def _whiten(self, h: np.ndarray) -> np.ndarray:
        return ((h - self.mu) @ self.basis.T) / self.scale

    def logits(self, q) -> np.ndarray:
        cache = getattr(q, "_cape_probe", {})
        if self.token not in cache:
            fresh = self.model.decision_function(self._whiten(probe_features(q)))
            cache[self.token] = np.array([self._out_of_fold(s, v) for s, v in zip(q.samples, fresh, strict=True)])
            q._cape_probe = cache
        return cache[self.token]

    def _out_of_fold(self, sample: dict, fresh: float) -> float:
        hit = self.out_of_fold.get(id(sample))
        return hit[1] if hit is not None and hit[0] is sample else fresh

    def residual(self, q) -> float:
        c = probe_features(q) - self.mu
        inside = ((c @ self.basis.T) ** 2).sum(1)
        return float(np.median(np.sqrt(np.maximum((c ** 2).sum(1) - inside, 0.0))))

    def p_value(self, q) -> float:
        return float((1 + np.sum(self.calibration >= self.residual(q))) / (1 + len(self.calibration)))

    def describe(self) -> dict:
        return {"components": len(self.scale), "C": self.C, "cross_fitted_nll": self.nll}


class AnswerModel:
    """The conditional logit over answer clusters and ∅ (module docstring), optionally with the
    hidden-state probe logit as one more evidence column."""

    def __init__(self, use_null: bool = True, use_votes: bool = True, use_difficulty: bool = False,
                 alpha: float | None = 1.0, features: list[str] | None = None, objective: str = "choice",
                 use_judge: bool = False, probe: HiddenProbe | None = None, null_features: bool = False,
                 format_weights: bool = True, learn_r: bool = False):
        self.use_null, self.use_votes = use_null, use_votes
        # ablation: True = cluster evidence scaled by κ(n) = n / (1 + (n - 1) r) with r learned (the posterior under
        # equicorrelated evidence); False = the cluster mean, its r → 1 limit
        self.learn_r = learn_r
        self.format_weights = format_weights  # ψ_f per answer format (False: one ψ for every format, ablation)
        self.null_features = null_features  # ablation: True = u_null also linear in mean g and the vote entropy
        self.use_judge = use_judge  # ablation: answer-set judge evidence in u_c (via g) and in u_null
        self.objective = objective  # "choice": log-likelihood of the correct cluster / null (default);
        #                             "response": per-response binary log loss of the same posterior
        self.use_difficulty = use_difficulty
        self.alpha_fixed = alpha  # 1.0 = exchangeable vote prior (default); None = learn α on the pool
        # evidence features used by g
        self.columns = [FEATURES.index(f) for f in (features or [])]   # trace statistics in g (ablation)
        self.probe = probe

    # ---------- features ----------
    def _features(self, q) -> np.ndarray:
        # trace statistics only when an ablation asks for them (they need the follow-up probes)
        x = scalar_features(q)[:, self.columns] if self.columns else np.zeros((q.k, 0))
        if self.probe is not None:
            s = self.probe.logits(q)[:, None]
            x = np.concatenate([x, s], axis=1)
            if self.format_weights:   # ψ_f = ψ_math + δ_f: the probe logit times each non-math format indicator
                x = np.concatenate([x, s * np.array([q.task == f for f in FORMATS[1:]], dtype=float)], axis=1)
        if self.use_judge:
            x = np.concatenate([x, judge_evidence(q)[0][:, None]], axis=1)
        return x

    def _difficulty(self, q) -> np.ndarray:
        """K-invariant question-level difficulty signals for the null logit (+ judge "none" log-prob)."""
        lengths = np.array([r["length"] for r in q.replay], dtype=float)
        has = [s["pred"] is not None for s in q.samples]
        n_clusters = len({c for c, h in zip(q.clusters, has, strict=True) if h})
        d = [np.log(lengths.mean()), 1.0 - np.mean([s["closed"] for s in q.samples]), n_clusters / q.k,
             1.0 - np.mean(has)]
        return np.array(d + [judge_evidence(q)[1]] if self.use_judge else d)

    def _prepare(self, train) -> None:
        x = np.concatenate([self._features(q) for q in train])
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-6
        d = np.stack([self._difficulty(q) for q in train])
        self.d_mu, self.d_sd = d.mean(0), d.std(0) + 1e-6

    def batch(self, questions):
        """Padded tensors: features X[Q,K,d], response mask R[Q,K], cluster membership M[Q,C,K],
        correct-cluster indicator Y[Q,C], difficulty D[Q,·]."""
        import torch

        kmax = max(q.k for q in questions)
        cmax = max(1, max(len({c for c, s in zip(q.clusters, q.samples, strict=True) if s["pred"] is not None})
                          for q in questions))
        X = np.zeros((len(questions), kmax, len(self.mu)))
        R = np.zeros((len(questions), kmax))
        M = np.zeros((len(questions), cmax, kmax))
        Y = np.zeros((len(questions), cmax))
        D = np.stack([(self._difficulty(q) - self.d_mu) / self.d_sd for q in questions])
        for a, q in enumerate(questions):
            X[a, : q.k] = (self._features(q) - self.mu) / self.sd
            R[a, : q.k] = 1.0
            ids = sorted({c for c, s in zip(q.clusters, q.samples, strict=True) if s["pred"] is not None})
            for j, cid in enumerate(ids):
                for i, (c, s) in enumerate(zip(q.clusters, q.samples, strict=True)):
                    if c == cid and s["pred"] is not None:
                        M[a, j, i] = 1.0
                        Y[a, j] = max(Y[a, j], q.labels[i])
        t = lambda z: torch.tensor(z, dtype=torch.float64)  # noqa: E731
        return t(X), t(R), t(M), t(Y), t(D)

    def log_posterior(self, params, X, R, M, D=None):
        """log P(cluster j correct) [Q,C] and log P(no cluster correct) [Q] (or None)."""
        import torch

        g = (X @ params["w"] + params["b"]) * R                      # [Q,K]
        n = M.sum(-1)                                                 # [Q,C]
        valid = n > 0
        u = (M @ g[..., None]).squeeze(-1) / n.clamp(min=1)
        if self.learn_r:
            u = u * n / (1 + (n - 1) * torch.sigmoid(params["rho"]))
        if self.use_votes:
            k = R.sum(-1, keepdim=True)
            alpha = params["alpha"] if self.alpha_fixed is None else self.alpha_fixed
            u = u + alpha * torch.log((n / k).clamp(min=1e-12))
        u = torch.where(valid, u, torch.full_like(u, -torch.inf))
        if not self.use_null:
            return torch.log_softmax(u, -1), None
        share = n / n.sum(-1, keepdim=True).clamp(min=1)
        ent = -(share * torch.log(share.clamp(min=1e-12))).sum(-1)
        g_mean = g.sum(-1) / R.sum(-1)
        u_null = params["theta"][0] + params["theta"][1] * g_mean + params["theta"][2] * ent
        if not self.null_features:
            u_null = params["theta"][0] + 0.0 * g_mean
        if self.use_difficulty and D is not None:
            u_null = u_null + D[:, :4] @ params["phi"][:4]
        if self.use_judge and D is not None:
            u_null = u_null + D[:, -1] * params["phi"][-1]
        logp = torch.log_softmax(torch.cat([u, u_null[:, None]], -1), -1)
        return logp[:, :-1], logp[:, -1]

    def nll(self, params, batch):
        import torch

        X, R, M, Y, D = batch
        logp_c, logp_null = self.log_posterior(params, X, R, M, D)
        if self.objective == "response":
            p = torch.nan_to_num(logp_c.exp(), nan=0.0).clamp(1e-9, 1 - 1e-9)       # [Q,C]
            bce = -(Y * torch.log(p) + (1 - Y) * torch.log1p(-p)) * M.sum(-1)     # summed over members
            return bce.sum() / M.sum()
        any_correct = Y.sum(-1) > 0
        masked = torch.where(Y > 0, logp_c, torch.full_like(logp_c, -torch.inf))
        ll_correct = torch.logsumexp(masked, -1)
        if logp_null is None:
            return -ll_correct[any_correct].mean()
        return -torch.where(any_correct, ll_correct, logp_null).mean()

    def _fit_once(self, batch, l2):
        import torch

        torch.manual_seed(SEED)
        dim = batch[0].shape[-1]
        params = {"w": torch.zeros(dim, dtype=torch.float64, requires_grad=True),
                  "b": torch.zeros((), dtype=torch.float64, requires_grad=True),
                  "alpha": torch.ones((), dtype=torch.float64, requires_grad=True),
                  "rho": torch.zeros((), dtype=torch.float64, requires_grad=True),
                  "theta": torch.zeros(3, dtype=torch.float64, requires_grad=True),
                  "phi": torch.zeros(batch[4].shape[-1], dtype=torch.float64, requires_grad=True)}
        opt = torch.optim.Adam(params.values(), lr=LR)
        for _ in range(EPOCHS):
            opt.zero_grad()
            loss = self.nll(params, batch) + l2 * ((params["w"] ** 2).sum() + (params["phi"] ** 2).sum())
            loss.backward()
            opt.step()
        return {k: v.detach() for k, v in params.items()}

    def fit(self, train, val) -> AnswerModel:
        import torch

        self._prepare(train)
        tr, va = self.batch(train), self.batch(val)
        best, self.val_nll = None, {}
        for l2 in L2_GRID:
            params = self._fit_once(tr, l2)
            with torch.no_grad():
                nll = float(self.nll(params, va))
            self.val_nll[l2] = nll
            if best is None or nll < best[0]:
                best = (nll, l2, params)
        _, self.l2, self.params = best
        return self

    def log_outcomes(self, q):
        """(log P over [clusters..., ∅], membership M[C,K]); ∅ is absent without the outside option."""
        import torch

        X, R, M, _, D = self.batch([q])
        with torch.no_grad():
            logp_c, logp_null = self.log_posterior(self.params, X, R, M, D)
        out = logp_c[0] if logp_null is None else torch.cat([logp_c[0], logp_null])
        return out, M[0]

    def describe(self) -> dict:
        import torch

        probe = (["hidden_probe"] + ([f"hidden_probe x {f}" for f in FORMATS[1:]] if self.format_weights else [])
                 if self.probe is not None else [])
        names = [FEATURES[i] for i in self.columns] + probe + (["judge"] if self.use_judge else [])
        raw = (self.params["w"] / torch.tensor(self.sd, dtype=torch.float64)).tolist()
        alpha = float(self.params["alpha"]) if self.alpha_fixed is None else self.alpha_fixed
        r = float(torch.sigmoid(self.params["rho"])) if self.learn_r else 1.0
        return {"l2": self.l2, "alpha": alpha, "r": r, "theta": self.params["theta"].tolist(),
                "weights_per_std": dict(zip(names, self.params["w"].tolist(), strict=True)),
                "weights_raw_units": dict(zip(names, raw, strict=True)),
                "use_null": self.use_null, "use_votes": self.use_votes, "null_features": self.null_features,
                "use_difficulty": self.use_difficulty, "phi": self.params["phi"].tolist(),
                "val_nll": self.val_nll}


class CAPE(Estimator):
    paper = "ours"
    probabilistic = True

    def __init__(self, evidence: str = "probe", gate: str = "format+residual", support_alpha: float = SUPPORT_ALPHA,
                 name: str = "CAPE", main: bool = True, probe_tasks: tuple[str, ...] | None = None, **answer_model):
        self.evidence = evidence  # "probe" (default): probe model | "gated": shift-gated pool | "scalar": no probe
        self.gate = gate          # "format+residual" (default) | "format" | "residual" (ablations)
        self.support_alpha = support_alpha
        self.probe_tasks = probe_tasks  # fit the probe model on these answer formats only (None: all in the pool)
        self.answer_model = answer_model
        self.name, self.main = name, main

    def fit(self, train, val) -> None:
        self.formats = {q.task for q in train}
        self.robust = AnswerModel(**self.answer_model).fit(train, val)
        if self.evidence != "scalar":
            if self.probe_tasks is not None:
                self.formats &= set(self.probe_tasks)
                train = [q for q in train if q.task in self.formats]
                val = [q for q in val if q.task in self.formats]
            self.probe = HiddenProbe().fit(train, val)
            self.full = AnswerModel(probe=self.probe, **self.answer_model).fit(train, val)

    def support(self, q) -> float:
        """Weight of the probe model for this question (1 = fully in the pool's support)."""
        if self.evidence == "scalar":
            return 0.0
        if self.evidence == "probe":
            return 1.0
        w = 1.0
        if "format" in self.gate and q.task not in self.formats:
            return 0.0
        if "residual" in self.gate:
            w = min(1.0, self.probe.p_value(q) / self.support_alpha)
        return w

    def log_outcomes(self, q):
        import torch

        log_r, M = self.robust.log_outcomes(q)
        w = self.support(q)
        if w <= 0.0:
            return log_r, M
        log_f, _ = self.full.log_outcomes(q)
        if w >= 1.0:
            return log_f, M
        finite = torch.isfinite(log_r)
        mixed = torch.where(finite, (1 - w) * log_r + w * log_f, log_r)
        return torch.log_softmax(mixed, -1), M

    def score(self, q):
        import torch

        logp, M = self.log_outcomes(q)
        n_clusters = M.shape[0]
        p = torch.nan_to_num(logp[:n_clusters].exp(), nan=0.0)
        return (M * p[:, None]).sum(0)[: q.k].numpy()

    def describe(self):
        out = self.robust.describe()
        out.update(evidence=self.evidence, gate=self.gate, support_alpha=self.support_alpha)
        if self.evidence != "scalar":
            out.update(probe=self.probe.describe(), probe_model=self.full.describe())
        return out
