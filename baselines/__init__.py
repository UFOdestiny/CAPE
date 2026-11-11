"""Baseline registry: one module per method or method family; notes on each paper in docs/baselines/."""
from baselines.cdg import CDG
from baselines.cisc import CISC
from baselines.consistency import ConsistCal, DeepConfVote
from baselines.deepconf import DeepConf
from baselines.fusion import CoCoA, DeepConfSC, Stacker
from baselines.lrmverbal import LRMVerbal
from baselines.orm import ORM
from baselines.reasonprobe import ReasonProbe
from baselines.reference import SC, SE
from baselines.selfcertainty import SelfCertainty
from baselines.semantic_energy import SemEnergy
from baselines.sep import SEP
from baselines.snne import SNNE


def build_all() -> list:
    """Main rows first, then variant / ablation rows (ReasonProbe layer ablation is added by the
    evaluator once the main probe's configuration is selected)."""
    return [
        DeepConf(), CDG(), SelfCertainty(), LRMVerbal(), CISC(), ReasonProbe(), SNNE(), DeepConfVote(), ConsistCal(),
        SEP(), ORM(), SemEnergy(), DeepConfSC(), CoCoA(), Stacker(), SC(), SE(),
        *[DeepConf(v) for v in ("mean", "bottom10", "lowest", "tail")],
        CDG("gain"), CDG("vote"), CISC(single=True), SNNE("math_eq"),
    ]


__all__ = ["CDG", "CISC", "CoCoA", "ConsistCal", "DeepConf", "DeepConfSC", "DeepConfVote", "LRMVerbal",
           "ORM", "ReasonProbe", "SC", "SE", "SEP", "SelfCertainty", "SemEnergy", "SNNE", "Stacker", "build_all"]
