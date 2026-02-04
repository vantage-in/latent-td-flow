# Baselines (Model-free)
from agents.gciql import GCIQLAgent
from agents.gcivl import GCIVLAgent
# from agents.ngcsacbc import NGCSACBCAgent
# from agents.sharsa import SHARSAAgent

agents = dict(
    gciql=GCIQLAgent,
    gcivl=GCIVLAgent,
    # ngcsacbc=NGCSACBCAgent,
    # sharsa=SHARSAAgent,
)
