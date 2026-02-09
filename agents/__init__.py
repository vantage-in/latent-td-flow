# Baselines (Model-free)
from agents.gciql import GCIQLAgent
from agents.gcivl import GCIVLAgent
from agents.latent_td_flow import LatentTDFlowAgent
from agents.latent_td_flow_gciql import LatentTDFlowGCIQLAgent
# from agents.ngcsacbc import NGCSACBCAgent
# from agents.sharsa import SHARSAAgent

agents = dict(
    gciql=GCIQLAgent,
    gcivl=GCIVLAgent,
    latent_td_flow=LatentTDFlowAgent,
    latent_td_flow_gciql=LatentTDFlowGCIQLAgent,
    # ngcsacbc=NGCSACBCAgent,
    # sharsa=SHARSAAgent,
)
