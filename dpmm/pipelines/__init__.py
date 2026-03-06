# PrivBayes-only: only export PrivBayesPipeline (no AIM/MST)
from dpmm.pipelines.priv_bayes import PrivBayesPipeline

PIPELINES = [PrivBayesPipeline]
PIPELINE_DICT = {PIPE.model.name: PIPE for PIPE in PIPELINES}
