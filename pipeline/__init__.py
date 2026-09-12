from .causal_inference import CausalInferencePipeline

# 上游 inference.py 还引用了多步扩散管线的类名; 本地 2-step config 只走
# CausalInferencePipeline 分支, 用别名兜底让官方脚本可直接运行
CausalDiffusionInferencePipeline = CausalInferencePipeline

__all__ = [
    "CausalInferencePipeline",
    "CausalDiffusionInferencePipeline",
]