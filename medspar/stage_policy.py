"""One fixed Stage2 trainable policy; no phase switching or checkpoint chain."""
from dataclasses import dataclass
from .experiments import ExperimentConfig, UNIFIED_VERSION

GROUNDING_MODULE_NAMES = (
    "reasoning_query_pooler", "seg_anchor_projector", "semantic_spatial_grounder",
    "automatic_geometric_prompt_generator", "semantic_prompt_projector", "prompt_fusion_gate",
)

@dataclass(frozen=True)
class TrainingStageConfig:
    stage: str = "unified"
    description: str = "Joint language, grounding and automatic segmentation"
    lambda_lm: float = 1.0
    lambda_mask: float = 1.0
    lambda_attn: float = 1.0
    lambda_box: float = 0.5
    lambda_prompt: float = 0.5
    best_metric_name: str = "val_auto_dice"

def get_stage_config(stage="unified"):
    if stage != "unified":
        raise ValueError("Stage2 now supports only stage=unified; historical phase checkpoints are not initialization sources.")
    return TrainingStageConfig()

def is_grounding_stage(stage):
    get_stage_config(stage)
    return True

def steer_trainable_modules(model):
    """Only Stage1 modules that are used by the vision forward (no auxiliary patch head)."""
    modules = []
    for module in model.medsam.image_encoder.modules():
        if module.__class__.__name__ in {"GatedTextCrossAttention", "GatedCrossAttention"}:
            modules.append(module)
    steer = getattr(getattr(model, "vision", None), "steer_model", None)
    connector = getattr(getattr(steer, "text_encoder", None), "connector", None)
    if connector is not None:
        modules.append(connector)
    return modules


def apply_stage_freeze_policy(model, stage="unified", unfreeze_steer_modules=False,
                              unfreeze_last_vit_blocks=0):
    get_stage_config(stage)
    policy = ExperimentConfig(model.experiment_id)
    model.requires_grad_(False)
    for name in GROUNDING_MODULE_NAMES:
        getattr(model, name).requires_grad_(True)
    model.visual_compressor.requires_grad_(True)
    model.visual_projector.requires_grad_(True)
    model.medsam.mask_decoder.requires_grad_(True)
    model.medsam.mask_decoder.iou_prediction_head.requires_grad_(False)
    for name, parameter in model.llm.named_parameters():
        if "lora_" in name:
            parameter.requires_grad_(True)
    for module in (model.llm.get_input_embeddings(), model.llm.get_output_embeddings()):
        module.requires_grad_(True)
    if not policy.reasoning_queries:
        model.reasoning_query_pooler.requires_grad_(False)
    if not policy.semantic_prompt:
        model.semantic_prompt_projector.requires_grad_(False)
        model.prompt_fusion_gate.requires_grad_(False)
    # box_only retains the Full trainable policy. The unused coarse head has
    # grad=None and receives no optimizer update (including weight decay).
    if unfreeze_last_vit_blocks not in {0, 2, 3, 4}:
        raise ValueError("Choose 0, 2, 3 or 4 last ViT blocks")
    if unfreeze_steer_modules:
        for module in steer_trainable_modules(model):
            module.requires_grad_(True)
    if unfreeze_last_vit_blocks:
        blocks = model.medsam.image_encoder.blocks
        if unfreeze_last_vit_blocks >= len(blocks):
            raise ValueError("Full ViT unfreezing is disabled")
        for block in blocks[-unfreeze_last_vit_blocks:]:
            block.requires_grad_(True)
        if not unfreeze_steer_modules:
            for module in steer_trainable_modules(model):
                module.requires_grad_(False)
    model.medsam.prompt_encoder.requires_grad_(False)

def keep_frozen_modules_eval(model):
    """Disable dropout in frozen backbones after the outer model.train() call."""
    for module in model.modules():
        parameters = tuple(module.parameters())
        if parameters and not any(parameter.requires_grad for parameter in parameters):
            module.eval()
