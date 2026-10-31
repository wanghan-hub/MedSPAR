"""Explicit orthogonal controls for the eight MedSPAR experiments."""
from dataclasses import dataclass, asdict
UNIFIED_VERSION = 'medspar_experiments_v1'
EXPERIMENTS = ('main', 'no_rq', 'no_semantic', 'box_only', 'no_spatial_heatmap_loss',
              'medsam_vision', 'coarse_mask_only', 'semantic_only')
@dataclass(frozen=True)
class ExperimentConfig:
    name: str = 'main'
    def __post_init__(self):
        if self.name not in EXPERIMENTS: raise ValueError(f'Unknown experiment: {self.name}')
    @property
    def vision_backbone(self): return 'medsam' if self.name == 'medsam_vision' else 'TC_MVE'
    @property
    def reasoning_queries(self): return self.name != 'no_rq'
    @property
    def spatial_supervision(self): return self.name != 'no_spatial_heatmap_loss'
    @property
    def point_prompts(self): return False
    @property
    def box_prompt(self): return self.name not in {'coarse_mask_only', 'semantic_only'}
    @property
    def coarse_mask_prompt(self): return self.name not in {'box_only', 'semantic_only'}
    @property
    def semantic_prompt(self): return self.name != 'no_semantic'
    @property
    def multi_geometry(self): return self.point_prompts and self.coarse_mask_prompt
    def flags(self):
        return {k: getattr(self,k) for k in ('vision_backbone','reasoning_queries','spatial_supervision',
                 'point_prompts','box_prompt','coarse_mask_prompt','semantic_prompt')}
MODEL_CONFIG_FIELDS = (
    'experiment_id','vision_type','stage','model_type','vision_dim','visual_grid',
    'steer_cross_attn_layers','steer_text_max_length','use_lora','lora_r','lora_alpha',
    'lora_dropout','lora_target_modules','seg_token','seg_loss_weight','num_reasoning_queries',
    'num_pos_points','num_neg_points','grounder_layers','grounder_dim','image_size','max_length','system_prompt')
