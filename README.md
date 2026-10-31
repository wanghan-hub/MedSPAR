# MedSPAR

独立的新项目：Stage1 训练完整 Steer-MedSAM（其文本条件视觉编码器命名为 TC-MVE / `TC_MVE`）；Stage2 训练 MedSPAR。Stage0 是普通 MedSAM 的独立对照准备分支。所有阶段重新训练，不加载旧 reaseg 项目的训练权重。基础预训练资源仍需要。

## 环境和数据

沿用已验证的服务器环境，依赖见 `requirements.txt`，不盲目升级或安装未经验证的版本。需要支持 Qwen3.5 的 Transformers、PEFT、匹配 CUDA 的 PyTorch/torchvision，及 numpy、Pillow、tqdm、psutil。许可证见 `THIRD_PARTY_NOTICES.md`。

服务器项目根目录 `/data/ReaSeg/MedSPAR`。所有以下命令从项目根目录执行。
数据根 `/data/ReaSeg/data`，保留原来的数据集子目录、train/val/test JSON 及图像/mask相对路径；每条 JSON 使用 `input` 问题及 `output` 参考答案。Stage0/1/2沿用相同数据读取协议，不重新划分数据。
基础资源目录 `/data/ReaSeg/pretrained`：`medsam_vit_b.pth`、`bert-base-uncased/`、`Qwen3.5-2B-SEG/`。
最后一项必须是永久加入 [SEG]、同步保存 tokenizer/embedding 的基础 LLM，不是旧 Stage2 checkpoint。如果尚未准备，使用实际原始 Qwen 目录运行：

```bash
python tools/add_seg_token_to_llm.py --model_path /data/ReaSeg/pretrained/Qwen3.5-2B --output_dir /data/ReaSeg/pretrained/Qwen3.5-2B-SEG --local_files_only --dtype bf16
```

在 `scripts/experiment_defaults.sh` 手动设置 SEED（默认2026）及 GPU_IDS（默认0,1,2,3,4,5）。环境变量可覆盖：

```bash
export DATA_ROOT=/data/ReaSeg/data PRETRAINED_ROOT=/data/ReaSeg/pretrained
export GPU_IDS=0,1,2,3,4,5 EVAL_GPU_ID=0 SEED=2026
```

默认训练和训练中验证使用六卡，每卡batch=2；独立验证/测试使用单卡、batch=2。`--batch_size`、`--val_batch_size`可显式覆盖，但公平对比应整组一致。每个实验从相同基础 LLM、同一新Stage1（或新Stage0）独立初始化，禁止从 main 继续训练消融。配置与有效开关均记录。

## 结构与训练范围

图像＋用户问题 → TC-MVE → 视觉token压缩/投影＋LLM → 回答隐藏状态、SEG anchor、Reasoning Queries → ACSG空间定位 → SGPA自动框/粗掩码/语义提示 → SAM mask decoder。
ACSG逻辑组为 `reasoning_query_pooler`、`seg_anchor_projector`、`semantic_spatial_grounder`；SGPA为 `automatic_geometric_prompt_generator`、`semantic_prompt_projector`、`prompt_fusion_gate`及提示适配。保留原参数结构，不新增包装层或checkpoint前缀。

Stage0/Stage1共有SAM部分采用相同冻结策略：原Image Encoder、Prompt Encoder及IoU head冻结；mask decoder分割部分训练。Stage1额外训练BERT connector、插入的GCA、Patch Head，BERT backbone冻结。Stage0没有文本引导或Patch Head。两者共同BCE＋Dice；Stage1附加0.2 Patch BCE，均50epoch、同bbox/数据协议。
Stage2默认15epoch，视觉分支冻结，训练LoRA、完整embedding/LM head、视觉压缩/投影、ACSG/SGPA及mask decoder分割部分；消融关闭的RQ或语义模块冻结。所有八组无点提示。`medsam_vision`加载新Stage0的完整SAM，而非混入Stage1 decoder；这是整体视觉方案对照。

## Stage0 与 Stage1

```bash
bash scripts/train_stage1_tc_mve.sh
bash scripts/eval_stage1_tc_mve.sh --split val
bash scripts/eval_stage1_tc_mve.sh
bash scripts/train_stage0_medsam.sh
bash scripts/eval_stage0_medsam.sh --split val
bash scripts/eval_stage0_medsam.sh
```

Stage1权重：`work_dir/stage1_TC-MVE/best.pth`，测试：`output/stage1_TC-MVE_test/`。
Stage0权重：`work_dir/stage0_medsam/best.pth`，测试：`output/stage0_medsam_test/`。
每epoch自动验证，best按原生GT网格平均Dice选择；Stage0/1使用GT派生bbox的提示分割，仅用于各自视觉训练/对照，不能与Stage2自动提示指标混作同一种任务。

## Stage2 八组训练、验证与测试

每epoch自动自由生成验证，best=`val_auto_dice`。训练使用参考output的teacher forcing监督，验证/测试仅图像和input生成，GT只用于最终评分；GT框不会进入自动推理。所有实验分别运行下面三条命令，独立val输出为 `output/<ID>_val/`。

### main

预测框＋粗掩码＋语义，保留RQ与空间监督。权重 `work_dir/stage2_MedSPAR/best.pth`；测试 `output/main_test/`。

```bash
bash scripts/train_main.sh
bash scripts/eval_stage2_medspar.sh --experiment main --split val
bash scripts/eval_stage2_medspar.sh --experiment main
```

### no_rq

仅关闭Reasoning Queries，保留SEG anchor。权重 `work_dir/no_rq/best.pth`；测试 `output/no_rq_test/`。

```bash
bash scripts/train_no_rq.sh
bash scripts/eval_stage2_medspar.sh --experiment no_rq --split val
bash scripts/eval_stage2_medspar.sh --experiment no_rq
```

### no_semantic

仅关闭语义提示及融合门。权重 `work_dir/no_semantic/best.pth`；测试 `output/no_semantic_test/`。

```bash
bash scripts/train_no_semantic.sh
bash scripts/eval_stage2_medspar.sh --experiment no_semantic --split val
bash scripts/eval_stage2_medspar.sh --experiment no_semantic
```

### box_only

仅关闭粗掩码SAM输入，仍保留语义提示。权重 `work_dir/box_only/best.pth`；测试 `output/box_only_test/`。

```bash
bash scripts/train_box_only.sh
bash scripts/eval_stage2_medspar.sh --experiment box_only --split val
bash scripts/eval_stage2_medspar.sh --experiment box_only
```

### no_spatial_heatmap_loss

仅关闭lambda_attn，框和prompt监督保留。权重 `work_dir/no_spatial_heatmap_loss/best.pth`；测试 `output/no_spatial_heatmap_loss_test/`。

```bash
bash scripts/train_no_spatial_heatmap_loss.sh
bash scripts/eval_stage2_medspar.sh --experiment no_spatial_heatmap_loss --split val
bash scripts/eval_stage2_medspar.sh --experiment no_spatial_heatmap_loss
```

### medsam_vision

视觉改为新Stage0微调的完整普通MedSAM。权重 `work_dir/medsam_vision/best.pth`；测试 `output/medsam_vision_test/`。

```bash
bash scripts/train_medsam_vision.sh
bash scripts/eval_stage2_medspar.sh --experiment medsam_vision --split val
bash scripts/eval_stage2_medspar.sh --experiment medsam_vision
```

### coarse_mask_only

仅关闭box SAM输入，语义及全部几何监督保留。权重 `work_dir/coarse_mask_only/best.pth`；测试 `output/coarse_mask_only_test/`。

```bash
bash scripts/train_coarse_mask_only.sh
bash scripts/eval_stage2_medspar.sh --experiment coarse_mask_only --split val
bash scripts/eval_stage2_medspar.sh --experiment coarse_mask_only
```

### semantic_only

关闭box和coarse SAM输入，几何分支/辅助监督保留。权重 `work_dir/semantic_only/best.pth`；测试 `output/semantic_only_test/`。

```bash
bash scripts/train_semantic_only.sh
bash scripts/eval_stage2_medspar.sh --experiment semantic_only --split val
bash scripts/eval_stage2_medspar.sh --experiment semantic_only
```

## 一键运行与续训

```bash
bash run_all.sh train
bash run_all.sh eval
bash scripts/run_all_ablations.sh train
bash scripts/run_all_ablations.sh eval
```

一键顺序：新Stage1训练/测试 → main训练/测试 → 新Stage0训练/测试 → 七组消融训练/测试。已有非空输出目录会报错，不跳过、不覆盖。仅评估已完成权重时，用新 `EVAL_ROOT` 保存结果，或使用单实验 `--output_dir`。
续训必须指定同一新实验的checkpoint，恢复模型/optimizer/scaler/epoch；配置漂移会拒绝，允许延长epoch总数。示例：

```bash
bash scripts/train_main.sh --resume work_dir/stage2_MedSPAR/last.pth --epochs 15
bash scripts/train_stage1_tc_mve.sh --resume work_dir/stage1_TC-MVE/last.pth --epochs 50
bash scripts/eval_stage2_medspar.sh --experiment main --checkpoint work_dir/stage2_MedSPAR/best.pth --output_dir output/main_retest
```

不要把 `--resume` 传给一键所有实验脚本。续训按epoch边界恢复，不承诺与未中断训练逐bit相同的随机数流。

## 输出与指标

训练输出：`config.json`、`train_config.json`、`training_budget.json`、`training_details.json`、`trainable_parameters.json`、`history.json`、`efficiency.json`、`compute_profile.json`、best/last及周期checkpoint。`validation/epoch_XXXX/`包含summary、per_dataset_summary和area_group_summary。
测试输出：`config.json`/`eval_config.json`、`summary.json`、`overall_summary.csv`、`per_dataset_summary.json`、`per_sample_metrics.csv`、`efficiency.json`及效率兼容输出、`compute_profile.json`、`area_group_summary.csv`；Stage2额外SEG诊断、可选visualizations等专属内容保留。
Dice/IoU/Recall在原生GT尺寸计算；缺失或多个SEG计失败，三项分数为0且仍计入总体分母，success rate单报。有效预测遇empty GT时Recall为1；Dice/IoU沿用epsilon公式。near_zero≤1e-6，low_dice<0.1；面积分组按原生GT前景占比：empty、<0.1%、0.1–0.5%、0.5–1%、≥1%。不删低分/小目标样本。
参数统计按唯一参数计数；各模块统计可能共享重叠。效率分别记录训练、验证、保存、推理时间和显存；计算量只报告profiler支持算子的部分FLOPs，不伪称完整FLOPs。

## 服务器验证顺序

先预览真实命令（不加载模型）：

```bash
python tools/run_experiments.py train --stage stage1 --dry_run
python -c "import torch, transformers, peft, torchvision; import train.train_stage1, train.train_stage2, evaluation.eval_stage2_medspar; print(torch.cuda.device_count())"
python -m compileall -q TC_MVE medspar train evaluation tools
for s in run_all.sh scripts/*.sh; do bash -n "$s"; done
```

基础资源准备好后先真实单batch forward/backward/optimizer；使用独立smoke输出，不保存假checkpoint：

```bash
bash scripts/train_stage1_tc_mve.sh --smoke_batches 1 --output_dir work_dir/smoke_stage1
bash scripts/train_stage0_medsam.sh --smoke_batches 1 --output_dir work_dir/smoke_stage0
```

正式完成Stage1后验证新视觉权重严格加载，再检查Stage2单batch：

```bash
bash scripts/train_main.sh --smoke_batches 1 --output_dir work_dir/smoke_main
```

首次Stage2 checkpoint生成后，用少量真实样本验证保存→加载→生成→GT评分：

```bash
bash scripts/eval_stage2_medspar.sh --experiment main --checkpoint work_dir/stage2_MedSPAR/best.pth --max_samples 2 --output_dir output/smoke_main_test
```

加载应显示missing/unexpected均0；不接受旧项目checkpoint。通过后继续正式训练/完整测试。真实大模型梯度、CUDA内存、DDP行为及最终精度须在服务器验证，本地结构测试不能替代。
