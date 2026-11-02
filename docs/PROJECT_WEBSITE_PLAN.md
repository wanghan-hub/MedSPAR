# MedSPAR Project Website — implementation and evidence plan

Snapshot: **2 November 2026 (Asia/Shanghai)**. This is a maintenance specification for the implemented website, an evidence ledger, and a replacement register. It is not a claim that the paper, experiments or release are complete.

## Scope and repository identity

- Task workspace / actual Git root: `C:/Users/wanghan/Documents/ChatGPT/Reaseg`. Branch `main`; no configured remote and no tracked files were returned at intake. Many existing untracked files are user work, not disposable generated files. No root README or root `docs/` website existed at intake. No applicable `AGENTS.md` was found in the workspace or checked ancestors.
- Current method source: **`E:/Project/MedSPAR`**, identified in the current paper plan and independently inspected. It is a separate local source tree, not the current Git root. It has `TC_MVE/`, disk package `MedSPAR/`, `train/`, `evaluation/`, `configs/`, processing/generation scripts, and no local `work_dir/` or `output/` directory. This does not establish whether server experiments are complete.
- Historical sources: workspace `reasegllm/`, `UNIFIED_WORK/`, patch directories and `E:/Reasegllm/` remain history. They do not replace the current `main` implementation.
- Actual data inspected read-only: `E:/Project/ReaSeg/data/<group>/<split>/<split>.json`. No source record, patient ID, medical image, mask, weight or private credential was copied into `docs/`.
- The user-supplied planning file was fully read as task requirements because the current request explicitly adopts its rules. Historical claims inside it were treated as leads requiring verification. Prompt files elsewhere in the workspace are historical material, not new instructions for this task.
- Only root documentation and new files under `docs/` are written. Research source, original figures, datasets and experiment files remain unchanged. Existing important content did not require a backup because no root README was replaced. The external source README is retained untouched.
- Reference organization was inspected at [MedReasoner repository](https://github.com/PRIS-CV/MedReasoner) and [project page](https://pris-cv.github.io/MedReasoner/). The new site uses original prose, diagrams, HTML, CSS and JS. No MedReasoner assets, source code or brand marks were reused.

## Positioning and terminology

Use **MedSPAR — Spatial Prompt Adaptation for Medical Reasoning Segmentation** as the confirmed project/method name, supported by `01_方案与目录确认提示词.md`, `PAPER_PLAN.md` and current source. This expansion is not a verified final publication title. Task: a 2D image and question produce a generated answer and the corresponding binary mask when generation has exactly one valid `[SEG]`.

Use TC-MVE = Text-Conditioned Medical Visual Encoder; ACSG = Answer-Conditioned Spatial Grounding; SGPA = Semantic–Geometric Prompt Adaptation. These are implemented interfaces/logical groups, not three independently proven innovations. Do not describe generated prose as independently validated clinical reasoning. Do not claim diagnosis, clinical testing, 3D segmentation, SOTA, significance or universal module benefit.

## Fact verification ledger

Source paths in this ledger are evidence references, never runtime dependencies of the website. Paths starting `SOURCE/` mean `E:/Project/MedSPAR/`; `DATA/` means `E:/Project/ReaSeg/data/`; other relative paths refer to this workspace. Source hashes captured before writing are in `qa/input_fingerprints.json`; aggregate manifest observations are in `qa/manifest_audit.json`. Machine observations do not certify scientific conclusions.

| Check | Evidence inspected | Finding / status | Public treatment |
|---|---|---|---|
| 1. Name / task | `PAPER_PLAN.md` §1.1–1.2; current `paper/drafts/03_method.tex`; `SOURCE/MedSPAR/model.py`, `generation.py` | Name and 2D question-conditioned task supported; final publication title not established | Use project expansion; leave publication identity pending |
| 2. Inputs / outputs / supervision | `SOURCE/experiment_protocol.py:question_text`, `answer_text`; `MedSPAR/dataset.py`, `collator.py`, `generation.py` | Image/question are model conditions; answer/mask supervise training; metadata not model input; valid SEG required | Explicit on README, teaser and Data section |
| 3. MedSAM / Stage 1 / LLM relationship | `SOURCE/TC_MVE/model.py`, `encoder.py`; `MedSPAR/vision_backbones.py`, `projectors.py` | Stage 1 trains Steer-MedSAM including decoder; TC-MVE is a parameter-free calling interface over owned BERT/image modules; LLM receives a projected prefix, not a separate LLM visual encoder | Preserve three visual routes; do not call TC-MVE a new standalone pretrained model |
| 4. Stage implementation / verification | `SOURCE/train/train_stage0_medsam.py`, `train_stage1.py`, `train_stage2.py`; three stage JSON configs; `MEDSPAR_DELIVERY_20261027/VALIDATION.md` | Entrypoints implemented. Earlier local structure tests are documented, not rerun as full model tests. Current normal launcher fails package resolution. No current local checkpoint/run directories | All stages described as implemented; full-runtime and performance verification pending |
| 5. Insertion / freeze / information flow | `SOURCE/TC_MVE/steered_image_encoder.py`, `model.py`; `MedSPAR/stage_policy.py`, `model.py:extract_assistant_hidden/_forward_grounding`; `spatial_grounding.py`, `medsam_bridge.py` | GCA before blocks [2,5,6,11], 0-based; original visual backbone/PromptEncoder/IoU head frozen; Stage 1 connector/GCA/patch/decoder train. Stage 2 default visual freeze; full embedding/head + LoRA + prefix/grounding/prompts/decoder train | Code-derived diagram and stage table; no claim of LoRA-only training |
| 6. Datasets / status | Stage configs; 30 actual split JSONs; `SOURCE/data_generate/*.py`, `data_process/*.py`; `MEDSPAR_DATASET_CONSTRUCTION_SPEC.md` | Ten configured groups with 40,338 records; source-level inclusion and final experiments not fully traced. Optional sources are not included merely because a script exists | Actual manifest counts, explicitly record units; no claimed completed multi-dataset benchmark |
| 7. Actual schema | Real Skin sample field/type inspection; all group samples; `SOURCE/MedSPAR/dataset.py` | image/mask/input/output/body_part/modality/target/target_position; position is dict with bbox_xyxy/bbox_region/bbox_text. Optional target_phrase is not mandatory. Split-relative paths; list or four accepted wrappers | Placeholder-only schema example; no raw patient question/image copied |
| 8. Commands | `SOURCE/tools/run_experiments.py`; `scripts/*.sh`; parser definitions in training/evaluation; actual normal Stage 2 dry run | Real launch syntax established from source. Normal dry run fails `ModuleNotFoundError: medspar`; commands are conditional recipes, not a fresh-clone guarantee | README names prerequisite; website resources state source-sync/package block |
| 9. Checkpoints | `SOURCE/checkpoint_io.py`; `TC_MVE/train_utils.py`; `MedSPAR/train_utils.py`, `vision_backbones.py`; launcher | Main uses new Stage 1 full Steer-MedSAM; medsam_vision uses new Stage 0 full SAM; Stage 2 strict restore of own experiment/config/state; no old weights accepted | Save/load relationship documented; no downloads invented; no checkpoint load executed |
| 10. Metrics / aggregation | `SOURCE/experiment_protocol.py:native_prediction/native_scores`; `evaluation/eval_stage2_medspar.py`; `experiment_reporting.py` | Native-mask Dice/IoU/Recall, threshold .5, bilinear logit resizing; failed SEG cases zero and retained. Per-record Overall and per-group summaries differ from group-macro averages. No HD95 in official evaluator | Explain implemented protocol; leave benchmark table unfilled |
| 11. Authors / publication / BibTeX | `paper/drafts/01_introduction.tex`, `02_related_work.tex`, `03_method.tex`, `04_1_experimental_setup.*`, `04_2_comparison.*`; `PAPER_OUTLINE.md`; `paper/references.bib` | Draft sections and planning exist. Bibliography concerns upstream work, not a verified MedSPAR publication record. No confirmed author list, affiliation, public identifier, final title or venue status | Explicit pending text; no invented BibTeX or venue badge |
| 12. Release permissions | Workspace root and `SOURCE/THIRD_PARTY_NOTICES.md`, `licenses/segment_anything_LICENSE`, unpinned `requirements.txt`; figure notes | SAM notice exists externally; no project-wide license confirmed; data/model/image permissions not established | No dataset/model download, medical image or project-wide license claim |

### Data support and source inventory

All ten groups below are **configured + manifests present**, with **current-model run provenance pending**. Counts are independently recounted JSON records, not patients, unique images or QA diversity. The table is also stored in `assets/data/project.json`; `index.html` is a checked-in readable copy, not fetched at runtime.

| Group | Processing/generation sources | Source inclusion boundary |
|---|---|---|
| BraTS | BraTS2019/2020/2023; MSD Task01 BrainTumour | Exact source versions and overlapping subjects require final manifest provenance |
| Breast | Breast_Cancer_DCE-MRI_Data; MAMA-MIA | DCE-MRI, tumor target; source/case map pending |
| Kidney | KiTS23 | Organ and tumor targets; do not rename every record “tumor” |
| Liver | LiTS; Task03 Liver; Task08 HepaticVessel | Task08 handling selects tumor target; LiTS/Task03 potential overlap needs case map |
| Lung | Task06 Lung; NLSTSeg; NSCLC-Radiomics; QIN-Lung-Seg | Source/case map and conversion provenance pending |
| Pancreas | Task07 Pancreas; PanTS | Organ and tumor/lesion targets; source map pending |
| Poly | CVC-300; CVC-ClinicDB; CVC-ColonDB; ETIS-LaribPolypDB; HyperKvasir; Kvasir-SEG | Image-level historical processing exists; universal patient-split claim unsupported |
| Prostate | PI-CAI; PROMISE12; Prostate158; optional Task05 support | Current mask/target and split conflicts require correction; optional Task05 not certified included |
| Skin | ISIC2017; ISIC2018; PH2 | Cross-source duplication and source revisions require final map |
| Thyroid | TN3K | Historical original-test preservation differs from uniform 8:1:1 story |

BUSI, FedBCa and BrainMRI have processing scripts but no separately confirmed integration into these ten manifests. PolypDB appears in preprocessing support but not the inspected polyp merge list. ISIC2016 is commented in historical configuration. None is advertised as a completed experiment or future commitment. Other datasets in the supplied specification are historical candidates until tied to a current manifest/run. No source is assumed redistributable; source-specific official entry links and exact license/version records must precede `TODO_DATA_RELEASE` closure.

### Conflicts and decisions

| Conflict | Evidence / consequence | Treatment in this delivery |
|---|---|---|
| Workspace identity versus current source | Current Git root has no remote; `E:/Project/MedSPAR` is elsewhere | Create requested root README/docs here. Do not silently copy/restructure source, invent a GitHub owner or link outside docs at runtime |
| Package name versus imports | Disk `MedSPAR/`; imports `medspar`; actual normal `python tools/run_experiments.py ... --dry_run` fails | Blocking reproduction issue, `TODO_PACKAGE_CASE`. Temporary in-process alias may be used solely to inspect command expansion; it is not a fixed checkout or a successful canonical CLI |
| Stage 2 duration | Current `configs/stage2_medspar.json` = **30 epochs**; source README and migration = **15** | Document 30 as current default only; actual runs must supply saved configs and budgets |
| Evaluation GPU count | Source README says single GPU, current launcher defaults to GPU_IDS unless explicit override | Examples select `--eval_gpu_id 0`; capability of distributed eval is not a measured run setting |
| Validation/test decoding | Training validation 1.0/0 versus independent evaluation 1.1/4 for repetition penalty/ngram | State difference; do not claim matched decoding or alter research code |
| Stage 0/1 versus Stage 2 task | GT-derived bbox segmentation versus image/question automatic prompts | Separate protocols; no joint ranking |
| New versus historical experiments | `full`, `none_test`, `box_coarse_mask` and current `main` differ in identity and initialization | No historical score renamed as current main; no old checkpoint reuse |
| Paper provisional comparison versus traceable runs | `paper/evidence/section_4_2_notes.md` traces summary arithmetic to `E:/Project/ReaSeg/结果.csv`, SHA256 `92661d5c4dd30af778f4af44c95977c65cd10ce994a3ca75c59632443d0859f4` (matched this audit) | Provisional overall 78.64 Dice / 71.36 IoU in draft is **not promoted as run-verified current-model evidence**. Need checkpoint, effective settings, predictions and corrected data provenance |
| Explicit simulation among result files | `outputs/medreasoner_fine_simulation_01a19654/README_SIMULATED.md` labels every output `SIMULATED_NOT_MEASURED` | Exclude all such outputs and numerical diagnostic plots from research evidence. This finding does not establish that every baseline in the independent summary CSV is simulated |
| Claimed patient-level 8:1:1 versus manifests | Prior `prostate_code_trace.md` documents conflicts; this task independently counted 152/20/17 shared image+mask groups across train/val/test and 5 numeric patient prefixes across splits | Reproduce aggregate observation only. Earlier audit decoded 189 masks and found tumor/foreground mismatch. No raw IDs copied; do not certify globally isolated splits or unmodified benchmark validity |
| Dataset scale units | Spec reports 40,338 samples; actual records sum to 40,338 | Publish “JSON records”; no patient/unique-image equivalence |
| QA review execution | Existing documents report 27B generation, 122B-A10B review, three experts and 1% sampling; underlying review logs not available here | Mark these as reported settings in dataset caption; no invented full review coverage, pass rate or inter-rater score |
| Old Stage 0 fine-tuning story | Historical plan discussed image-encoder tuning; current code freezes base image encoder | Describe current policy and do not silently propagate the historical claim |
| Figures versus data | Existing CT-containing architecture/dataset drawings include schematic masks/prompts; some panels still placeholders | Do not label them measured prediction/GT or copy without image permissions. Create text-only vector explanations from code |
| Figure numbering / paper layout | Existing paper plan excludes a paper teaser, dataset Figure 2 work overlaps old planned numbering | Website teaser is independent of paper layout. No paper figures or numbering are changed |
| Upstream versus project license | SAM Apache-2.0 text exists; own project license unresolved | Preserve upstream facts; do not assign Apache/MIT to all work |

## Implemented website sections

Design: white and pale sage surfaces; dark ink, restrained teal accent and rust for pending states. System fonts require no downloads. Content max-width 1120px; responsive transitions at 800px and 540px. Navigation stays visible; diagrams have explanatory alt text and captions. No animation is required to read the page.

### Hero — `#top`

- **Purpose:** identify the task and offer a direct path to method/resources.
- **Core copy:** “MedSPAR”; “Spatial Prompt Adaptation for Medical Reasoning Segmentation”; “From a medical question to a target mask: connecting answer representations with spatial visual features.”
- **Layout:** centered title, two-line expansion, small author-status line, two real in-page resource actions and a compact release-status strip. Pending paper/code/weights are plain text, not fake buttons.
- **Files:** `index.html`; `assets/css/style.css`; `assets/images/favicon.svg`; name evidence in `PAPER_PLAN.md` and current source.
- **New imagery:** original typographic favicon only; no institutional logos.
- **Caption/interaction:** semantic home link and in-page actions; no submission/acceptance badge.
- **Pending:** `TODO_AUTHORS`, `TODO_PAPER`, `TODO_CODE_SYNC`, `TODO_MODELS`, `TODO_PROJECT_URL`, `TODO_TITLE`.

### Teaser / motivation — `#teaser-title`

- **Purpose:** explain actual user inputs/outputs before technical structure.
- **Core copy:** “Ask about a target. Ground it in the image.” Image + question → answer + binary mask.
- **Layout:** full-width input/output vector diagram, explicitly labeled “Task schematic”, followed by a caption explaining valid SEG and the missing real case.
- **Files:** `assets/images/teaser/task-overview.svg`; original authoring source `maintenance/draw_diagrams.py`; implementation evidence `SOURCE/MedSPAR/generation.py`.
- **New imagery:** delivered text-only schematic. Future replacement needs an authorized image and current-model outputs. Do not insert artificial anatomy, mask silhouettes or GT.
- **English caption:** “No user-provided box or point is required by the automatic inference path. This is an input–output schematic, with no medical image or measured prediction.”
- **Interaction:** none needed; figure remains readable without JS.
- **Pending:** `TODO_TEASER_CASE`. The method figure must never replace this task teaser.

### Introduction — `#motivation`

- **Purpose:** motivate answer-to-spatial-prompt adaptation without unsupported superiority claims.
- **Core copy:** “How does an answer become a spatial prompt?” Explain description/extent distinction and the controlled evaluation question.
- **Layout:** two-column heading and short prose; single column on narrow screens.
- **Files:** `index.html`; evidence `paper/drafts/01_introduction.tex`, `03_method.tex`, current code. No new picture required.
- **Caption/interaction:** scope note states single-image question-conditioned segmentation; no special interaction.
- **Pending:** sharpen claims only after `TODO_RESULTS`; no “first”, “SOTA”, diagnosis or clinical validation language without evidence.

### Dataset construction — `#dataset`

- **Purpose:** separate offline annotation-guided language generation from inference; report the actual local inventory with its limits.
- **Core copy:** “Linking images, language and pixel supervision.” Ten configured task groups; 40,338 JSON records (31,761/4,240/4,337).
- **Layout:** explanatory pipeline; compact count table; visible data-audit note; optional native `<details>` schema expansion. Horizontal table scroll is keyboard focusable if needed.
- **Files:** `assets/images/dataset/construction.svg`, `assets/data/project.json`, `qa/manifest_audit.json`; authoring source `maintenance/draw_diagrams.py`; actual manifests and `SOURCE/MedSPAR/dataset.py`; `MEDSPAR_DATASET_CONSTRUCTION_SPEC.md`.
- **New imagery:** original text-only diagram; no original CT thumbnail, mask or source image copied.
- **English caption:** use the delivered caption identifying Qwen generation/recheck and expert sampling as **reported** settings, with review logs still pending. Its arrow order is conceptual, not an execution-timeline claim.
- **Interaction:** expandable schema; no synthetic medical QA or dataset filtering that could imply source inclusion certainty.
- **Pending:** `TODO_DATA_RELEASE`, `TODO_SPLITS_MASKS`, `TODO_REVIEW_LOGS`. Source-level counts, target classes and unique patients/images need explicit final manifests.

### Methodology — `#method`

- **Purpose:** show real information flow and distinguish training setup from generation-time inference.
- **Core copy:** “Preserve the grid. Adapt the prompts.” Explain TC-MVE, ACSG, SGPA and three visual routes.
- **Layout:** technical vector diagram; three short module descriptions; inference strip; expandable training-stage policies. On phones the diagram can be enlarged and scrolled.
- **Files:** `assets/images/framework/medspar-architecture.png` (unchanged existing diagram); compact original `assets/images/framework/method-overview.svg`; `maintenance/draw_diagrams.py`, `assets/js/main.js`; evidence `SOURCE/TC_MVE/model.py`, `steered_image_encoder.py`, `MedSPAR/model.py`, `spatial_grounding.py`, `medsam_bridge.py`, `stage_policy.py`.
- **New imagery:** existing `figures/MedSPAR_architecture/MedSPAR_architecture.png` was visually inspected: it is a text/vector schematic without patient imagery. It was copied byte-for-byte for display under the user’s authorization to use existing project figures. A compact original code-derived SVG is also supplied for the README. The existing paper figure is not modified. Replacement final figure must preserve X → prefix, X → grounding and E → decoder; geometry via PromptEncoder; semantic bypass; answer/RQ and SEG anchor distinctions; no active point path.
- **English caption:** “Existing architecture draft, copied unchanged and checked against the current source.” The delivered caption adds geometry/semantic routing, disabled points and schematic status. Final paper selection remains pending.
- **Interaction:** progressive-enhancement modal; native dialog keyboard focus, Escape/Close/backdrop dismissal and trigger focus restoration. Button hidden if dialog unsupported. Core image and prose always visible.
- **Pending:** `TODO_FINAL_FIGURE`; `TODO_PACKAGE_CASE` and `TODO_ENVIRONMENT` constrain runtime claims, not the presence of source definitions.

### Experimental evidence — `#results`

- **Purpose:** state implemented scoring and reserve a real benchmark location.
- **Core copy:** “An explicit protocol. A traceable result.” Dice/IoU/Recall on native GT; all records in denominator; separate GT-box protocols.
- **Layout:** three protocol columns, a clearly bordered “Quantitative results pending verification” block, and expandable unresolved settings. No fake zero/percentage table or rank charts.
- **Files:** `index.html` pending block; future table data in `assets/data/project.json` only after confirmation; source evaluator and reporting files; provisional `paper/evidence/section_4_2_notes.md` is an audit lead, not a final site data source.
- **New imagery:** none until verified results; a future chart must include test split, unit, denominator, seed aggregation, checkpoint/run identities and scale.
- **Caption/interaction:** no ranking/filter interaction before real comparable records exist.
- **Pending:** `TODO_RESULTS`, `TODO_DECODING_PROTOCOL`, `TODO_HD95` (optional separate research scope), corrected data and run provenance. Never reuse simulation exports.

### Qualitative cases — `#qualitative`

- **Purpose:** align text, GT and prediction for the same record, including failure behavior.
- **Core copy:** “One question. One aligned case.”
- **Layout:** visibly empty labeled positions for Source image / Ground truth / Prediction / Overlay; explanation explicitly states no experimental images displayed.
- **Files:** future assets under `assets/images/qualitative/` after approval. Candidates in `outputs/comparative_delivery_20261101/visual_recommended_30.csv` are selection suggestions, not current-main predictions. Existing diagnostic screenshots must not be repurposed as polished results.
- **New imagery:** required future same-record model outputs. Keep originals outside docs; export public copies with exact provenance and permissions. Store anonymized case IDs only; remove DICOM/header/burned-in identifiers. Do not alter boundaries or metrics for presentation.
- **English caption after replacement:** “Image and question from [approved source/version, anonymized case]. Ground truth from source annotation. Prediction from [checkpoint/run]. [Display colors and native resolution].” Brackets here are maintenance instructions, never final unverified claims.
- **Interaction:** consider a keyboard-operable case selector/zoom only when multiple real cases exist. No current mock overlay slider.
- **Pending:** `TODO_QUALITATIVE`; `TODO_TEASER_CASE` may share an approved case but retains a different explanatory purpose.

### Resources — `#resources`

- **Purpose:** distinguish available upstream resources from unreleased project artifacts.
- **Core copy:** “Follow the evidence.” Status rows for paper, code, data/weights; real links to the maintenance plan, upstream MedSAM and SAM.
- **Layout:** restrained editorial rows; pending items are noninteractive text. README recipes remain developer-facing.
- **Files:** `index.html`; `assets/data/project.json` release register; source root/requirements/launcher for recipes.
- **New imagery:** none.
- **Caption/interaction:** expandable preview command, optional clipboard button with readable failure status. No JS fetch or external runtime dependency.
- **Pending:** source sync, package naming, pinned environment, exact model IDs/revisions, paper/data/model URLs and standalone inference demo.

### Citation / acknowledgements — `#citation`

- **Purpose:** provide accurate attribution and eventually a verified citation.
- **Core copy:** “Citation forthcoming.” Publication metadata pending; SAM/MedSAM/BERT/LoRA attribution; own project license unresolved.
- **Layout:** simple text with no fabricated BibTeX block.
- **Files:** `index.html`; root README; future bibliography only after approval. Upstream license evidence in `SOURCE/licenses/segment_anything_LICENSE` and `THIRD_PARTY_NOTICES.md`.
- **New imagery:** none.
- **Interaction:** add citation-copy only after an actual verified entry exists. Current JS copies only preview commands.
- **Pending:** `TODO_CITATION`, `TODO_AUTHORS`, `TODO_PAPER`, `TODO_LICENSE`.

## Existing asset inventory and selection

| Original path | Meaning / version | Decision and permission boundary | Website destination |
|---|---|---|---|
| `figures/MedSPAR_architecture/MedSPAR_architecture.png` (+ PPTX/PDF) | Existing architecture draft; visually inspected pure text/vector schematic, no patient imagery | Copied unchanged under the user’s instruction to use project figures; code paths checked; final paper-version preference pending | `assets/images/framework/medspar-architecture.png` |
| `figure_redesign/notation_simplified/MedSPAR_main_figure_minimal.svg` (+ PNG/PDF) | 2026-10-31 notation revision; source notes state embedded real CT and schematic outputs | Not a measured segmentation visualization. Permission for embedded CT and final preferred version pending | `TODO_FINAL_FIGURE` |
| `figure_redesign/reference_based_v2/` and `figures/medspar_overview_abcd_v2/`, `medspar_overview_reference/` | Earlier architecture alternatives | Earlier versions; do not mix labels or pretend tokens/heatmaps are measured | Not copied |
| `figures/figure2_dataset/v2_annotation_guided/Figure2_dataset_annotation_guided_v2.svg` (+ PNG) | Offline QA pipeline; actual CT thumbnail, drawn annotation aids, lower panels pending | Source notes explicitly identify schematic boxes/masks. CT permission not confirmed; some content incomplete | New text-only construction SVG |
| `figures/figure2_dataset/` earlier exports | Older dataset diagram and QA reports | Preserve originals; final preferred revision not assumed | Not copied |
| `teaser_figure_analysis.md` | Teaser decision report, not a teaser image | Reports missing current-main same-record assets | New task-only schematic + `TODO_TEASER_CASE` |
| `outputs/data_quality_audit_01a19654/*.png` | Statistical audit and annotation-conflict diagnostic plots | Not model-performance figures; contain synthetic-score diagnostics or source imagery | Not copied |
| `outputs/comparative_delivery_20261101/visual_*csv` | Candidate case selections | Suggestions only; do not establish prediction identity or display rights | `TODO_QUALITATIVE` |
| Historical training/evaluation CSV/JSON in reaseg trees | Older model runs | Version mismatch; current-main provenance unavailable | No benchmark copy |
| `outputs/medreasoner_fine_simulation_01a19654/*` | Explicit conditional simulation | Not measured evidence | Excluded |

Published assets are **six original SVG files** (desktop/mobile task, compact method, desktop/mobile construction, favicon), created specifically for this delivery with text and vector geometry only, plus **one unchanged existing architecture PNG** (pure schematic, no patient image). The architecture-copy SHA256 is recorded in `qa/asset_provenance.json`. No actual GT, prediction or medical image is displayed. Source `maintenance/draw_diagrams.py` is optional authoring convenience, never a website build requirement.

## Placeholder replacement contract

`assets/data/project.json` is the machine-readable replacement register. It is not fetched by the website. Public placeholders use `data-placeholder="TODO_…"`; missing URLs are `null`, never guessed links. README uses the same IDs. All pending states are honest text, not hidden omissions. Future maintainers must update both static HTML and README, the JSON register and this ledger when evidence arrives. A future chat is not presumed to have completed the research.

| ID | Required replacement evidence | Locations / action |
|---|---|---|
| `TODO_TITLE` | Confirm final publication title separate from project expansion | Hero/metadata/README/Citation together |
| `TODO_AUTHORS` | Author order, affiliations, consent and author URLs | Hero byline; README header; citation |
| `TODO_PAPER` | Reachable paper identifier and verified status/version | Hero/resources/README; add live link only then |
| `TODO_PROJECT_URL` | Confirm Git remote owner/name and verified Pages deployment | README/project resource URL/metadata; no deployed badge now |
| `TODO_CODE_SYNC` | Approved release source integrated in target repo, with version identity | Replace source-location caveat and validate commands from fresh clone; no automatic source migration |
| `TODO_PACKAGE_CASE` | Release directory/import naming consistency and normal CLI pass | Root README and source note; separate authorized research-code fix |
| `TODO_ENVIRONMENT` | Successful pinned Python, torch/torchvision/CUDA, Transformers/PEFT and model-revision record | README installation and `project.json` |
| `TODO_DATA_RELEASE` | Per-source/version permissions, authoritative source/case/target manifest and reachable official/release URLs | Dataset section, source table, resources, README |
| `TODO_SPLITS_MASKS` | Correct target masks; cross-split identity audit; corrected manifests/hashes; affected reevaluations | Visible data caveat; counts/register/README must reflect corrected inventory |
| `TODO_REVIEW_LOGS` | Generation/review model revisions, coverage, sampling denominator and review records | Dataset caption; keep reported/verified distinction until then |
| `TODO_TEASER_CASE` | Authorized same-record image/question/GT/current prediction and exact run identity | Replace task teaser with true case or append case; preserve task caption |
| `TODO_FINAL_FIGURE` | Agreed final diagram, verified labels/arrows/caption and embedded-image rights | Method SVG/caption; preserve three visual routes |
| `TODO_RESULTS` | New-main checkpoint/run/config/decoding/data identity; per-record scores and prediction evidence; corrected data; comparable baselines | Results block, README and JSON. Recompute all tables from raw evidence |
| `TODO_DECODING_PROTOCOL` | Explicitly matched selected decoding settings or transparent separate protocols | Evaluation prose/table notes; cannot silently fix code in docs task |
| `TODO_HD95` | Optional separately implemented/validated spacing and failure conventions | Leave absent unless completed; do not block otherwise complete release merely to add a new metric |
| `TODO_QUALITATIVE` | Display rights and anonymization check; exact image/question/mask/prediction mapping; representative failure | Replace empty case panel with real assets/captions |
| `TODO_INFERENCE_DEMO` | Verified mask-free inference command using real released weights and image/question | README Quick start; do not represent evaluator as a generic demo |
| `TODO_MODELS` | Actual downloadable weight, checksum, full dependencies, model license, successful load/generate/score | Hero/resources/checkpoint table |
| `TODO_CITATION` | Confirmed paper metadata matching public identifier | Add accurate BibTeX and only then citation-copy interaction |
| `TODO_LICENSE` | Owner's project-wide license decision plus third-party/data/weight obligations | README/site licensing and release metadata; do not auto-apply upstream license |

Close an item only after evidence is checked. Replace every dependent text/asset location, remove its visible pending message, record the evidence/version/date in this plan, set its JSON status to `resolved`, then rerun validation. Never fill a placeholder simply because the visual layout looks incomplete. Final content completion is presently **pending**, while the static website implementation is complete and previewable.

## Files and maintenance

```text
README.md
docs/
  .nojekyll
  index.html
  PROJECT_WEBSITE_PLAN.md
  VALIDATION.md
  assets/css/style.css
  assets/js/main.js
  assets/images/favicon.svg
  assets/images/teaser/task-overview.svg
  assets/images/teaser/task-overview-mobile.svg
  assets/images/framework/method-overview.svg
  assets/images/framework/medspar-architecture.png
  assets/images/dataset/construction.svg
  assets/images/dataset/construction-mobile.svg
  assets/data/project.json
  maintenance/draw_diagrams.py
  maintenance/validate_site.py
  maintenance/browser_qa.cjs
  qa/                          # validation reports and website screenshots
```

HTML5/CSS3/vanilla JS only, no external fonts/scripts/libraries. Core content is static HTML. JS only enhances dialog, navigation state and clipboard. JSON is an editorial registry, so no fetch, CORS or `file://` data dependency exists. Runtime resources use `./assets/...` and remain inside docs. README links to docs from repository root; the website does not link to local research files outside docs. Do not introduce a root `/assets/...` reference or a backend to maintain this site.

Local preview from the current workspace root:

```bash
python -m http.server 8000 --directory docs
```

Visit `http://localhost:8000`. Stop the temporary preview with Ctrl+C. No process is required after deployment.

## Testing and deployment

Run `python docs/maintenance/validate_site.py` and `node --check docs/assets/js/main.js`. Optional browser QA uses the host's installed Playwright/Chrome, not npm dependencies of this website: run `node docs/maintenance/browser_qa.cjs` with `MEDSPAR_PLAYWRIGHT_PATH`, `MEDSPAR_BROWSER_PATH` and optional `MEDSPAR_PREVIEW_URL` set if auto-discovery does not find them. With no URL override, it starts and automatically stops a temporary Python HTTP preview on a free loopback port. It checks desktop/tablet/phone widths, document overflow, asset requests, console errors, modal keyboard behavior, hash navigation, no-JS rendering, reduced-motion style and local project-subpath serving. Screenshots and machine reports are in `qa/`.

Read `VALIDATION.md` for **actual executed outcomes and limits**, including external links. Do not infer a pass from the mere presence of test scripts. No model inference, CUDA training, DDP, checkpoint load, corrected-data evaluation or independent clinical/privacy review is certified by website tests. Privacy review of shipped assets is bounded: text-only newly authored diagrams, aggregate counts, no raw cases, no copied weights, no embedded base64 medical content, and a pattern scan for common credentials.

After source/release decisions are resolved, publish only the intended reviewed files to the confirmed target GitHub repository. Do not `git add .` across this mixed untracked workspace. In GitHub **Settings → Pages**, choose **Deploy from a branch**, **main**, **/docs**, and save. `.nojekyll` keeps these files as plain static assets. The expected URL is `https://<GitHub-username>.github.io/MedSPAR/`; verify its actual deployment and resource loading before resolving `TODO_PROJECT_URL`. This task performs no commit, push or deployment. [GitHub official deployment-source instructions](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site) were consulted.
