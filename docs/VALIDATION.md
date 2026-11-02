# MedSPAR documentation and website validation

Executed on **2 November 2026**. The website is locally implemented and GitHub Pages compatible. No public deployment was performed.

## Executed checks

| Check | Actual result / evidence |
|---|---|
| Intake Git inspection | Root `C:/Users/wanghan/Documents/ChatGPT/Reaseg`, branch main, no configured remote; existing untracked work retained. New root README and docs only. |
| HTML structure | Dependency-free parser passed balanced elements, unique IDs, one H1, language/doctype, anchor and ARIA ID references. This is not a W3C validator certification. `qa/static_checks.json`. |
| Paths and assets | All HTML src/href/srcset/zoom references resolve inside docs; no domain-root asset paths; CSS has no external imports; SVGs are valid XML without scripts, embedded medical rasters or external image references. |
| JavaScript | `node --check docs/assets/js/main.js` and browser-QA helper syntax passed. Browser page/console errors: zero in final website checks. |
| Dataset inventory | Read all 30 split JSON manifests, counted 40,338 records = 31,761 train + 4,240 val + 4,337 test. HTML table and JSON register agree. `qa/manifest_audit.json`. Counts do not validate mask targets or patient isolation. |
| Data conflict reproduction | Independently counted 189 shared-image/mask groups with different targets (152 train, 20 val, 17 test), and five numeric patient-ID prefixes spanning splits. No patient IDs or source records exported. Prior audit's decoded-mask findings are attributed separately. |
| Asset provenance | Existing architecture PNG copied byte-for-byte; SHA256 matches original. Visually inspected original: schematic diagram, no patient image. Six new SVGs are original vector/text explanations or favicon. `qa/asset_provenance.json`. |
| External links | Browser-search tool retrieved the exact official SAM, MedSAM, MedReasoner repository/page and GitHub Pages documentation URLs; expected identities confirmed. `qa/external_links.json`. No fabricated MedSPAR links tested or added. |
| HTTP preview | Actual `python -m http.server` preview served HTTP 200. Automated test owns a temporary free loopback port and shuts it down; same docs directory as the user-facing port-8000 command. Initial sandbox attempts were denied/timed out; the completed run used an approved local browser/server execution context. |
| Responsive browser QA | Installed Chrome through host Playwright: **1440 × 1000**, **768 × 1024**, **390 × 844**, **320 × 740**. All images decoded; no horizontal document overflow. Screenshots recorded for all four widths. |
| Interaction | Navigation hash; dialog open/Close/Escape; return focus to trigger; copy preview command; reduced-motion smooth-scroll disable; no-JS method text and ten dataset rows. All passed. |
| Project subpath | Local static server emulated `/MedSPAR/`; images loaded under `/MedSPAR/assets/`, stylesheet applied, JS enhancement active. This verifies relative-path behavior, not an actual GitHub Pages deployment. |
| SVG geometry | All text in five explanatory diagrams stays inside its SVG viewBox. No fabricated image/mask used. |
| Visual inspection | Actual desktop and mobile browser screenshots were opened and inspected. Mobile task and construction diagrams use readable vertical variants. Existing architecture has enlarge-and-scroll support. Top crops: `qa/desktop-top.png`, `qa/mobile-top.png`; full pages in `qa/desktop.png`, `tablet.png`, `mobile.png`, `small-mobile.png`. |
| Secrets / medical content | Pattern scan for common private keys and GitHub/OpenAI/Hugging Face tokens passed in README and shipped runtime/metadata assets. No source medical image, GT, model prediction, raw patient question, identity list, checkpoint or training dataset was copied. This is a bounded content check, not an independent compliance certification. |
| Preservation | SHA256 comparison: **563 pre-existing source/paper/figure/document files checked, zero changed**. `qa/preservation_checks.json`. Scope excludes unrelated binary archives and full datasets. |
| Pending content | **20** registered replacement items in `assets/data/project.json`, with evidence and replacement locations in `PROJECT_WEBSITE_PLAN.md`. Pending URLs remain null; benchmark data and citation remain null. |

Final browser suite: **36 assertions passed**, zero console/page errors, zero failed browser requests. Details: `qa/browser_checks.json`. Screenshots are page previews, not scientific figures or evaluation evidence.

## Research-command verification limit

The actual normal command

```text
python E:/Project/MedSPAR/tools/run_experiments.py train --stage stage2 --experiment main --dry_run
```

failed with `ModuleNotFoundError: No module named 'medspar'`. The inspected package directory is `MedSPAR/` while imports request `medspar`. No source files were changed to fix this.

Separately, a temporary **in-process import alias** allowed 20 train/eval command expansions across Stages 0/1 and the eight Stage 2 experiments. Those diagnostic expansions passed without loading a model or launching training. They verify documented launcher syntax only; they do not establish a functioning canonical checkout, installed dependencies or successful inference. `qa/command_audit.json` distinguishes the two outcomes.

## Not completed / still pending

- Full research-model import, licensed model acquisition, CUDA/DDP forward/backward, trained checkpoint save/load round trip, free-generation inference and metric reproduction were **not** performed.
- Corrected source masks, globally patient-disjoint manifests and affected reevaluations remain pending. Source-summary arithmetic does not replace prediction/run provenance.
- Authors, final publication title/status, public repository identity, citation, model/data releases, pinned environment and project-wide license remain unconfirmed.
- No authorized same-record medical image/GT/prediction set was available for public qualitative figures.
- Actual GitHub Pages hosting, cross-browser Safari/Firefox behavior, physical mobile devices and a screen-reader session were not tested.
- No commit, push, upstream source migration or deployment was performed.

## Reproduce the website checks

From this workspace root:

```bash
python docs/maintenance/validate_site.py
node --check docs/assets/js/main.js
```

Optional host-browser QA (requires an already installed Playwright and Chrome; no website npm install):

```bash
node docs/maintenance/browser_qa.cjs
```

Set `MEDSPAR_PLAYWRIGHT_PATH` and `MEDSPAR_BROWSER_PATH` if host auto-discovery is unsuitable. Set `MEDSPAR_PREVIEW_URL` only when using an existing preview. Otherwise the helper starts/stops Python's static server on a free loopback port. Local network permission may be needed in a restricted execution environment. Every run records the actual outcome; do not relabel a failed run as a pass.

For normal local browsing:

```bash
python -m http.server 8000 --directory docs
```

Open `http://localhost:8000`, then stop the temporary preview with Ctrl+C. To deploy, use the confirmed repository's Settings → Pages → Deploy from a branch → main → /docs. Verify the resulting deployed address before advertising it.
