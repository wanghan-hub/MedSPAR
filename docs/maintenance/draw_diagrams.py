"""Optional authoring helper: redraw original, text-only explanatory SVGs.

No medical image, target mask, heatmap or experimental output is synthesized.
The site serves the checked-in SVGs directly and needs no build step.
"""
from pathlib import Path
from html import escape

ROOT = Path(__file__).resolve().parents[1] / "assets" / "images"
INK, TEAL, MUTED, LINE, BG, RUST = "#172d33", "#216960", "#52676d", "#d8e2de", "#f4f7f5", "#9a4b2d"


def start(width, height, title, desc):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc>',
            '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#216960"/></marker></defs>',
            f'<rect width="{width}" height="{height}" fill="#fff"/>',
            '<g font-family="Arial, sans-serif">']


def text(out, x, y, value, size=17, color=INK, weight="400", anchor="start"):
    out.append(f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">{escape(value)}</text>')


def box(out, x, y, w, h, title, lines=(), fill=BG, accent=TEAL):
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="{fill}" stroke="{LINE}"/>')
    out.append(f'<path d="M{x+1} {y+6}V{y+h-6}" stroke="{accent}" stroke-width="3"/>')
    text(out, x+18, y+29, title, 20, INK, "600")
    for i, line in enumerate(lines):
        text(out, x+18, y+56+i*23, line, 15, MUTED)


def path(out, d, dashed=False, color=TEAL):
    out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2" marker-end="url(#arrow)"'+(' stroke-dasharray="6 5"' if dashed else '')+'/>')


def save(out, relative):
    target=ROOT/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text("\n".join(out+["</g></svg>"])+"\n", encoding="utf-8")


out=start(1200,360,"MedSPAR task schematic", "Text-only task diagram. No real or synthetic medical image, mask or experimental prediction is displayed.")
text(out,40,39,"INPUT",12,TEAL,"700")
text(out,890,39,"OUTPUT",12,TEAL,"700")
box(out,40,65,300,90,"2D medical image",["Single image or extracted slice"])
box(out,40,177,300,90,"Natural-language question",["The target to be segmented"])
box(out,455,113,290,119,"MedSPAR",["Question-conditioned segmentation","with automatic spatial prompts"],"#edf5f1")
path(out,"M340 110H396V153H455")
path(out,"M340 222H396V193H455")
box(out,890,65,270,90,"Generated answer",["Exactly one valid [SEG]"])
box(out,890,177,270,90,"Binary target mask",["Produced for a valid generation"])
path(out,"M745 153H815V110H890")
path(out,"M745 193H815V222H890")
text(out,40,320,"TASK SCHEMATIC ONLY",12,RUST,"700")
text(out,340,320,"No user-provided box or point  /  Approved real case pending",15,MUTED)
save(out,"teaser/task-overview.svg")

out=start(1200,650,"MedSPAR code-derived method overview", "TC-MVE supplies a compressed prefix to the LLM, a full grid to ACSG, and direct image embeddings to the SAM decoder. SGPA produces geometric and semantic prompts. No data or predictions are illustrated.")
text(out,30,35,"QUESTION-CONDITIONED VISION",12,TEAL,"700")
text(out,425,35,"ANSWER-CONDITIONED GROUNDING",12,TEAL,"700")
text(out,875,35,"PROMPT ADAPTATION",12,TEAL,"700")
box(out,30,65,310,83,"Image + question",["Question also enters BERT and LLM"])
box(out,30,185,310,111,"TC-MVE",["BERT → connector → gated attention","before ViT blocks 2, 5, 6, 11 (0-based)"])
path(out,"M185 148V185")
box(out,425,65,340,83,"Compress + project grid X",["Visual prefix + question → LLM"])
box(out,425,185,340,111,"Answer states + SEG state",["Free generation, then hidden forward","Pool 4 answer queries + project anchor"])
path(out,"M595 148V185")
box(out,425,355,340,126,"ACSG",["Query self-attention + grid cross-attention","Updated anchor ↔ full spatial grid X","Spatial logits + grounded queries"])
path(out,"M595 296V355")
path(out,"M340 220H380V105H425")
text(out,349,197,"X",16,TEAL,"700")
path(out,"M340 263H395V416H425")
text(out,300,336,"Full grid X",14,TEAL)
box(out,875,65,295,83,"SGPA · geometry",["Predicted box + coarse-mask logits"])
box(out,875,195,295,83,"SAM prompt encoder",["Geometric sparse + dense prompts"])
path(out,"M1025 148V195")
box(out,875,335,295,106,"SGPA · semantics",["Grounded queries → project + gate","Bypass the prompt encoder"])
path(out,"M765 389H820V105H875")
path(out,"M765 431H800V389H875")
box(out,875,513,295,85,"SAM mask decoder",["Binary target mask"])
path(out,"M1170 236H1186V555H1170")
path(out,"M1025 441V513")
text(out,1003,484,"Concat sparse prompts",13,TEAL)
path(out,"M185 296V555H875")
text(out,247,541,"Post-neck image embeddings E → direct decoder path",16,TEAL,"600")
text(out,30,622,"MAIN CONFIGURATION  /  No point prompts  /  Original explanatory drawing; no experimental tensors shown",13,MUTED)
save(out,"framework/method-overview.svg")

out=start(1200,300,"Offline image-language-mask construction", "Conceptual workflow derived from processing and generation material. Annotation-derived aids belong to offline QA construction, not automatic segmentation inference.")
text(out,30,35,"OFFLINE CONSTRUCTION",12,TEAL,"700")
box(out,30,70,245,137,"Source pairs",["Image + original target mask","Source-specific preprocessing","and 2D sample extraction"])
box(out,328,70,245,137,"Language construction",["Target metadata + visual aids","QA prompt requirements","Question + reference answer"])
box(out,626,70,245,137,"Checks and review",["Format / completeness checks","Model recheck (reported)","Expert sampling (reported)"])
box(out,924,70,245,137,"Structured records",["image / input / output / mask","Auxiliary target metadata","Split manifests"])
for x in (275,573,871):path(out,f"M{x} 137H{x+53}")
text(out,30,255,"ANNOTATION BOUNDARY",12,RUST,"700")
text(out,282,255,"Boxes / mask visualizations guide offline QA; automatic inference takes image + question.",16,MUTED)
save(out,"dataset/construction.svg")

# Narrow-screen variants retain readable labels instead of shrinking a wide figure.
out=start(480,800,"MedSPAR task schematic, narrow-screen layout", "The same image/question to answer/mask task, vertically arranged. All panels are text only.")
text(out,30,35,"INPUTS",14,TEAL,"700")
box(out,30,60,400,95,"2D medical image",["Single image or extracted slice"])
box(out,30,185,400,95,"Natural-language question",["The target to be segmented"])
path(out,"M430 107H450V357H430")
path(out,"M430 232H450")
box(out,30,320,400,110,"MedSPAR",["Question-conditioned segmentation","with automatic spatial prompts"],"#edf5f1")
text(out,30,474,"OUTPUTS",14,TEAL,"700")
box(out,30,500,400,95,"Generated answer",["Exactly one valid [SEG]"])
box(out,30,625,400,95,"Binary target mask",["Produced for a valid generation"])
path(out,"M230 430V455H15V547H30")
path(out,"M15 547V672H30")
text(out,30,762,"SCHEMATIC ONLY / Approved real case pending",15,RUST,"600")
save(out,"teaser/task-overview-mobile.svg")

out=start(480,800,"Offline construction, narrow-screen layout", "The same source-pair, language construction, review and structured-record workflow, vertically arranged.")
text(out,30,32,"OFFLINE CONSTRUCTION",14,TEAL,"700")
nodes=[("Source pairs",["Image + original target mask","Source-specific preprocessing","and 2D sample extraction"]),
       ("Language construction",["Target metadata + visual aids","QA prompt requirements","Question + reference answer"]),
       ("Checks and review",["Format / completeness checks","Model recheck (reported)","Expert sampling (reported)"]),
       ("Structured records",["image / input / output / mask","Auxiliary target metadata","Split manifests"])]
for i,(title,lines) in enumerate(nodes):
    y=60+i*170
    box(out,30,y,420,135,title,lines)
    if i<3:path(out,f"M240 {y+135}V{y+170}")
text(out,30,777,"Offline visual aids are not inference inputs.",16,RUST)
save(out,"dataset/construction-mobile.svg")

(ROOT/'favicon.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><title>MedSPAR</title><rect width="64" height="64" rx="12" fill="#173d43"/><path d="M14 46V18h8l10 17 10-17h8v28h-8V31L32 47 22 31v15z" fill="white"/></svg>\n',encoding='utf-8')
print("Wrote 5 original explanatory SVGs and favicon; no medical data used.")
