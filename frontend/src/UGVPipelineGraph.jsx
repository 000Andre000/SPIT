  import { useState, useRef, useEffect, useCallback, useMemo } from "react";

  // ═══════════════════════════════════════════════════════════════
  //  COLOUR PALETTE
  // ═══════════════════════════════════════════════════════════════
  const C = {
    data:   "#ffffff",
    pre:    "#00cfff",
    xent:   "#fb923c",
    focal:  "#f97316",
    dino:   "#a855f7",
    seg:    "#38bdf8",
    nav:    "#22c55e",
    plan:   "#eab308",
    result: "#00ffaa",
  };

  const COLOR_OPTIONS = [
    { label: "White (Data)",      value: "#ffffff" },
    { label: "Cyan (Preprocess)", value: "#00cfff" },
    { label: "Orange (XEnt)",     value: "#fb923c" },
    { label: "Amber (Focal)",     value: "#f97316" },
    { label: "Purple (DINOv2)",   value: "#a855f7" },
    { label: "Sky (SegFormer)",   value: "#38bdf8" },
    { label: "Green (Nav)",       value: "#22c55e" },
    { label: "Yellow (Plan)",     value: "#eab308" },
    { label: "Teal (Result)",     value: "#00ffaa" },
    { label: "Pink",              value: "#ec4899" },
    { label: "Red",               value: "#ef4444" },
  ];

  const getVideoMimeType = (url = "") => {
    const clean = String(url).split("?")[0].toLowerCase();
    if (clean.endsWith(".webm")) return "video/webm";
    if (clean.endsWith(".ogg") || clean.endsWith(".ogv")) return "video/ogg";
    if (clean.endsWith(".mov")) return "video/quicktime";
    if (clean.endsWith(".mkv")) return "video/x-matroska";
    return "video/mp4";
  };

  const normalizeVideoUrl = (input = "") => {
    const value = String(input || "").trim();
    if (!value) return "";
    if (/^https?:\/\//i.test(value) || value.startsWith("blob:")) return value;
    const cleaned = value.replace(/^\/+/, "");
    const encodedPath = cleaned.split("/").filter(Boolean).map(part => encodeURIComponent(part)).join("/");
    if (cleaned.startsWith("videos/")) return `/${encodedPath}`;
    return `/videos/${encodedPath}`;
  };

  const toBackendUrl = (input = "") => {
    const value = String(input || "").trim();
    if (!value) return "";
    if (/^https?:\/\//i.test(value)) return value;
    return `http://localhost:8000${value.startsWith("/") ? "" : "/"}${value}`;
  };

  const toConvertedPreviewUrl = (input = "") => {
    const raw = String(input || "").trim();
    if (!raw) return "";
    const fileName = raw.split("/").pop() || raw;
    const noExt = fileName.replace(/\.[^/.]+$/, "");
    const safeBase = noExt.replace(/\s+/g, "_").replace(/[^a-zA-Z0-9_\-]/g, "");
    return safeBase ? `/videos/converted/${safeBase}.mp4` : "";
  };

  // ═══════════════════════════════════════════════════════════════
  //  INITIAL NODE DATA
  // ═══════════════════════════════════════════════════════════════
  const INITIAL_NODES = [
    // Shared: Dataset & Preprocessing
    { id:"dataset",      group:"shared",    layer:0, color:C.data,   
      label:"DESERT\nDATASET",         sublabel:"Synthetic · 10 classes",
      metric:"10 classes", mLabel:"Classes",  shapeOut:"960×540 RGB",   ms:0,
      parentId:null,
      desc:"Synthetic digital-twin desert dataset. 10 semantic classes: Sky (37.64%), Landscape (24.45%), Rocks (1.20%), Dry Bushes (1.10%), Logs (0.08%). Highly imbalanced — rare classes are safety-critical for UGV navigation.",
      plain:"Raw synthetic desert images with pixel-level terrain labels." },

    { id:"brightness",   group:"shared",    layer:1, color:C.pre,
      label:"BRIGHTNESS\nNORMALISE",   sublabel:"Gamma + histogram eq",
      metric:"100%",      mLabel:"Pass rate", shapeOut:"960×540 float32", ms:3,
      parentId:"dataset",
      desc:"Gamma and histogram normalisation to balance synthetic brightness. Prevents the model from learning exposure artefacts instead of terrain features.",
      plain:"Fixes lighting so shadows and bright sky do not confuse the model." },

    { id:"dehazing",     group:"shared",    layer:1, color:C.pre,   
      label:"DEHAZING",                sublabel:"Dark-channel prior",
      metric:"100%",      mLabel:"Pass rate", shapeOut:"960×540 float32", ms:5,
      parentId:"brightness",
      desc:"Dark-channel prior dehazing removes atmospheric noise from synthetic desert renders. Improves edge clarity for small obstacles.",
      plain:"Removes foggy haze so small rocks and logs are clearly visible." },

    { id:"class_weight", group:"shared",    layer:1, color:C.pre,   
      label:"CLASS\nWEIGHTING",        sublabel:"MFB · w=median(f)/f",
      metric:"MFB",       mLabel:"Method",    shapeOut:"10-dim weights",  ms:1,
      parentId:"dehazing",
      desc:"Median Frequency Balancing: w = median(freq)/freq. Sky & Landscape get low weights; Logs and Rocks get high weights (max=10). Prevents model from ignoring safety-critical rare classes.",
      plain:"Tells the model to care more about rare dangerous obstacles." },

    { id:"rand_crop",    group:"shared",    layer:1, color:C.pre, 
      label:"RANDOM\nCROP",            sublabel:"Divisible by 14 · hi-res",
      metric:"÷14",       mLabel:"Patch align",shapeOut:"Cropped patch",  ms:2,
      parentId:"class_weight",
      desc:"Random crop patches divisible by 14 for DINOv2 patch-token compatibility. Forces the model to learn small obstacle regions and improves scale robustness.",
      plain:"Randomly zooms in so the model learns to spot small rocks and logs." },

    // Model Branch 1 — Weighted Cross-Entropy
    { id:"xent_root",    group:"xent",      layer:2, color:C.xent,  
      label:"CROSS-ENTROPY\nLOSS",     sublabel:"Experiment ①",
      metric:"~5%",       mLabel:"Val mIoU",  shapeOut:"[1,10,H,W]",    ms:45,
      parentId:"rand_crop",
      desc:"Standard weighted cross-entropy: L = -Σ w_c·y_c·log(p_c). Bias toward dominant classes (Sky, Landscape) persisted. Small obstacles still poorly detected.",
      plain:"First attempt — weighted penalty for wrong predictions. Struggled with rare obstacles." },

    { id:"xent_enc",     group:"xent",      layer:3, color:C.xent, 
      label:"CNN\nENCODER",            sublabel:"Basic backbone",
      metric:"weak",      mLabel:"Repr",      shapeOut:"Feature maps",   ms:30,
      parentId:"xent_root",
      desc:"Basic CNN encoder without pretrained weights. Lacked semantic richness for 10 visually similar desert terrain classes. Identified as the primary bottleneck.",
      plain:"Basic image reader — not powerful enough for complex desert terrain." },

    { id:"xent_out",     group:"xent",      layer:3, color:C.xent,   
      label:"CLASS MAP\n~5% IoU",      sublabel:"Biased to sky & landscape",
      metric:"~5%",       mLabel:"mIoU",      shapeOut:"[1,10,H,W]",    ms:5,
      parentId:"xent_enc",
      desc:"Per-pixel softmax + argmax. Very low mIoU (~5%). Heavy bias toward sky and landscape. Logs, rocks, dry bushes almost never correctly detected.",
      plain:"Pixel labels produced — very low accuracy on dangerous rare obstacles." },

    // Model Branch 2 — Focal Loss
    { id:"focal_root",   group:"focal",     layer:2, color:C.focal,  
      label:"FOCAL\nLOSS",             sublabel:"Experiment ② · γ focusing",
      metric:"5.11%",     mLabel:"Val mIoU",  shapeOut:"[1,10,H,W]",    ms:42,
      parentId:"rand_crop",
      desc:"FL = -α(1−pt)^γ·log(pt). Down-weights easy pixels, focuses on hard misclassifications. Train Loss=1.4084, Val Loss=1.3919. IoU=0.0511, Dice=0.0871, PixAcc=0.0955.",
      plain:"Focal Loss focuses on hard-to-classify pixels. Better than cross-entropy but still limited." },

    { id:"focal_enc",    group:"focal",     layer:3, color:C.focal, 
      label:"CNN\nENCODER",            sublabel:"Same backbone · bottleneck",
      metric:"weak",      mLabel:"Repr",      shapeOut:"Feature maps",   ms:28,
      parentId:"focal_root",
      desc:"Same basic CNN encoder as Experiment 1. Despite better loss dynamics from Focal Loss, the encoder still lacked depth for challenging off-road terrain.",
      plain:"Same basic image reader — the loss improved but backbone was still the problem." },

    { id:"focal_out",    group:"focal",     layer:3, color:C.focal,  
      label:"VAL METRICS\nIoU 0.051",  sublabel:"Dice 0.087 · PixAcc 0.095",
      metric:"5.11%",     mLabel:"mIoU",      shapeOut:"[1,10,H,W]",    ms:5,
      parentId:"focal_enc",
      desc:"Train: Loss=1.4084, IoU=0.0527, Dice=0.0903, PixAcc=0.0935. Val: Loss=1.3919, IoU=0.0511, Dice=0.0871, PixAcc=0.0955. Confirmed backbone upgrade to DINOv2 was needed.",
      plain:"The numbers confirmed the backbone was the real problem — not the loss function." },

    // Model Branch 3 — DINOv2 + ConvNeXt (WINNING)
    { id:"dino_root",    group:"dinov2",    layer:2, color:C.dino, 
      label:"DINOv2\nBACKBONE ★",     sublabel:"ViT-S/14 · 21M · FROZEN",
      metric:"94%",       mLabel:"Confidence",shapeOut:"[1,646,384]",   ms:42,
      parentId:"rand_crop",
      desc:"Pretrained DINOv2 ViT-S/14 frozen feature extractor. Self-supervised pretraining gives rich patch-level semantic embeddings. Freezing prevents overfitting and reduces training cost.",
      plain:"Powerful pretrained AI brain — understands terrain textures without extra training." },

    { id:"dino_attn",    group:"dinov2",    layer:3, color:C.dino,  
      label:"SELF-ATTENTION\n646 patches",sublabel:"6 heads · 12 layers",
      metric:"646 tokens",mLabel:"Patch tokens",shapeOut:"[1,646,384]", ms:28,
      parentId:"dino_root",
      desc:"Image divided into 34×19=646 patches of 14×14 pixels. Multi-head self-attention propagates context across all patches. A rock patch attends to surrounding terrain for accurate scene understanding.",
      plain:"Each image patch looks at every other patch to understand its full context." },

    { id:"convnext",     group:"dinov2",    layer:3, color:C.dino,   
      label:"ConvNeXt\nSEG HEAD",      sublabel:"7×7 stem → DW → 1×1 → ↑",
      metric:"88%",       mLabel:"Confidence",shapeOut:"[1,10,H,W]",   ms:14,
      parentId:"dino_attn",
      desc:"ConvNeXt-style segmentation head: reshape 646 tokens → 7×7 depthwise conv (384→128ch) → depthwise-separable block → 1×1 classifier (128→10) → bilinear upsample to full resolution.",
      plain:"Translates DINOv2 features into a per-pixel terrain label for every part of the image." },

    { id:"dino_out",     group:"dinov2",    layer:3, color:C.dino,  
      label:"PIXEL\nCLASS MAP",        sublabel:"Best results · rare class IoU ↑",
      metric:"high",      mLabel:"mIoU",      shapeOut:"[1,10,266,476]",ms:4,
      parentId:"convnext",
      desc:"Full-resolution 10-class pixel label map. Best performance across all models — especially on rare safety-critical classes (logs, rocks, dry bushes) thanks to DINOv2 strong visual representations.",
      plain:"Every pixel labelled — best accuracy of all four models." },

    // Model Branch 4 — SegFormer
    { id:"seg_root",     group:"segformer", layer:2, color:C.seg,  
      label:"SegFormer",               sublabel:"MiT encoder · MLP decoder",
      metric:"86%",       mLabel:"Confidence",shapeOut:"[1,10,H,W]",   ms:38,
      parentId:"rand_crop",
      desc:"End-to-end transformer segmentation. MiT encoder extracts features at 4 scales without positional encoding — robust to resolution variation. Lightweight all-MLP decoder fuses scales efficiently.",
      plain:"An all-in-one fast model reading the image at 4 zoom levels simultaneously." },

    { id:"mit_enc",      group:"segformer", layer:3, color:C.seg,  
      label:"MiT\nENCODER",           sublabel:"4-scale · no pos encoding",
      metric:"4 scales",  mLabel:"Feature scales",shapeOut:"H/4,H/8,H/16,H/32",ms:22,
      parentId:"seg_root",
      desc:"Mix Transformer hierarchical encoder: features at 1/4, 1/8, 1/16, 1/32 resolution. No positional encoding makes it robust to varying image sizes in off-road deployments.",
      plain:"Reads the image at 4 zoom levels — captures both big patterns and fine details." },

    { id:"mlp_dec",      group:"segformer", layer:3, color:C.seg,   
      label:"MLP\nDECODER",           sublabel:"Lightweight fusion · fast",
      metric:"fast",      mLabel:"Inference", shapeOut:"[1,10,H,W]",   ms:10,
      parentId:"mit_enc",
      desc:"All-MLP decoder fuses 4 encoder feature scales into a unified segmentation map. Much lighter than UPerNet or FPN decoders while maintaining competitive accuracy.",
      plain:"Combines all zoom-level features quickly into one final segmentation map." },

    { id:"seg_out",      group:"segformer", layer:3, color:C.seg,   
      label:"PIXEL\nCLASS MAP",       sublabel:"Competitive · scale-robust",
      metric:"86%",       mLabel:"Confidence",shapeOut:"[1,10,H,W]",   ms:6,
      parentId:"mlp_dec",
      desc:"Full-resolution 10-class pixel label map. Competitive accuracy with faster inference than DINOv2+ConvNeXt. Strong robustness to scale variation.",
      plain:"Every pixel labelled — fast and accurate alternative to the DINOv2 path." },

    // Convergence + Navigation
    { id:"collision",    group:"shared",    layer:4, color:C.nav,   
      label:"COLLISION\nDETECTION",    sublabel:"All models converge here",
      metric:"4-neigh",   mLabel:"Check method",shapeOut:"Risk: LOW/MED/HIGH",ms:6,
      parentId:null,
      desc:"All four model outputs converge here. Semantic class map analysed: Logs and Rocks flagged as BLOCKED. 4-neighbour proximity scan checks each path candidate. Outputs risk level: LOW / MED / HIGH.",
      plain:"All models feed here — checks which terrain areas are dangerous for the UGV." },

    { id:"cost_map",     group:"shared",    layer:5, color:C.nav,  
      label:"COST MAP\nTerrain→Cost",  sublabel:"Traversability grid",
      metric:"48×30",     mLabel:"Grid size", shapeOut:"48×30 float grid",ms:5,
      parentId:"collision",
      desc:"Maps 10 class IDs to movement costs: Sky=∞, Log=∞, Rock=high, DryBush=medium, Landscape=low. Coarse 48×30 grid derived by max-pooling cost within each cell.",
      plain:"Creates a difficulty grid — green for easy terrain, red for blocked obstacles." },

    { id:"astar",        group:"shared",    layer:5, color:C.plan, 
      label:"A* PATH\nPLANNING",       sublabel:"f(n) = g(n) + h(n)",
      metric:"Manhattan", mLabel:"Heuristic", shapeOut:"Path: N waypoints",ms:28,
      parentId:"cost_map",
      desc:"A* graph search on the 48×30 cost grid with Manhattan heuristic. Finds minimum-cost path from UGV position to goal. Avoids BLOCKED cells (logs, rocks). Produces ordered waypoints.",
      plain:"Finds the safest, lowest-cost route through the terrain to the destination." },

    { id:"waypoint",     group:"shared",    layer:5, color:C.plan, 
      label:"NEXT\nWAYPOINT",          sublabel:"Path[0] → heading + speed",
      metric:"30 Hz",     mLabel:"Update rate",shapeOut:"(x,y) + heading°",ms:2,
      parentId:"astar",
      desc:"First node on the A* path issued as next waypoint. Includes: target (x,y), heading angle, recommended speed %. Sent to motor controller at up to 30Hz.",
      plain:"Picks the very next position the UGV should move to and at what speed." },

    { id:"ugv_move",     group:"shared",    layer:6, color:C.result,
      label:"UGV\nMOVEMENT",           sublabel:"PROCEED / HALT · 30Hz",
      metric:"30 Hz",     mLabel:"Control rate",shapeOut:"Motor CMD",     ms:2,
      parentId:"waypoint",
      desc:"Final motor command: heading angle, speed %, PROCEED/HALT flag sent to UGV motor controller. Full pipeline runs at up to 30Hz. Emergency HALT triggered if HIGH risk detected.",
      plain:"The UGV gets its final instructions — direction, speed, and whether it is safe to proceed." },
  ];

  const MODEL_OUTPUTS = { xent:"xent_out", focal:"focal_out", dinov2:"dino_out", segformer:"seg_out" };
  const MODEL_GROUPS  = ["xent","focal","dinov2","segformer"];

  function getVisibleIds(nodes, model) {
    const allIds = nodes.map(n => n.id);
    if (model === "all") return allIds;
    const modelNodes = nodes.filter(n => n.group === model || n.group === "shared").map(n => n.id);
    return modelNodes;
  }

  // ═══════════════════════════════════════════════════════════════
  //  LAYOUT HELPERS
  // ═══════════════════════════════════════════════════════════════
  // Smarter overlap resolver: use node sizes + margin to avoid collisions
  function resolveOverlaps(positions, nodesList = [], margin = 8, iterations = 6) {
    const ids = Object.keys(positions);
    if (ids.length < 2) return positions;

    // build quick lookup for node sizes (approximate bounding circle)
    const sizeMap = {};
    const rectHalfW = NW / 2;
    const rectHalfH = NH / 2;
    const rectRadius = Math.sqrt(rectHalfW * rectHalfW + rectHalfH * rectHalfH);
    ids.forEach(id => {
      const node = nodesList.find(n => n.id === id) || {};
      const r = node.shape === "hex" ? NR : rectRadius;
      sizeMap[id] = r + margin;
    });

    for (let it = 0; it < iterations; it++) {
      for (let i = 0; i < ids.length; i++) {
        for (let j = i + 1; j < ids.length; j++) {
          const a = ids[i];
          const b = ids[j];
          const pa = positions[a];
          const pb = positions[b];
          if (!pa || !pb) continue;
          let dx = pa.x - pb.x;
          let dy = pa.y - pb.y;
          let d2 = dx * dx + dy * dy;
          if (d2 === 0) {
            dx = (Math.random() - 0.5) * 0.1;
            dy = (Math.random() - 0.5) * 0.1;
            d2 = dx * dx + dy * dy;
          }
          const d = Math.sqrt(d2);
          const minDist = (sizeMap[a] || 40) + (sizeMap[b] || 40);
          if (d < minDist) {
            const overlap = (minDist - d) / 2;
            const ux = dx / d;
            const uy = dy / d;
            positions[a] = { x: pa.x + ux * overlap, y: pa.y + uy * overlap };
            positions[b] = { x: pb.x - ux * overlap, y: pb.y - uy * overlap };
          }
        }
      }
    }
    return positions;
  }

  function computeLayout(nodes, model, W, H) {
    const visible = getVisibleIds(nodes, model);
    const visNodes = nodes.filter(n => visible.includes(n.id));
    const layers = {};
    visNodes.forEach(n => { if (!layers[n.layer]) layers[n.layer] = []; layers[n.layer].push(n); });
    const numL = Object.keys(layers).length || 1;
    const levelH = Math.min(140, (H - 100) / numL);
    const positions = {};

    const presentModels = MODEL_GROUPS.filter(m => visNodes.some(n => n.group === m && n.layer === 2));
    const numCols = presentModels.length || 1;
    const colSpacing = Math.min(260, (W - 100) / (numCols + 0.35));
    const colStartX = (W - colSpacing * (numCols - 1)) / 2;
    const modelColX = {};
    presentModels.forEach((m, i) => { modelColX[m] = colStartX + i * colSpacing; });

    Object.keys(layers).sort((a, b) => a - b).forEach((lv, li) => {
      const lvi = parseInt(lv);
      const group = layers[lv];
      const rowY = 70 + li * levelH;

      if (lvi === 2 || lvi === 3) {
        const byGroup = {};
        group.forEach(n => { if (!byGroup[n.group]) byGroup[n.group] = []; byGroup[n.group].push(n); });
        Object.entries(byGroup).forEach(([m, ns]) => {
          const cx = modelColX[m] ?? W / 2;
          const count = ns.length;
          const localSpacing = Math.min(92, Math.max(52, colSpacing * 0.42));
          const startX = cx - ((count - 1) * localSpacing) / 2;
          ns.forEach((n, ni) => {
            const x = startX + ni * localSpacing;
            positions[n.id] = { x, y: rowY };
          });
        });
      } else {
        const spacing = Math.min(240, (W - 80) / (group.length + 0.35));
        const startX = (W - spacing * (group.length - 1)) / 2;
        group.forEach((n, gi) => { positions[n.id] = { x: startX + gi * spacing, y: rowY }; });
      }
    });

    // nudge overlapping nodes so labels don't collide (use visible node data)
    resolveOverlaps(positions, visNodes, 8, 6);
    return positions;
  }

  // ═══════════════════════════════════════════════════════════════
  //  SVG EDGE
  // ═══════════════════════════════════════════════════════════════
  function EdgeSVG({ from, to, color }) {
    if (!from || !to) return null;
    const x1 = from.x, y1 = from.y, x2 = to.x, y2 = to.y;
    const my = (y1 + y2) / 2;
    const d = `M ${x1} ${y1} C ${x1} ${my} ${x2} ${my} ${x2} ${y2}`;
    return (
      <path
        d={d}
        fill="none"
        stroke={color + "55"}
        strokeWidth="1.5"
        strokeDasharray="5 3"
        style={{ transition: "d 0.4s ease" }}
      />
    );
  }

  // ═══════════════════════════════════════════════════════════════
  //  NODE SHAPE COMPONENTS
  // ═══════════════════════════════════════════════════════════════
  function HexOutline({ r, color, filled, active, selected }) {
    const pts = Array.from({ length: 6 }, (_, i) => {
      const a = (i * Math.PI) / 3 - Math.PI / 6;
      return `${r * Math.cos(a)},${r * Math.sin(a)}`;
    }).join(" ");
    return (
      <polygon
        points={pts}
        fill={filled ? color + "16" : active ? "rgba(255,204,0,0.12)" : "#060c13"}
        stroke={active ? "#ffcc00" : selected ? "#ffffff" : color}
        strokeWidth={active || selected ? 2 : 1.5}
        style={{
          filter: active || selected ? `drop-shadow(0 0 8px ${active ? "#ffcc00" : color})` : filled ? `drop-shadow(0 0 4px ${color}66)` : "none",
          transition: "all 0.2s",
        }}
      />
    );
  }

  function RectOutline({ w, h, color, filled, active, selected }) {
    return (
      <rect
        x={-w} y={-h} width={w * 2} height={h * 2} rx={4}
        fill={filled ? color + "10" : active ? "rgba(255,204,0,0.08)" : "#060c13"}
        stroke={active ? "#ffcc00" : selected ? "#ffffff" : color + "99"}
        strokeWidth={active || selected ? 2 : 1}
        style={{
          filter: active || selected ? `drop-shadow(0 0 6px ${active ? "#ffcc00" : color})` : filled ? `drop-shadow(0 0 3px ${color}44)` : "none",
          transition: "all 0.2s",
        }}
      />
    );
  }

  // ═══════════════════════════════════════════════════════════════
  //  SINGLE NODE COMPONENT
  // ═══════════════════════════════════════════════════════════════
  const NW = 100, NH = 44, NR = 33;

  function PipelineNode({ node, pos, selected, onSelect, onDragStart, isWinner }) {
    const isHex = node.shape === "hex";
    const lines = node.label.split("\n");
    const lh = isHex ? 13 : 11;
    const botY = isHex ? NR : NH / 2;

    return (
      <g
        transform={`translate(${pos.x},${pos.y})`}
        style={{ cursor: "pointer", userSelect: "none" }}
        onClick={e => { e.stopPropagation(); onSelect(node.id); }}
        onMouseDown={e => { e.stopPropagation(); onDragStart(e, node.id); }}
      >
        {isHex
          ? <HexOutline r={NR} color={node.color} filled={true} active={false} selected={selected} />
          : <RectOutline w={NW / 2} h={NH / 2} color={node.color} filled={true} active={false} selected={selected} />
        }

        {lines.map((line, i) => (
          <text
            key={i}
            x={0} y={(i - (lines.length - 1) / 2) * lh}
            textAnchor="middle" dominantBaseline="middle"
            fontSize={isHex ? 10 : 9}
            fontWeight={700}
            fontFamily="'JetBrains Mono', monospace"
            fill={selected ? "#ffffff" : node.color}
            style={{ pointerEvents: "none", transition: "fill 0.2s" }}
          >
            {line}
          </text>
        ))}

        {node.sublabel && (
          <text
            x={0} y={botY - 7}
            textAnchor="middle" dominantBaseline="middle"
            fontSize={7} fontFamily="'JetBrains Mono', monospace"
            fill={node.color + "44"}
            style={{ pointerEvents: "none" }}
          >
            {node.sublabel}
          </text>
        )}

        {node.ms > 0 && (
          <>
            <rect x={-20} y={botY} width={40} height={13} rx={2} fill="rgba(0,0,0,0.8)" stroke={node.color + "45"} strokeWidth={0.5} />
            <text x={0} y={botY + 7} textAnchor="middle" dominantBaseline="middle"
              fontSize={8} fontWeight={700} fontFamily="'JetBrains Mono', monospace" fill={node.color}
              style={{ pointerEvents: "none" }}
            >{node.ms}ms</text>
          </>
        )}

        {isWinner && (
          <text x={NR - 3} y={-NR + 4} textAnchor="end" dominantBaseline="hanging"
            fontSize={11} fill="#fde047" style={{ pointerEvents: "none" }}
          >★</text>
        )}
      </g>
    );
  }

  // ═══════════════════════════════════════════════════════════════
  //  DETAILS PANEL — read-only side panel for node info
  // ═══════════════════════════════════════════════════════════════
  function DetailsPanel({ node, onClose }) {
    if (!node) return null;
    // Use static images from `src/assets` as placeholders for common model outputs
    const PLACEHOLDER_MAP = {
      xent_out: new URL("./assets/cross_entropy.png", import.meta.url).href,
      focal_out: new URL("./assets/foucalLoss.png", import.meta.url).href,
      dino_out: new URL("./assets/baseline.png", import.meta.url).href,
      seg_out: new URL("./assets/segformer.png", import.meta.url).href,
      rand_crop: new URL("./assets/randormCorp.png", import.meta.url).href,
    };
    const getPlaceholderDataUrl = id => PLACEHOLDER_MAP[id] || null;
    const fieldView = (label, value) => (
      <div style={{ marginBottom: 8 }}>
        <div style={{ fontSize: 7, letterSpacing: "1.5px", color: "#2a4a60", textTransform: "uppercase", marginBottom: 4 }}>{label}</div>
        <div style={{ fontSize: 11, color: "#cfeaf8", fontFamily: "'JetBrains Mono', monospace", whiteSpace: "pre-wrap" }}>{value ?? "—"}</div>
      </div>
    );

    const imgSrc = node.metricImage || node.metricImageUrl || getPlaceholderDataUrl(node.id);

    return (
      <div style={{
        position: "absolute", right: 0, top: 0, bottom: 0, width: 300,
        background: "#070d14", borderLeft: "1px solid #1a3550",
        display: "flex", flexDirection: "column", zIndex: 50, overflowY: "auto",
      }}>
        <div style={{
          padding: "12px 14px", borderBottom: "1px solid #0f2035",
          display: "flex", alignItems: "center", gap: 8, flexShrink: 0,
        }}>
          <div style={{ width: 10, height: 10, borderRadius: "50%", background: node.color, boxShadow: `0 0 8px ${node.color}` }} />
          <span style={{ fontFamily: "'Syne', sans-serif", fontWeight: 800, fontSize: 13, color: "#d0e8f4", letterSpacing: 1 }}>
            Node Details
          </span>
          <button onClick={onClose} style={{ ...btnStyle, marginLeft: "auto", padding: "2px 8px", fontSize: 10 }}>✕</button>
        </div>
        {imgSrc && (
          <div style={{ padding: 12 }}>
            <img src={imgSrc} alt="metrics" style={{ width: "100%", borderRadius: 4, border: "1px solid rgba(255,255,255,0.03)", background: "#000" }} />
          </div>
        )}

        {/* video output (if present) */}
        {(node.metricVideo || node.metricVideoUrl) && (
          <div style={{ padding: 12 }}>
            <video controls preload="metadata" crossOrigin="anonymous" style={{ width: "100%", borderRadius: 4, background: "#000" }} onError={e => console.error('Video playback error', e)}>
              <source src={normalizeVideoUrl(node.metricVideo || node.metricVideoUrl)} type={getVideoMimeType(node.metricVideo || node.metricVideoUrl)} />
              Your browser does not support the video tag.
            </video>
          </div>
        )}

        <div style={{ padding: "14px 14px", flex: 1, overflowY: "auto", }}>
          {fieldView("Label", node.label)}
          {fieldView("Sublabel", node.sublabel)}
          <div style={{ marginBottom: 10, display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{ width: 16, height: 16, borderRadius: "50%", background: node.color }} />
            <span style={{ fontSize: 11, color: node.color, fontFamily: "'JetBrains Mono', monospace" }}>{node.color}</span>
          </div>

          {fieldView("Shape", node.shape)}
          {fieldView("Group", node.group)}
          {fieldView("Layer", node.layer)}
          {fieldView("Latency (ms)", node.ms)}
          {fieldView("Metric", node.metric)}
          {fieldView("Metric Label", node.mLabel)}
          {fieldView("Output Shape", node.shapeOut)}
          {fieldView("Plain", node.plain)}
          {fieldView("Technical", node.desc)}
        </div>

        <div style={{ padding: "10px 14px", borderTop: "1px solid #0f2035", display: "flex", gap: 8, flexShrink: 0 }}>
          <button onClick={onClose} style={{ ...btnStyle, flex: 1 }}>CLOSE</button>
        </div>
      </div>
    );
  }

  function VideosSection({ videos, onPlay, onClose }) {
    return (
      <div style={{
        position: "absolute", right: 0, top: 52, bottom: 0, width: 360,
        background: "#061116", borderLeft: "1px solid #112334",
        display: "flex", flexDirection: "column", zIndex: 60, overflowY: "auto",
      }}>
        <div style={{ padding: 12, borderBottom: "1px solid #0f2035", display: "flex", alignItems: "center" }}>
          <span style={{ fontFamily: "'Syne', sans-serif", fontWeight: 800, fontSize: 13, color: "#d0e8f4" }}>Videos</span>
          <button onClick={onClose} style={{ ...btnStyle, marginLeft: "auto", padding: "2px 8px", fontSize: 10 }}>✕</button>
        </div>
        <div style={{ padding: 12, display: "grid", gridTemplateColumns: "1fr", gap: 10 }}>
          {(!videos || videos.length === 0) && <div style={{ color: "#7a9cb0", fontSize: 12 }}>No videos found in /videos. Place an index.json or video files in public/videos/</div>}
          {videos.map(v => (
            <div key={v.url} style={{ background: "#07121a", border: "1px solid #172a3a", padding: 8, borderRadius: 6 }}>
              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <div style={{ width: 140, height: 80, background: "#000", borderRadius: 4, overflow: "hidden", flexShrink: 0 }}>
                  <video muted autoPlay loop playsInline preload="metadata" crossOrigin="anonymous" style={{ width: "100%", height: "100%", objectFit: "cover" }} ref={el => { /* keep ref-free */ }} onError={e => { try { const el = e?.target || e?.currentTarget; tryVideoFallbacks(el, v); } catch(err){ console.error('Thumbnail video error', err); } }}>
                    <source src={v.previewUrl || normalizeVideoUrl(v.url)} type={getVideoMimeType(v.previewUrl || v.url)} />
                  </video>
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ color: "#cfeaf8", fontSize: 12, fontWeight: 700 }}>{v.title}</div>
                  <div style={{ color: "#7a9cb0", fontSize: 11, marginTop: 6 }}>{v.url}</div>
                  <div style={{ marginTop: 8, display: "flex", gap: 8 }}>
                    <button onClick={() => onPlay(v.url)} style={{ ...btnStyle, padding: "6px 10px", width: 96 }}>Play</button>
                    <a href={v.url} target="_blank" rel="noreferrer" style={{ textDecoration: "none" }}>
                      <button style={{ ...btnStyle, padding: "6px 10px", width: 96 }}>Open</button>
                    </a>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  const inputStyle = {
    width: "100%", background: "rgba(0,0,0,0.5)", border: "1px solid #1a3550",
    borderRadius: 3, padding: "5px 8px", fontSize: 10,
    color: "#d0e8f4", fontFamily: "'JetBrains Mono', monospace", outline: "none",
    boxSizing: "border-box", resize: "vertical",
  };
  const btnStyle = {
    width: "100%", padding: "7px 14px", background: "rgba(0,0,0,0.3)",
    border: "1px solid #1a3550", color: "#7a9cb0",
    fontFamily: "'JetBrains Mono', monospace", fontSize: 9, letterSpacing: "1.5px",
    fontWeight: 700, cursor: "pointer", borderRadius: 2, textTransform: "uppercase",
    transition: "all 0.2s",
  };

  // ═══════════════════════════════════════════════════════════════
  //  VIDEO UI: modal player and videos tray
  // ═══════════════════════════════════════════════════════════════
  function VideoModal({ src, onClose }) {
    if (!src) return null;
    return (
      <div style={{ position: "fixed", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(0,0,0,0.7)", zIndex: 300 }}>
        <div style={{ width: "80%", maxWidth: 900, background: "#070d14", border: "1px solid #1a3550", padding: 12, borderRadius: 6 }}>
          <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
            <button onClick={onClose} style={{ ...btnStyle, width: "auto", padding: "6px 10px" }}>Close</button>
          </div>
          <video key={src} controls autoPlay playsInline preload="metadata" crossOrigin="anonymous" style={{ width: "100%", background: "#000" }} onError={e => { try { const el = e?.target || e?.currentTarget; tryVideoFallbacks(el, { url: src, title: src.split('/').pop() }); } catch(err){ console.error('Modal video error', err); } }}>
            <source src={normalizeVideoUrl(src)} type={getVideoMimeType(src)} />
          </video>
        </div>
      </div>
    );
  }

  function VideosTray({ nodesWithVideo, onPlay }) {
    if (!nodesWithVideo || nodesWithVideo.length === 0) return null;
    return (
      <div style={{ position: "absolute", bottom: 16, left: 16, background: "rgba(7,13,20,0.9)", border: "1px solid #1a3550", padding: 8, borderRadius: 4, zIndex: 40 }}>
        <div style={{ fontSize: 8, color: "#2a4a60", textTransform: "uppercase", marginBottom: 6 }}>Videos</div>
        <div style={{ display: "flex", gap: 6 }}>
          {nodesWithVideo.map(n => (
            <button key={n.id} onClick={() => onPlay(n.metricVideo || n.metricVideoUrl)} style={{ ...btnStyle, padding: "6px 8px", width: 110, fontSize: 11 }}>
              {n.label.split("\n")[0]}
            </button>
          ))}
        </div>
      </div>
    );
  }

  // ═══════════════════════════════════════════════════════════════
  //  ADD NODE MODAL
  // ═══════════════════════════════════════════════════════════════
  function AddNodeModal({ parentId, onConfirm, onCancel }) {
    const [form, setForm] = useState({
      label: "NEW NODE", sublabel: "Description", color: "#00cfff",
      shape: "rect", group: "shared", layer: 2, ms: 10,
      metric: "—", mLabel: "Metric", shapeOut: "[1,C,H,W]",
      plain: "Plain description.", desc: "Technical description.",
      parentId,
    });

    return (
      <div style={{
        position: "fixed", inset: 0, background: "rgba(0,0,0,0.7)", display: "flex",
        alignItems: "center", justifyContent: "center", zIndex: 200, backdropFilter: "blur(4px)",
      }}>
        <div style={{ background: "#070d14", border: "1px solid #1a3550", borderRadius: 6, width: 340, maxHeight: "80vh", overflow: "hidden", display: "flex", flexDirection: "column" }}>
          <div style={{ padding: "12px 16px", borderBottom: "1px solid #0f2035", display: "flex", alignItems: "center" }}>
            <span style={{ fontFamily: "'Syne', sans-serif", fontWeight: 800, color: "#d0e8f4", fontSize: 14 }}>Add New Node</span>
            <button onClick={onCancel} style={{ ...btnStyle, width: "auto", padding: "2px 8px", marginLeft: "auto", fontSize: 10 }}>✕</button>
          </div>
          <div style={{ padding: 16, overflowY: "auto", flex: 1 }}>
            {[
              ["label","Label"], ["sublabel","Sublabel"], ["group","Group"],
              ["layer","Layer","number"], ["ms","Latency ms","number"],
              ["metric","Metric"], ["shapeOut","Output Shape"], ["plain","Plain description","textarea"],
            ].map(([key, lbl, type = "text"]) => (
              <div key={key} style={{ marginBottom: 8 }}>
                <div style={{ fontSize: 7, letterSpacing: "1.5px", color: "#2a4a60", textTransform: "uppercase", marginBottom: 2 }}>{lbl}</div>
                {type === "textarea"
                  ? <textarea value={form[key] || ""} rows={2} onChange={e => setForm(f => ({ ...f, [key]: e.target.value }))} style={inputStyle} />
                  : <input type={type} value={form[key] !== undefined ? form[key] : ""} onChange={e => setForm(f => ({ ...f, [key]: type === "number" ? Number(e.target.value) : e.target.value }))} style={inputStyle} />
                }
              </div>
            ))}
            <div style={{ marginBottom: 8 }}>
              <div style={{ fontSize: 7, letterSpacing: "1.5px", color: "#2a4a60", textTransform: "uppercase", marginBottom: 2 }}>Color</div>
              <select value={form.color} onChange={e => setForm(f => ({ ...f, color: e.target.value }))} style={inputStyle}>
                {COLOR_OPTIONS.map(o => <option key={o.value} value={o.value} style={{ background: "#070d14" }}>{o.label}</option>)}
              </select>
            </div>
            <div style={{ marginBottom: 8 }}>
              <div style={{ fontSize: 7, letterSpacing: "1.5px", color: "#2a4a60", textTransform: "uppercase", marginBottom: 2 }}>Shape</div>
              <select value={form.shape} onChange={e => setForm(f => ({ ...f, shape: e.target.value }))} style={inputStyle}>
                <option value="rect" style={{ background: "#070d14" }}>Rectangle</option>
                <option value="hex" style={{ background: "#070d14" }}>Hexagon</option>
              </select>
            </div>
          </div>
          <div style={{ padding: "10px 16px", borderTop: "1px solid #0f2035", display: "flex", gap: 8 }}>
            <button onClick={() => onConfirm(form)} style={{ ...btnStyle, flex: 1, background: "rgba(0,255,170,0.08)", borderColor: "rgba(0,255,170,0.4)", color: "#00ffaa" }}>+ ADD NODE</button>
            <button onClick={onCancel} style={{ ...btnStyle, flex: 1 }}>CANCEL</button>
          </div>
        </div>
      </div>
    );
  }

  // ═══════════════════════════════════════════════════════════════
  //  LEGEND
  // ═══════════════════════════════════════════════════════════════
  function Legend() {
    const items = [
      { color: "#ffffff", label: "Dataset" },
      { color: C.pre,    label: "Preprocessing" },
      { color: C.xent,   label: "① Cross-Entropy" },
      { color: C.focal,  label: "② Focal Loss" },
      { color: C.dino,   label: "③ DINOv2 + ConvNeXt ★" },
      { color: C.seg,    label: "④ SegFormer" },
      { color: C.nav,    label: "Collision Detection" },
      { color: C.plan,   label: "A* Path Planning" },
      { color: C.result, label: "UGV Movement" },
    ];
    return (
      <div style={{
        position: "absolute", top: 12, left: 12,
        background: "rgba(7,13,20,0.92)", border: "1px solid #1a3550",
        borderRadius: 3, padding: "9px 11px", backdropFilter: "blur(8px)", zIndex: 10,
      }}>
        <div style={{ fontSize: 7, letterSpacing: "2px", color: "#2a4a60", textTransform: "uppercase", marginBottom: 5 }}>Node Types</div>
        {items.map(({ color, label }) => (
          <div key={label} style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 3 }}>
            <div style={{ width: 7, height: 7, borderRadius: "50%", background: color, flexShrink: 0, boxShadow: `0 0 4px ${color}88` }} />
            <span style={{ fontSize: 8, color: "#7a9cb0", fontFamily: "'JetBrains Mono', monospace" }}>{label}</span>
          </div>
        ))}
        <div style={{ marginTop: 5, paddingTop: 5, borderTop: "1px solid #0f2035", fontSize: 7, color: "#2a4a60", lineHeight: 1.6 }}>
          Click: select · Drag: move<br />★ = Best model
        </div>
      </div>
    );
  }

  // ═══════════════════════════════════════════════════════════════
  //  TOOLBAR
  // ═══════════════════════════════════════════════════════════════
  function Toolbar({ activeModel, onModelChange, onAddRoot, nodeCount }) {
    // receive optional video navigator prop via rest args
    const { onOpenVideos } = arguments[0] || {};
    const models = [
      { value: "all",       label: "◈ All Models" },
      { value: "xent",      label: "① Cross-Entropy" },
      { value: "focal",     label: "② Focal Loss" },
      { value: "dinov2",    label: "③ DINOv2 + ConvNeXt ★" },
      { value: "segformer", label: "④ SegFormer" },
    ];

    return (
      <div style={{
        height: 52, background: "#070d14", borderBottom: "1px solid #1a3550",
        display: "flex", alignItems: "center", padding: "0 16px", gap: 12, flexShrink: 0,
        position: "relative", overflow: "hidden",
      }}>
        {/* gradient underline */}
        <div style={{
          position: "absolute", bottom: 0, left: 0, right: 0, height: 1,
          background: "linear-gradient(90deg,transparent,#00cfff 20%,#a855f7 50%,#38bdf8 70%,#00ffaa 100%)",
          opacity: 0.45,
        }} />

        <div style={{ fontFamily: "'Syne', sans-serif", fontWeight: 800, fontSize: 15, color: "#d0e8f4", letterSpacing: 3, textTransform: "uppercase", whiteSpace: "nowrap" }}>
          UGV·<span style={{ color: "#00cfff" }}>AI</span> PIPELINE
        </div>

        {/* Model selector */}
        <div style={{ display: "flex", alignItems: "center", gap: 7, marginLeft: 8, background: "rgba(0,0,0,0.3)", border: "1px solid #1a3550", borderRadius: 3, padding: "4px 10px" }}>
          <span style={{ fontSize: 8, letterSpacing: "1.5px", color: "#2a4a60", textTransform: "uppercase", whiteSpace: "nowrap" }}>View Model</span>
          <select
            value={activeModel}
            onChange={e => onModelChange(e.target.value)}
            style={{
              background: "transparent", border: "none", color: "#00cfff",
              fontFamily: "'JetBrains Mono', monospace", fontSize: 10, fontWeight: 700,
              cursor: "pointer", outline: "none", padding: "0 18px 0 0",
            }}
          >
            {models.map(m => <option key={m.value} value={m.value} style={{ background: "#070d14" }}>{m.label}</option>)}
          </select>
        </div>

        {/* Node count */}
        <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-start", gap: 1, marginLeft: 8 }}>
          <span style={{ fontSize: 7, letterSpacing: "1.5px", color: "#2a4a60", textTransform: "uppercase" }}>Visible</span>
          <span style={{ fontSize: 13, fontWeight: 700, color: "#a855f7", fontFamily: "'JetBrains Mono', monospace" }}>{nodeCount}</span>
        </div>

        <div style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
          <button onClick={onAddRoot} style={{
            ...btnStyle, width: "auto", padding: "5px 14px",
            background: "rgba(0,255,170,0.08)", borderColor: "rgba(0,255,170,0.4)", color: "#00ffaa",
          }}>+ ADD NODE</button>
          <button onClick={() => onOpenVideos && onOpenVideos()} style={{
            ...btnStyle, width: "auto", padding: "5px 14px",
            background: "rgba(168,85,247,0.06)", borderColor: "rgba(168,85,247,0.28)", color: "#a855f7",
          }}>Videos</button>
        </div>
      </div>
    );
  }

  // ═══════════════════════════════════════════════════════════════
  //  MAIN APP
  // ═══════════════════════════════════════════════════════════════
  export default function UGVPipelineGraph() {
    const [nodes, setNodes] = useState(INITIAL_NODES);
    const [activeModel, setActiveModel] = useState("all");
    const [selectedId, setSelectedId] = useState(null);
    const [addingChildTo, setAddingChildTo] = useState(null);
    const [showAddModal, setShowAddModal] = useState(false);

    // Video player state
    const [videoSrc, setVideoSrc] = useState(null);
    // Videos section state
    const [showVideosSection, setShowVideosSection] = useState(false);
    const [folderVideos, setFolderVideos] = useState([]);


    // Pan / zoom
    const [pan, setPan] = useState({ x: 0, y: 0 });
    const [zoom, setZoom] = useState(0.9);
    const isPanning = useRef(false);
    const panStart = useRef({ x: 0, y: 0 });
    const panOrigin = useRef({ x: 0, y: 0 });

    // Drag node
    const draggingNode = useRef(null);
    const dragOffset = useRef({ x: 0, y: 0 });
    const [manualPositions, setManualPositions] = useState({});

    const svgRef = useRef(null);
    const containerRef = useRef(null);
    const [svgSize, setSvgSize] = useState({ w: 900, h: 700 });

    // Measure container
    useEffect(() => {
      const obs = new ResizeObserver(entries => {
        for (const e of entries) {
          setSvgSize({ w: e.contentRect.width, h: e.contentRect.height });
        }
      });
      if (containerRef.current) obs.observe(containerRef.current);
      return () => obs.disconnect();
    }, []);

    // Computed layout
    const autoPositions = useMemo(() => {
      return computeLayout(nodes, activeModel, svgSize.w / zoom, svgSize.h / zoom);
    }, [nodes, activeModel, svgSize, zoom]);

    const positions = useMemo(() => {
      const merged = { ...autoPositions };
      Object.entries(manualPositions).forEach(([id, pos]) => { if (merged[id]) merged[id] = pos; });
      return merged;
    }, [autoPositions, manualPositions]);

    const visibleIds = useMemo(() => getVisibleIds(nodes, activeModel), [nodes, activeModel]);
    const visibleNodes = useMemo(() => nodes.filter(n => visibleIds.includes(n.id)), [nodes, visibleIds]);

    const nodesWithVideo = useMemo(() => nodes.filter(n => n.metricVideo || n.metricVideoUrl), [nodes]);

    // load videos from public/videos and backend /result/videos (history)
    const loadFolderVideos = useCallback(async () => {
      const collected = [];

      try {
        const idx = await fetch("/videos/index.json");
        if (idx.ok) {
            const list = await idx.json();
            // expect array of strings or objects {url, title}
            const vids = list.map(item => {
              if (typeof item === "string") {
                const title = item.split("/").pop() || item;
                return {
                  url: normalizeVideoUrl(item),
                  title,
                  previewUrl: toConvertedPreviewUrl(title),
                };
              }
              // if object, ensure url is safe
              const title = item.title ?? item.url;
              return {
                url: normalizeVideoUrl(item.url),
                title,
                previewUrl: toConvertedPreviewUrl(title),
              };
            });
            collected.push(...vids);
          }
      } catch (err) {
        // ignore and try HTML fallback
      }

      try {
        const res = await fetch("/videos/");
        if (res.ok) {
          const txt = await res.text();
          // crude parse for links ending with video extensions
          const hrefs = Array.from(txt.matchAll(/href\s*=\s*\"([^\"]+)\"/g)).map(m => m[1]);
          const vids = hrefs.filter(h => /\.(mp4|webm|ogg)$/i.test(h)).map(h => {
            const name = h.split("/").pop();
            const url = h.startsWith("/") ? normalizeVideoUrl(h) : normalizeVideoUrl(name);
            return { url, title: name, previewUrl: toConvertedPreviewUrl(name) };
          });
          collected.push(...vids);
        }
      } catch (err) {
        // ignore
      }

      try {
        const resultRes = await fetch("http://localhost:8000/result/videos", { cache: "no-store" });
        if (resultRes.ok) {
          const payload = await resultRes.json();
          const backendVideos = Array.isArray(payload?.videos) ? payload.videos : [];
          const vids = backendVideos.map(item => {
            const title = item.filename || item.url || "processed_video.mp4";
            return {
              url: toBackendUrl(item.url || `/results/${item.filename}`),
              title,
              previewUrl: "",
            };
          });
          collected.push(...vids);
        }
      } catch (err) {
        // backend unavailable; keep public videos only
      }

      const deduped = [];
      const seen = new Set();
      for (const video of collected) {
        const key = String(video.url || "");
        if (!key || seen.has(key)) continue;
        seen.add(key);
        deduped.push(video);
      }
      setFolderVideos(deduped);
    }, []);

    // Helper: attempt to validate a video URL (HEAD) and return boolean
    const validateVideoUrl = useCallback(async url => {
      try {
        const res = await fetch(url, { method: "HEAD" });
        return res.ok;
      } catch (err) {
        return false;
      }
    }, []);

    // Try fallback strategies when a video element errors: try decoded filenames or raw title
    const tryVideoFallbacks = useCallback(async (videoEl, v) => {
      try {
        const current = videoEl && (videoEl.currentSrc || videoEl.src);
        console.error("Video failed to load:", current || v.url || v.title);

        const candidates = [];
        if (v?.previewUrl) candidates.push(v.previewUrl);
        if (v?.url) candidates.push(v.url);
        if (v?.title) candidates.push(v.title);

        // 1) try decodeURIComponent on the filename portion
        const parts = (v.url || v).split("/");
        const rawName = parts[parts.length - 1];
        let decoded;
        try { decoded = decodeURIComponent(rawName); } catch (e) { decoded = rawName; }
        if (decoded && decoded !== rawName) {
          candidates.push(decoded);
        }

        // 2) try using the title field or rawName without encoding
        if (v.title && v.title !== rawName) {
          candidates.push(v.title);
        }

        // 3) as last resort try the un-encoded rawName path
        candidates.push(rawName);
        candidates.push(toConvertedPreviewUrl(rawName));

        const normalizedCandidates = [];
        for (const item of candidates) {
          const str = String(item || "").trim();
          if (!str) continue;
          normalizedCandidates.push(str);
          normalizedCandidates.push(normalizeVideoUrl(str));
          normalizedCandidates.push(toBackendUrl(str));
          normalizedCandidates.push(toConvertedPreviewUrl(str));
        }

        const unique = [];
        const seen = new Set();
        for (const url of normalizedCandidates) {
          if (!url || seen.has(url)) continue;
          seen.add(url);
          unique.push(url);
        }

        for (const tryUrl of unique) {
          if (await validateVideoUrl(tryUrl)) {
            videoEl.src = tryUrl;
            videoEl.load();
            videoEl.play().catch(() => {});
            return;
          }
        }

        console.error("All fallback attempts failed for video:", v);
      } catch (err) {
        console.error("Fallback handler error", err);
      }
    }, [validateVideoUrl]);

    // Build edges
    const edges = useMemo(() => {
      const result = [];
      visibleNodes.forEach(n => {
        if (n.id === "collision") return;
        if (n.parentId && positions[n.parentId] && positions[n.id]) {
          const parentNode = nodes.find(p => p.id === n.parentId);
          result.push({ id: `${n.parentId}-${n.id}`, from: positions[n.parentId], to: positions[n.id], color: n.color });
        }
      });
      // collision edges from model outputs
      const collNode = nodes.find(n => n.id === "collision");
      if (collNode && positions["collision"]) {
        const activeModels = activeModel === "all" ? MODEL_GROUPS : [activeModel];
        activeModels.forEach(m => {
          const outId = MODEL_OUTPUTS[m];
          if (positions[outId]) {
            const outNode = nodes.find(n => n.id === outId);
            result.push({ id: `${outId}-collision`, from: positions[outId], to: positions["collision"], color: outNode?.color || C.nav });
          }
        });
      }
      return result;
    }, [visibleNodes, positions, activeModel, nodes]);

    // ── EVENT HANDLERS ──────────────────────────────────────────
    const handleSvgMouseDown = useCallback(e => {
      const isMiddleButton = e.button === 1;
      const isShiftPan = e.button === 0 && e.shiftKey;
      // treat clicks on the SVG or the background rect as canvas clicks
      const tag = e.target && e.target.tagName && e.target.tagName.toLowerCase();
      const clickedCanvas = e.target === svgRef.current || tag === "svg" || tag === "rect";

      if (isMiddleButton || isShiftPan || clickedCanvas) {
        e.preventDefault();
        isPanning.current = true;
        panStart.current = { x: e.clientX, y: e.clientY };
        panOrigin.current = { ...pan };
        if (clickedCanvas) setSelectedId(null);
      }
    }, [pan]);

    const handleMouseMove = useCallback(e => {
      if (draggingNode.current) {
        const rect = svgRef.current.getBoundingClientRect();
        const wx = (e.clientX - rect.left - pan.x) / zoom;
        const wy = (e.clientY - rect.top - pan.y) / zoom;
        setManualPositions(prev => ({
          ...prev,
          [draggingNode.current]: { x: wx - dragOffset.current.x, y: wy - dragOffset.current.y },
        }));
        return;
      }
      if (isPanning.current) {
        setPan({ x: panOrigin.current.x + (e.clientX - panStart.current.x), y: panOrigin.current.y + (e.clientY - panStart.current.y) });
      }
    }, [pan, zoom]);

    const handleMouseUp = useCallback(() => {
      isPanning.current = false;
      draggingNode.current = null;
    }, []);

    const handleNodeDragStart = useCallback((e, id) => {
      if (e.button !== 0 || e.shiftKey) return;
      const rect = svgRef.current.getBoundingClientRect();
      const wx = (e.clientX - rect.left - pan.x) / zoom;
      const wy = (e.clientY - rect.top - pan.y) / zoom;
      const pos = positions[id];
      if (pos) {
        dragOffset.current = { x: wx - pos.x, y: wy - pos.y };
        draggingNode.current = id;
      }
    }, [pan, zoom, positions]);

    const handleWheel = useCallback(e => {
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.1 : 0.91;
      const rect = svgRef.current.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      setPan(p => ({ x: mx - (mx - p.x) * factor, y: my - (my - p.y) * factor }));
      setZoom(z => Math.max(0.15, Math.min(4, z * factor)));
    }, []);

    useEffect(() => {
      const el = svgRef.current;
      if (!el) return;
      el.addEventListener("wheel", handleWheel, { passive: false });
      return () => el.removeEventListener("wheel", handleWheel);
    }, [handleWheel]);

    // ── NODE CRUD ───────────────────────────────────────────────
    const handleUpdate = useCallback(updated => {
      setNodes(ns => ns.map(n => n.id === updated.id ? updated : n));
    }, []);

    const handleDelete = useCallback(id => {
      setNodes(ns => {
        // reparent children to deleted node's parent
        const del = ns.find(n => n.id === id);
        return ns
          .map(n => n.parentId === id ? { ...n, parentId: del?.parentId ?? null } : n)
          .filter(n => n.id !== id);
      });
      setSelectedId(null);
    }, []);

    const handleAddChild = useCallback(parentId => {
      setAddingChildTo(parentId);
      setShowAddModal(true);
    }, []);

    const handleAddRoot = useCallback(() => {
      setAddingChildTo(null);
      setShowAddModal(true);
    }, []);

    const handleConfirmAdd = useCallback(form => {
      const newId = "node_" + Date.now();
      const parentNode = nodes.find(n => n.id === addingChildTo);
      const newNode = {
        ...form,
        id: newId,
        parentId: addingChildTo ?? null,
        layer: form.layer ?? (parentNode ? parentNode.layer + 1 : 0),
        label: form.label.replace(/\\n/g, "\n"),
      };
      setNodes(ns => [...ns, newNode]);
      setShowAddModal(false);
      setAddingChildTo(null);
      setSelectedId(newId);
    }, [nodes, addingChildTo]);

    const selectedNode = nodes.find(n => n.id === selectedId);

    // Collision zone position
    const collPos = positions["collision"];

    useEffect(() => {
      // pre-load videos when section is opened
      if (showVideosSection) loadFolderVideos();
    }, [showVideosSection, loadFolderVideos]);

    return (
      <div style={{ width: "100%", height: "100%", background: "#030508", display: "flex", flexDirection: "column", fontFamily: "'JetBrains Mono', monospace", overflow: "hidden" }}>
        <Toolbar
          activeModel={activeModel}
          onModelChange={m => { setActiveModel(m); setSelectedId(null); }}
          onAddRoot={handleAddRoot}
          nodeCount={visibleNodes.length}
          onOpenVideos={() => setShowVideosSection(s => !s)}
        />

        <div ref={containerRef} style={{ flex: 1, position: "relative", overflow: "hidden" }}>
          {/* SVG canvas */}
          <svg
            ref={svgRef}
            width={svgSize.w} height={svgSize.h}
            style={{ position: "absolute", inset: 0, cursor: isPanning.current ? "grabbing" : "grab" }}
            onContextMenu={e => e.preventDefault()}
            onMouseDown={handleSvgMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
          >
            {/* Grid */}
            <defs>
              <pattern id="grid" width={40 * zoom} height={40 * zoom} patternUnits="userSpaceOnUse"
                patternTransform={`translate(${pan.x % (40 * zoom)},${pan.y % (40 * zoom)})`}>
                <path d={`M ${40 * zoom} 0 L 0 0 0 ${40 * zoom}`} fill="none" stroke="rgba(15,32,53,0.5)" strokeWidth="0.5" />
              </pattern>
            </defs>
            <rect width={svgSize.w} height={svgSize.h} fill="url(#grid)" />

            <g transform={`translate(${pan.x},${pan.y}) scale(${zoom})`}>
              {/* Convergence zone */}
              {collPos && (
                <g>
                  <rect
                    x={collPos.x - 80} y={collPos.y - 50} width={160} height={250} rx={8}
                    fill="rgba(34,197,94,0.025)" stroke="rgba(34,197,94,0.12)"
                    strokeWidth={1} strokeDasharray="5,4"
                  />
                  <text x={collPos.x} y={collPos.y - 58} textAnchor="middle"
                    fontSize={7} fontWeight={600} fontFamily="'JetBrains Mono', monospace"
                    fill="rgba(34,197,94,0.4)">▼ CONVERGENCE ▼</text>
                </g>
              )}

              {/* Edges */}
              <g>
                {edges.map(e => (
                  <EdgeSVG key={e.id} from={e.from} to={e.to} color={e.color} />
                ))}
              </g>

              {/* Nodes */}
              {visibleNodes.map(n => {
                const pos = positions[n.id];
                if (!pos) return null;
                return (
                  <PipelineNode
                    key={n.id}
                    node={n}
                    pos={pos}
                    selected={selectedId === n.id}
                    onSelect={setSelectedId}
                    onDragStart={handleNodeDragStart}
                    isWinner={n.group === "dinov2" && n.layer === 2}
                  />
                );
              })}
            </g>
          </svg>

          {/* Legend */}
          <Legend />

          {/* Zoom controls */}
          <div style={{ position: "absolute", bottom: 16, right: selectedNode ? 316 : 16, display: "flex", flexDirection: "column", gap: 4 }}>
            {[
              { label: "+", fn: () => setZoom(z => Math.min(4, z * 1.2)) },
              { label: "−", fn: () => setZoom(z => Math.max(0.15, z * 0.83)) },
              { label: "⊡", fn: () => { setPan({ x: 0, y: 0 }); setZoom(0.9); } },
              { label: "↺", fn: () => { setManualPositions({}); } },
            ].map(({ label, fn }) => (
              <button key={label} onClick={fn} style={{
                width: 28, height: 28, background: "#070d14", border: "1px solid #1a3550",
                color: "#7a9cb0", fontSize: 14, cursor: "pointer", borderRadius: 2,
                display: "flex", alignItems: "center", justifyContent: "center",
              }}>{label}</button>
            ))}
          </div>

          {/* Details Panel (read-only) */}
          {selectedNode && (
            <DetailsPanel
              node={selectedNode}
              onClose={() => setSelectedId(null)}
            />
          )}

          {/* Videos Section (folder listing) */}
          {showVideosSection && (
            <VideosSection
              videos={folderVideos}
              onPlay={src => setVideoSrc(src)}
              onClose={() => setShowVideosSection(false)}
            />
          )}

          {/* Add Node Modal */}
          {showAddModal && (
            <AddNodeModal
            style={{ marginRight:'10px'}} 
              parentId={addingChildTo}
              onConfirm={handleConfirmAdd}
              onCancel={() => { setShowAddModal(false); setAddingChildTo(null); }}
            />
          )}

          {/* Videos tray + modal */}
          <VideosTray nodesWithVideo={nodesWithVideo.filter(n => visibleIds.includes(n.id))} onPlay={src => setVideoSrc(src)} />
          <VideoModal src={videoSrc} onClose={() => setVideoSrc(null)} />
        </div>
      </div>
    );
  }
