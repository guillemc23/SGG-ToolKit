# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "networkx",
#     "numpy",
#     "pandas",
#     "pillow",
#     "plotly>=5.18",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import json
    from pathlib import Path

    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from matplotlib.colors import to_hex
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None  # STAR images are huge satellite tiles
    return Image, Path, go, json, mo, np, nx, pd, plt, to_hex


@app.cell
def _(mo):
    mo.md(r"""
    # STAR dataset overview

    STAR is a large-scale scene-graph-generation dataset for **very high resolution satellite imagery**.
    Every image has oriented-bounding-box (OBB) object annotations (`STAR-object`) and
    object-level relationships (`STAR-relationship`), which together form a scene graph:

    - **nodes** = oriented boxes with a category (ship, dock, bridge, ...)
    - **edges** = `(subject, relationship, object)` triplets, e.g. *ship – parallelly docked at – dock*
    """)
    return


@app.cell
def _(mo):
    root_input = mo.ui.text(
        value="/Volumes/PortableSSD/star/STAR/STAR", label="Dataset root", full_width=True
    )
    root_input
    return (root_input,)


@app.cell
def _(Path, root_input):
    root = Path(root_input.value)

    def find_dir(parent, prefix):
        # folder names carry a trailing non-ASCII suffix, so match by prefix
        hits = sorted(p for p in parent.iterdir() if p.is_dir() and p.name.startswith(prefix))
        return hits[0] if hits else None

    # split -> dict of directories (None where a split has no such folder)
    splits = {}
    for split in ["train", "val", "test"]:
        obj, rel = root / "STAR-object" / split, root / "STAR-relationship" / split
        splits[split] = {
            "images": find_dir(obj, f"{split}img"),
            "json": find_dir(obj, "json"),
            "labelTxt": find_dir(obj, "labelTxt"),
            "csv": find_dir(rel, "csv") if rel.exists() else None,
        }
    return (splits,)


@app.cell
def _(pd, splits):
    _rows = []
    for _s, _d in splits.items():
        _rows.append(
            {
                "split": _s,
                "images": len(list(_d["images"].glob("*.png"))) if _d["images"] else 0,
                "object labels": len(list(_d["labelTxt"].glob("*.txt"))) if _d["labelTxt"] else 0,
                "relationship csvs": len(list(_d["csv"].glob("*.csv"))) if _d["csv"] else 0,
            }
        )
    split_summary = pd.DataFrame(_rows)
    split_summary
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Object & relationship statistics
    Computed from the `labelTxt` (DOTA-style OBB: `x1 y1 ... x4 y4 category difficulty`) and the relationship CSVs.
    """)
    return


@app.cell
def _(pd, splits):
    def load_objects(split):
        d = splits[split]["labelTxt"]
        recs = []
        for f in sorted(d.glob("*.txt")):
            for line in f.read_text().splitlines():
                p = line.split()
                if len(p) >= 10 and not p[0].startswith(("imagesource", "gsd")):
                    recs.append({"image": f.stem, "category": p[8], "difficulty": int(p[9])})
        return pd.DataFrame(recs)

    def load_relations(split):
        d = splits[split]["csv"]
        frames = []
        for f in sorted(d.glob("*.csv")):
            df = pd.read_csv(f)
            df["image"] = f.stem
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    objects = pd.concat(
        [load_objects(s).assign(split=s) for s in ["train", "val"]], ignore_index=True
    )
    relations = pd.concat(
        [load_relations(s).assign(split=s) for s in ["train", "val"] if splits[s]["csv"]],
        ignore_index=True,
    )
    return objects, relations


@app.cell
def _(objects, plt):
    _counts = objects.groupby(["category", "split"]).size().unstack(fill_value=0)
    _counts = _counts.loc[_counts.sum(axis=1).sort_values().index]
    _ax = _counts.plot.barh(stacked=True, figsize=(8, 6), logx=True)
    _ax.set_title(f"Object instances per category ({len(objects):,} total, train+val)")
    _ax.set_xlabel("instances (log)")
    plt.tight_layout()
    _ax
    return


@app.cell
def _(plt, relations):
    _counts = relations.groupby(["relationship", "split"]).size().unstack(fill_value=0)
    _counts = _counts.loc[_counts.sum(axis=1).sort_values().index]
    _ax = _counts.plot.barh(stacked=True, figsize=(8, 9), logx=True)
    _ax.set_title(f"Relationship predicates ({len(relations):,} triplets, train+val)")
    _ax.set_xlabel("triplets (log)")
    plt.tight_layout()
    _ax
    return


@app.cell
def _(objects, plt, relations):
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.5))
    objects.groupby("image").size().hist(bins=40, ax=_axes[0])
    _axes[0].set(title="Objects per image", xlabel="# objects", ylabel="# images")
    relations.groupby("image").size().hist(bins=40, ax=_axes[1], color="tab:orange")
    _axes[1].set(title="Relationships per image", xlabel="# triplets")
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo, relations):
    _top = (
        relations.groupby(["subject", "relationship", "object"])
        .size()
        .sort_values(ascending=False)
        .rename("count")
        .reset_index()
    )
    mo.vstack([mo.md("### Most frequent (subject, predicate, object) triplets"), mo.ui.table(_top, page_size=15)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Sample images and their scene graphs
    Pick a split and sample. Left: image with oriented boxes. Middle: the scene graph
    overlaid on the image (nodes at box centres). Right: the graph with a force layout.
    """)
    return


@app.cell
def _(mo, splits):
    split_dd = mo.ui.dropdown(["train", "val"], value="train", label="Split")
    n_slider = mo.ui.slider(0, 770, 1, value=0, label="Sample index", full_width=True)
    mo.vstack([split_dd, n_slider])
    return n_slider, split_dd


@app.cell
def _(Image, json, n_slider, np, pd, split_dd, splits):
    d = splits[split_dd.value]
    stems = sorted(p.stem for p in d["csv"].glob("*.csv"))
    stem = stems[min(n_slider.value, len(stems) - 1)]

    img_full = Image.open(next(d["images"].glob(f"{stem}.*")))
    W, H = img_full.size
    scale = 1600 / max(W, H)
    img_small = img_full.convert("RGB").resize((int(W * scale), int(H * scale)))

    _meta = json.loads((d["json"] / f"{stem}.json").read_text())
    shapes = [
        {
            "label": s["label"],
            "gid": s["group_id"],
            "pts": np.array(s["points"]) * scale,
        }
        for s in _meta["shapes"]
    ]
    rels = pd.read_csv(d["csv"] / f"{stem}.csv")
    return H, W, img_small, rels, shapes, stem


@app.cell
def _(H, W, img_small, mo, rels, shapes, stem):
    mo.md(
        f"**Image `{stem}`** — {W}×{H} px, {len(shapes)} objects, {len(rels)} relationships"
    )
    return


@app.cell
def _(img_small, nx, np, plt, rels, shapes):
    # --- build graph: node key = (category, group_id) ---
    cats = sorted({s["label"] for s in shapes})
    _cmap = plt.get_cmap("tab20")
    cat_color = {c: _cmap(i % 20) for i, c in enumerate(cats)}

    G = nx.MultiDiGraph()
    pos = {}
    for s in shapes:
        key = (s["label"], s["gid"])
        G.add_node(key, label=s["label"])
        pos[key] = s["pts"].mean(axis=0)

    for r in rels.itertuples():
        a, b = (r.subject, r.subject_ID), (r.object, r.object_ID)
        if a in G and b in G:
            G.add_edge(a, b, rel=r.relationship)

    preds = sorted(rels["relationship"].unique())
    _ecmap = plt.get_cmap("tab10")
    pred_color = {p: _ecmap(i % 10) for i, p in enumerate(preds)}

    # connected nodes only for the graph panels
    connected = [n for n in G.nodes if G.degree(n) > 0]
    Gc = G.subgraph(connected)

    fig, axes = plt.subplots(1, 3, figsize=(22, 7.5))

    # (1) image + oriented boxes
    axes[0].imshow(img_small)
    for s in shapes:
        poly = np.vstack([s["pts"], s["pts"][:1]])
        axes[0].plot(poly[:, 0], poly[:, 1], color=cat_color[s["label"]], lw=1.2)
    axes[0].set_title("Oriented bounding boxes")

    # (2) graph overlaid on image
    axes[1].imshow(img_small, alpha=0.8)
    for u, v, data in Gc.edges(data=True):
        axes[1].annotate(
            "", xy=pos[v], xytext=pos[u],
            arrowprops=dict(arrowstyle="->", color=pred_color[data["rel"]], lw=1.3, alpha=0.9,
                            shrinkA=3, shrinkB=3),
        )
    axes[1].scatter(
        [pos[n][0] for n in connected], [pos[n][1] for n in connected],
        c=[cat_color[n[0]] for n in connected], s=45, edgecolors="white", linewidths=0.7, zorder=3,
    )
    axes[1].set_title("Scene graph overlaid on image")

    # (3) abstract graph
    for _ax in axes[:2]:
        _ax.axis("off")
    if len(Gc):
        gpos = nx.spring_layout(nx.Graph(Gc), seed=0, k=1.2 / max(1, np.sqrt(len(Gc))))
        nx.draw_networkx_nodes(Gc, gpos, ax=axes[2], node_color=[cat_color[n[0]] for n in Gc.nodes],
                               node_size=140, edgecolors="k", linewidths=0.5)
        nx.draw_networkx_labels(Gc, gpos, ax=axes[2], labels={n: f"{n[0][:3]}{n[1]}" for n in Gc.nodes}, font_size=6)
        for u, v, data in Gc.edges(data=True):
            nx.draw_networkx_edges(Gc, gpos, ax=axes[2], edgelist=[(u, v)],
                                   edge_color=[pred_color[data["rel"]]], arrows=True, alpha=0.8,
                                   connectionstyle="arc3,rad=0.1", node_size=140)
    axes[2].set_title(f"Graph ({len(Gc)} connected nodes, {Gc.number_of_edges()} edges)")
    axes[2].axis("off")

    # shared legends
    from matplotlib.lines import Line2D
    _handles = [Line2D([], [], marker="o", ls="", color=cat_color[c], label=c) for c in cats] + [
        Line2D([], [], color=pred_color[p], lw=2, label=p) for p in preds
    ]
    fig.legend(handles=_handles, loc="lower center", ncol=min(8, len(_handles)), fontsize=9, frameon=False)
    plt.tight_layout(rect=(0, 0.07, 1, 1))
    fig
    return Gc, cat_color, cats, connected, pos, pred_color, preds


@app.cell
def _(mo):
    mo.md(r"""
    ### Interactive graph
    Zoom, pan and hover. Hover a node for its category/ID/degree, and the arrow at the middle
    of each edge for `subject –predicate→ object`. Click legend entries to toggle categories.
    """)
    return


@app.cell
def _(mo, preds):
    layout_dd = mo.ui.dropdown(["On image", "Force layout"], value="On image", label="Layout")
    pred_sel = mo.ui.multiselect(preds, value=preds, label="Predicates shown", full_width=True)
    mo.vstack([layout_dd, pred_sel])
    return layout_dd, pred_sel


@app.cell
def _(
    Gc,
    cat_color,
    cats,
    connected,
    go,
    img_small,
    layout_dd,
    np,
    nx,
    pos,
    pred_color,
    pred_sel,
    stem,
    to_hex,
):
    on_image = layout_dd.value == "On image"
    if on_image:
        xy = {n: np.asarray(pos[n]) for n in Gc.nodes}
    else:
        xy = {n: np.asarray(p) for n, p in nx.spring_layout(nx.Graph(Gc), seed=0, k=1.2 / max(1, np.sqrt(len(Gc)))).items()}

    ifig = go.Figure()

    # edges: one line trace + one arrow-marker trace (carries the hover text) per predicate
    for pred in pred_sel.value:
        lx, ly, mx, my, ang, txt = [], [], [], [], [], []
        for src, dst, edata in Gc.edges(data=True):
            if edata["rel"] != pred:
                continue
            (x0, y0), (x1, y1) = xy[src], xy[dst]
            lx += [x0, x1, None]
            ly += [y0, y1, None]
            mx.append((x0 + x1) / 2)
            my.append((y0 + y1) / 2)
            sy_up = -(y1 - y0) if on_image else (y1 - y0)  # image y axis is reversed
            ang.append(float(np.degrees(np.arctan2(x1 - x0, sy_up))))
            txt.append(f"{src[0]} {src[1]} –{pred}→ {dst[0]} {dst[1]}")
        col = to_hex(pred_color[pred])
        ifig.add_trace(go.Scatter(x=lx, y=ly, mode="lines", line=dict(color=col, width=1.5),
                                  opacity=0.8, hoverinfo="skip", legendgroup=pred, showlegend=False))
        ifig.add_trace(go.Scatter(x=mx, y=my, mode="markers", name=pred, legendgroup=pred,
                                  marker=dict(symbol="arrow", size=10, color=col, angle=ang, angleref="up",
                                              line=dict(width=0.5, color="white")),
                                  text=txt, hoverinfo="text"))

    # nodes: one trace per category
    for cat in cats:
        nodes = [n for n in connected if n[0] == cat]
        if not nodes:
            continue
        ifig.add_trace(go.Scatter(
            x=[xy[n][0] for n in nodes], y=[xy[n][1] for n in nodes], mode="markers", name=cat,
            marker=dict(size=9, color=to_hex(cat_color[cat]), line=dict(width=1, color="white")),
            text=[f"<b>{n[0]} {n[1]}</b><br>in: {Gc.in_degree(n)} · out: {Gc.out_degree(n)}" for n in nodes],
            hoverinfo="text",
        ))

    ifig.update_layout(height=800, title=f"Image {stem}: scene graph", legend=dict(itemsizing="constant"),
                       margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor="#111")
    ifig.update_xaxes(visible=False)
    if on_image:
        _w, _h = img_small.size
        ifig.add_layout_image(dict(source=img_small, xref="x", yref="y", x=0, y=0, sizex=_w, sizey=_h,
                                   xanchor="left", yanchor="top", sizing="stretch", layer="below"))
        ifig.update_xaxes(range=[0, _w], constrain="domain")
        ifig.update_yaxes(range=[_h, 0], scaleanchor="x", visible=False)
    else:
        ifig.update_yaxes(visible=False, scaleanchor="x")
    ifig
    return


@app.cell
def _(mo, rels):
    mo.vstack([mo.md("### Relationship triplets for this image"), mo.ui.table(rels, page_size=10)])
    return


if __name__ == "__main__":
    app.run()
