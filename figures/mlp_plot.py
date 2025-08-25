#!/usr/bin/env python3
from __future__ import annotations
import argparse
import numpy as np
import math
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def make_data(n_samples: int, noise: float, seed: int = 42, centers: int = 4, dataset: str = "spiral"):
    if dataset == "spiral":
        rng = np.random.default_rng(seed)
        n_per = max(1, n_samples // max(1, centers))
        X = np.zeros((n_per*centers, 2), dtype=float)
        y = np.zeros(n_per*centers, dtype=int)
        idx = 0
        for i in range(centers):
            r = np.linspace(0.2, 1.0, n_per)
            t = np.linspace(i * 2 * math.pi / centers, (i + 2) * 2 * math.pi / centers, n_per) + rng.normal(0.0, noise, n_per)
            Xi = np.stack([r*np.sin(t), r*np.cos(t)], axis=1)
            X[idx:idx+n_per] = Xi
            y[idx:idx+n_per] = i
            idx += n_per
        if n_per*centers > n_samples:
            X = X[:n_samples]
            y = y[:n_samples]
    else:
        X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=noise, random_state=seed)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    return Xtr, Xte, ytr, yte

def train_mlp(Xtr, ytr, activation: str = "relu", seed: int = 42) -> MLPClassifier:
    clf = MLPClassifier(
        hidden_layer_sizes=(16, 16),
        activation=activation,
        solver="adam",
        alpha=1e-4,
        max_iter=5_000,
        random_state=seed,
    )
    clf.fit(Xtr, ytr)
    return clf

def _act(a, name):
    if name == 'identity':
        return a
    if name == 'relu':
        return np.maximum(a, 0.0)
    if name == 'tanh':
        return np.tanh(a)
    if name == 'logistic':
        return 1.0 / (1.0 + np.exp(-a))
    return a

def forward_all(clf: MLPClassifier, XY: np.ndarray):
    W1, W2, W3 = clf.coefs_
    b1, b2, b3 = clf.intercepts_
    h1_pre = XY @ W1 + b1
    act = getattr(clf, 'activation', 'relu')
    h1 = _act(h1_pre, act)
    h2_pre = h1 @ W2 + b2
    h2 = _act(h2_pre, act)
    logits = h2 @ W3 + b3
    s = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(s)
    probs = e / e.sum(axis=1, keepdims=True)
    return probs, logits, h2_pre, h2

def _domains_grid_2x2():
    return {
        'scene':  dict(domain=dict(x=[0.000, 0.497], y=[0.503, 1.000])),
        'scene2': dict(domain=dict(x=[0.503, 1.000], y=[0.503, 1.000])),
        'scene3': dict(domain=dict(x=[0.000, 0.497], y=[0.000, 0.497])),
        'scene4': dict(domain=dict(x=[0.503, 1.000], y=[0.000, 0.497])),
    }

def _colorbar_pos_for(scene_key: str):
    centers = {
        'scene':  dict(x=0.49, y=0.76),
        'scene2': dict(x=1.00, y=0.76),
        'scene3': dict(x=0.49, y=0.24),
        'scene4': dict(x=1.00, y=0.24),
    }
    return centers.get(scene_key, dict(x=1.02, y=0.5))

def build_figure(xx, yy, Z_prob, Z_logit, X, y, clf, title_extra: str = "", point_subsample: int = 500):
    colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
    point_colors = np.array([colors[int(label) % len(colors)] for label in y])
    if point_subsample is not None and len(X) > point_subsample:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=point_subsample, replace=False)
    else:
        idx = np.arange(len(X))
    Xp, yp = X[idx], y[idx]
    point_colors_sub = point_colors[idx]
    probs_pts, logits_pts, h2_pre_pts, h2_post_pts = forward_all(clf, Xp)
    probs_grid, logits_grid, h2_pre_grid, h2_post_grid = forward_all(clf, np.c_[xx.ravel(), yy.ravel()])
    n_classes = probs_grid.shape[1]
    prob_range = (0.0, 1.0)
    logit_range = (float(np.min(logits_grid)), float(np.max(logits_grid)))
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'surface'}],
               [{'type': 'surface'}, {'type': 'surface'}]],
        horizontal_spacing=0.02, vertical_spacing=0.02,
    )
    index = dict(
        prob_surf=[None]*4, prob_pts=[None]*4,
        logit_surf=[None]*4, logit_pts=[None]*4,
        pre_surf=[None]*4, pre_pts=[None]*4,
        post_surf=[None]*4, post_pts=[None]*4,
        plane_hidden=[None]*4,
        simplex_pts=None, simplex_tri=None, simplex_edges=None, simplex_labels=None, simplex_regions=[None]*3, cube_pts=None, cube_edges=None,
        dataset2d_plane=None, dataset2d_pts=None, dataset2d_grid=None,
    )
    scene_target = {0: 'scene', 1: 'scene2', 2: 'scene3', 3: 'scene4'}
    for c in range(min(4, n_classes)):
        r, ccol = (1 if c < 2 else 2), (1 if c % 2 == 0 else 2)
        scene_key = scene_target[c]
        cbpos = _colorbar_pos_for(scene_key)
        fig.add_trace(
            go.Surface(
                x=xx, y=yy, z=Z_prob[:, c].reshape(xx.shape),
                colorscale='Viridis', cmin=prob_range[0], cmax=prob_range[1],
                showscale=False,
                colorbar=dict(title=f"P(class={c})", x=cbpos['x'], y=cbpos['y'], len=0.42),
                opacity=0.95, name=f"P(class={c})", scene=scene_key
            ),
            row=r, col=ccol
        )
        index['prob_surf'][c] = len(fig.data)-1
        fig.add_trace(
            go.Scatter3d(x=Xp[:,0], y=Xp[:,1], z=probs_pts[:,c], mode="markers",
                         marker=dict(size=3, color=point_colors_sub), name=f"pts→P{c}", scene=scene_key),
            row=r, col=ccol
        )
        index['prob_pts'][c] = len(fig.data)-1
        fig.add_trace(
            go.Surface(
                x=xx, y=yy, z=Z_logit[:, c].reshape(xx.shape),
                colorscale='Viridis', cmin=logit_range[0], cmax=logit_range[1],
                showscale=False,
                opacity=0.95, name=f"logit(class={c})", scene=scene_key, visible=False
            ),
            row=r, col=ccol
        )
        index['logit_surf'][c] = len(fig.data)-1
        fig.add_trace(
            go.Scatter3d(x=Xp[:,0], y=Xp[:,1], z=logits_pts[:,c], mode="markers",
                         marker=dict(size=3, color=point_colors_sub), name=f"pts→logit{c}", scene=scene_key, visible=False),
            row=r, col=ccol
        )
        index['logit_pts'][c] = len(fig.data)-1
    for node in range(4):
        r, ccol = (1 if node < 2 else 2), (1 if node % 2 == 0 else 2)
        scene_key = scene_target[node]
        cbpos = _colorbar_pos_for(scene_key)
        fig.add_trace(
            go.Surface(x=xx, y=yy, z=h2_pre_grid[:, node].reshape(xx.shape),
                       colorscale='Viridis', showscale=False,
                       opacity=0.95, name=f"h2[{node}] pre", scene=scene_key, visible=False),
            row=r, col=ccol
        )
        index['pre_surf'][node] = len(fig.data)-1
        fig.add_trace(
            go.Scatter3d(x=Xp[:,0], y=Xp[:,1], z=h2_pre_pts[:, node], mode="markers",
                         marker=dict(size=3, color=point_colors_sub), name=f"pts→h2[{node}] pre", scene=scene_key, visible=False),
            row=r, col=ccol
        )
        index['pre_pts'][node] = len(fig.data)-1
        fig.add_trace(
            go.Surface(x=xx, y=yy, z=h2_post_grid[:, node].reshape(xx.shape),
                       colorscale='Viridis', showscale=False,
                       opacity=0.95, name=f"h2[{node}] post", scene=scene_key, visible=False),
            row=r, col=ccol
        )
        index['post_surf'][node] = len(fig.data)-1
        fig.add_trace(
            go.Scatter3d(x=Xp[:,0], y=Xp[:,1], z=h2_post_pts[:, node], mode="markers",
                         marker=dict(size=3, color=point_colors_sub), name=f"pts→h2[{node}] post", scene=scene_key, visible=False),
            row=r, col=ccol
        )
        index['post_pts'][node] = len(fig.data)-1
        z0 = np.zeros_like(xx)
        fig.add_trace(
            go.Surface(x=xx, y=yy, z=z0, showscale=False, opacity=0.2,
                       colorscale=[[0, "#AAAAAA"], [1, "#AAAAAA"]], name=f"z=0 (h2 scene {node})",
                       scene=scene_key, visible=False),
            row=r, col=ccol
        )
        index['plane_hidden'][node] = len(fig.data)-1
    probs_all, _, _, _ = forward_all(clf, X)
    P3 = probs_all[:, :3]
    denom = P3.sum(axis=1, keepdims=True)
    denom = np.clip(denom, 1e-12, None)
    P3n = P3 / denom
    tri_xyz = np.array([[1,0,0], [0,1,0], [0,0,1]])
    simplex_tri = go.Mesh3d(
        x=tri_xyz[:,0], y=tri_xyz[:,1], z=tri_xyz[:,2],
        i=[0], j=[1], k=[2],
        color='lightblue', opacity=0.15, name='simplex p0+p1+p2=1',
        scene='scene', visible=False, showscale=False
    )
    fig.add_trace(simplex_tri, row=1, col=1)
    index['simplex_tri'] = len(fig.data) - 1
    edges = [((1,0,0),(0,1,0)), ((1,0,0),(0,0,1)), ((0,1,0),(0,0,1))]
    xs, ys, zs = [], [], []
    for (x0,y0,z0),(x1,y1,z1) in edges:
        xs += [x0,x1,None]; ys += [y0,y1,None]; zs += [z0,z1,None]
    simplex_edges = go.Scatter3d(x=xs, y=ys, z=zs, mode='lines',
                                 line=dict(color='black', width=2), showlegend=False,
                                 name='simplex edges', scene='scene', visible=False)
    fig.add_trace(simplex_edges, row=1, col=1)
    index['simplex_edges'] = len(fig.data) - 1
    simplex_labels = go.Scatter3d(
        x=tri_xyz[:,0], y=tri_xyz[:,1], z=tri_xyz[:,2], mode='text',
        text=['p0','p1','p2'], textposition='top center',
        showlegend=False, scene='scene', visible=False
    )
    fig.add_trace(simplex_labels, row=1, col=1)
    index['simplex_labels'] = len(fig.data) - 1
    simplex_pts = go.Scatter3d(
        x=P3n[:,0], y=P3n[:,1], z=P3n[:,2], mode='markers',
        marker=dict(size=3, color=point_colors, opacity=0.9),
        name='probs simplex (p0,p1,p2)', scene='scene', visible=False
    )
    fig.add_trace(simplex_pts, row=1, col=1)
    index['simplex_pts'] = len(fig.data) - 1
    m01 = np.array([0.5,0.5,0.0]); m02 = np.array([0.5,0.0,0.5]); m12 = np.array([0.0,0.5,0.5])
    r0 = np.vstack([tri_xyz[0], m01, m02])
    r1 = np.vstack([tri_xyz[1], m01, m12])
    r2 = np.vstack([tri_xyz[2], m02, m12])
    reg_colors = ['red','blue','green']
    for ri, R in enumerate([r0, r1, r2]):
        reg = go.Mesh3d(x=R[:,0], y=R[:,1], z=R[:,2], i=[0], j=[1], k=[2],
                        color=reg_colors[ri], opacity=0.35, scene='scene', visible=False, name=f'region {ri}')
        fig.add_trace(reg, row=1, col=1)
        index['simplex_regions'][ri] = len(fig.data) - 1
    z0_ds = np.zeros_like(xx)
    ds_plane = go.Surface(x=xx, y=yy, z=z0_ds, showscale=False, opacity=0.18,
                          colorscale=[[0, "#D0D0D0"], [1, "#D0D0D0"]], scene='scene', visible=False,
                          name='dataset plane z=0')
    fig.add_trace(ds_plane, row=1, col=1)
    index['dataset2d_plane'] = len(fig.data) - 1
    ds_pts = go.Scatter3d(x=X[:,0], y=X[:,1], z=np.zeros(len(X)), mode='markers',
                          marker=dict(size=2, color=point_colors, opacity=0.9), scene='scene', visible=False,
                          name='dataset 2D')
    fig.add_trace(ds_pts, row=1, col=1)
    index['dataset2d_pts'] = len(fig.data) - 1
    labels_grid = np.argmax(Z_prob, axis=1)
    grid_colors = np.array([colors[int(k) % len(colors)] for k in labels_grid])
    ds_grid = go.Scatter3d(x=xx.ravel(), y=yy.ravel(), z=np.zeros(xx.size), mode='markers',
                           marker=dict(size=2, color=grid_colors, opacity=0.5), scene='scene', visible=False,
                           name='dataset grid preds')
    fig.add_trace(ds_grid, row=1, col=1)
    index['dataset2d_grid'] = len(fig.data) - 1
    def _vis_all_false(n):
        return [False] * n
    view_buttons = []
    vis = _vis_all_false(len(fig.data))
    for c in range(4):
        for key in ('prob_surf','prob_pts'):
            i = index[key][c]
            if i is not None:
                vis[i] = True
    view_buttons.append(dict(
        label="All probs (2×2)", method="update",
        args=[{"visible": vis}, {"title": "", "updatemenus[1].visible": False, "updatemenus[2].visible": False}],
    ))
    vis = _vis_all_false(len(fig.data))
    for c in range(4):
        for key in ('logit_surf','logit_pts'):
            i = index[key][c]
            if i is not None:
                vis[i] = True
    view_buttons.append(dict(
        label="All logits (2×2)", method="update",
        args=[{"visible": vis}, {"title": f"All logits (2×2){title_extra}", "updatemenus[1].visible": False, "updatemenus[2].visible": False}],
    ))
    vis = _vis_all_false(len(fig.data))
    for n in range(4):
        vis[index['pre_surf'][n]] = True
        vis[index['pre_pts'][n]] = True
        vis[index['plane_hidden'][n]] = True
    view_buttons.append(dict(
        label="Hidden pre (2×2)", method="update",
        args=[{"visible": vis}, {"title": f"Hidden layer pre-activations (2×2){title_extra}", "updatemenus[1].visible": True, "updatemenus[2].visible": False}],
    ))
    vis = _vis_all_false(len(fig.data))
    for n in range(4):
        vis[index['post_surf'][n]] = True
        vis[index['post_pts'][n]] = True
    view_buttons.append(dict(
        label="Hidden post (2×2)", method="update",
        args=[{"visible": vis}, {"title": f"Hidden layer post-activations (2×2){title_extra}", "updatemenus[1].visible": False, "updatemenus[2].visible": False}],
    ))
    vis = _vis_all_false(len(fig.data))
    for key in ('simplex_tri','simplex_edges','simplex_labels','simplex_pts'):
        i = index.get(key)
        if isinstance(i, int):
            vis[i] = True
    for i in index['simplex_regions']:
        if i is not None:
            vis[i] = True
    view_buttons.append(dict(
        label="Probs simplex (p0,p1,p2)", method="update",
        args=[{"visible": vis}, {"title": "Probabilities simplex (barycentric)", "updatemenus[1].visible": False, "updatemenus[2].visible": True}],
    ))
    vis = _vis_all_false(len(fig.data))
    if index['dataset2d_plane'] is not None:
        vis[index['dataset2d_plane']] = True
    if index['dataset2d_pts'] is not None:
        vis[index['dataset2d_pts']] = True
    view_buttons.append(dict(
        label="Dataset (2D)", method="update",
        args=[{"visible": vis}, {"title": "Original dataset (2D plane)", "updatemenus[1].visible": False, "updatemenus[2].visible": False}],
    ))
    if index['cube_pts'] is not None:
        vis[index['cube_pts']] = True
    if index['cube_edges'] is not None:
        vis[index['cube_edges']] = True
    view_buttons.append(dict(
        label="Probs in [0,1]^3", method="update",
        args=[{"visible": vis}, {"title": "Probabilities in [0,1]^3 (p0,p1,p2)", "updatemenus[1].visible": False}],
    ))
    for b in view_buttons:
        lab = b.get("label", "")
        if len(b.get("args", [])) >= 2 and isinstance(b["args"][1], dict):
            b["args"][1]["updatemenus[3].visible"] = (lab == "Dataset (2D)")
    plane_indices = [i for i in index['plane_hidden'] if i is not None]
    plane_buttons = [
        dict(label="z=0 plane", method="restyle",
             args=[{"visible": [True]*len(plane_indices)}, plane_indices],
             args2=[{"visible": [False]*len(plane_indices)}, plane_indices])
    ]
    region_indices = [i for i in index['simplex_regions'] if i is not None]
    region_buttons = [
        dict(label="regions", method="restyle",
             args=[{"visible": [True]*len(region_indices)}, region_indices],
             args2=[{"visible": [False]*len(region_indices)}, region_indices])
    ]
    dataset_indices = [index['dataset2d_pts'], index['dataset2d_grid']]
    vis_dataset_pts  = [False] * len(fig.data)
    vis_dataset_grid = [False] * len(fig.data)
    if index['dataset2d_plane'] is not None:
        vis_dataset_pts[index['dataset2d_plane']]  = True
        vis_dataset_grid[index['dataset2d_plane']] = True
    if index['dataset2d_pts'] is not None:
        vis_dataset_pts[index['dataset2d_pts']] = True
    if index['dataset2d_grid'] is not None:
        vis_dataset_grid[index['dataset2d_grid']] = True

    dataset_buttons = [
        dict(label="grid/points", method="update",
            args=[{"visible": vis_dataset_grid}, {}],
            args2=[{"visible": vis_dataset_pts},  {}])
    ]
    cam = {"eye": {"x": 1.6, "y": 1.6, "z": 1.1}, "up": {"x": 0, "y": 0, "z": 1}, "center": {"x": 0, "y": 0, "z": 0}}
    view_buttons = [b for b in view_buttons if b.get("label") != "Probs in [0,1]^3"]
    fig.update_layout(
        updatemenus=[
            dict(type="dropdown", direction="down",
                x=0.5, xanchor="center", y=1.08, yanchor="top",
                buttons=view_buttons),
            dict(type="buttons",  direction="right",
                x=0.0, xanchor="left",   y=1.08, yanchor="top",
                buttons=plane_buttons,   visible=False),
            dict(type="buttons",  direction="right",
                x=0.0, xanchor="left",   y=1.04, yanchor="top",
                buttons=region_buttons,  visible=False),
            dict(type="buttons",  direction="right",
                x=0.0, xanchor="left",   y=1.00, yanchor="top",
                buttons=dataset_buttons, visible=False),
        ]
    )

    fig.update_layout(
        title="",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="val", aspectmode="cube",
                   xaxis=dict(showbackground=True, backgroundcolor="rgb(235,245,255)", showgrid=True, gridcolor="#ffffff", gridwidth=1.2),
                   yaxis=dict(showbackground=True, backgroundcolor="rgb(235,245,255)", showgrid=True, gridcolor="#ffffff", gridwidth=1.2),
                   zaxis=dict(showbackground=True, backgroundcolor="rgb(235,245,255)", showgrid=True, gridcolor="#ffffff", gridwidth=1.2),
                   camera=cam),
        scene2=dict(xaxis_title="x", yaxis_title="y", zaxis_title="val", aspectmode="cube",
                    xaxis=dict(showbackground=True, backgroundcolor="rgb(235,245,255)", showgrid=True, gridcolor="#ffffff", gridwidth=1.2),
                    yaxis=dict(showbackground=True, backgroundcolor="rgb(235,245,255)", showgrid=True, gridcolor="#ffffff", gridwidth=1.2),
                    zaxis=dict(showbackground=True, backgroundcolor="rgb(235,245,255)", showgrid=True, gridcolor="#ffffff", gridwidth=1.2),
                    camera=cam),
        scene3=dict(xaxis_title="x", yaxis_title="y", zaxis_title="val", aspectmode="cube",
                    xaxis=dict(showbackground=True, backgroundcolor="rgb(235,245,255)", showgrid=True, gridcolor="#ffffff", gridwidth=1.2),
                    yaxis=dict(showbackground=True, backgroundcolor="rgb(235,245,255)", showgrid=True, gridcolor="#ffffff", gridwidth=1.2),
                    zaxis=dict(showbackground=True, backgroundcolor="rgb(235,245,255)", showgrid=True, gridcolor="#ffffff", gridwidth=1.2),
                    camera=cam),
        scene4=dict(xaxis_title="x", yaxis_title="y", zaxis_title="val", aspectmode="cube",
                    xaxis=dict(showbackground=True, backgroundcolor="rgb(235,245,255)", showgrid=True, gridcolor="#ffffff", gridwidth=1.2),
                    yaxis=dict(showbackground=True, backgroundcolor="rgb(235,245,255)", showgrid=True, gridcolor="#ffffff", gridwidth=1.2),
                    zaxis=dict(showbackground=True, backgroundcolor="rgb(235,245,255)", showgrid=True, gridcolor="#ffffff", gridwidth=1.2),
                    camera=cam),
        width=1200, height=800, margin=dict(l=4, r=4, t=28, b=4), showlegend=False,
        updatemenus=[
        dict(type="dropdown", direction="down", x=0.5, xanchor="center", y=1.08, yanchor="top", buttons=view_buttons),
        dict(type="buttons", direction="right", x=0.0, xanchor="left", y=1.08, yanchor="top", buttons=plane_buttons,   visible=False),
        dict(type="buttons", direction="right", x=0.0, xanchor="left", y=1.04, yanchor="top", buttons=region_buttons,  visible=False),
        dict(type="buttons", direction="right", x=0.0, xanchor="left", y=1.00, yanchor="top", buttons=dataset_buttons, visible=False),
    ],
    )
    domains = _domains_grid_2x2()
    fig.update_layout(**domains)
    borders = []
    for key in ['scene', 'scene2', 'scene3', 'scene4']:
        d = domains[key]['domain']
        borders.append(dict(
            type="rect", xref="paper", yref="paper",
            x0=d['x'][0], x1=d['x'][1], y0=d['y'][0], y1=d['y'][1],
            line=dict(color="black", width=2), fillcolor="rgba(0,0,0,0)", layer="above"
        ))
    fig.update_layout(shapes=borders)
    return fig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=5_000)
    parser.add_argument("--noise", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--centers", type=int, default=3, help="Number of classes (spiral or blobs; <=4 fits UI)")
    parser.add_argument("--grid-n", type=int, default=160, help="Grid resolution per axis")
    parser.add_argument("--dataset", type=str, default="spiral", choices=["spiral","blobs"], help="Dataset type")
    parser.add_argument("--activation", type=str, default="tanh", choices=["relu","identity","tanh","logistic"], help="MLP hidden activation")
    parser.add_argument("--point-subsample", type=int, default=500, help="Max points to plot as projections")
    parser.add_argument("--save-html", type=str, default=None, help="If set, save figure to this HTML file")
    args = parser.parse_args()
    Xtr, Xte, ytr, yte = make_data(args.n_samples, args.noise, args.seed, centers=args.centers, dataset=args.dataset)
    clf = train_mlp(Xtr, ytr, activation=args.activation, seed=args.seed)
    test_acc = accuracy_score(yte, clf.predict(Xte))
    print(f"Test accuracy: {test_acc:.4f}")
    X_all = np.vstack([Xtr, Xte])
    y_all = np.hstack([ytr, yte])
    x_min, x_max = X_all[:, 0].min() - 0.6, X_all[:, 0].max() + 0.6
    y_min, y_max = X_all[:, 1].min() - 0.6, X_all[:, 1].max() + 0.6
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, args.grid_n), np.linspace(y_min, y_max, args.grid_n))
    probs_grid, logits_grid, _, _ = forward_all(clf, np.c_[xx.ravel(), yy.ravel()])
    fig = build_figure(xx, yy, probs_grid, logits_grid, X_all, y_all, clf,
                       title_extra=f" | {args.dataset} {args.centers} classes n={args.n_samples}, noise/std={args.noise}, acc={test_acc:.3f}",
                       point_subsample=args.point_subsample)
    if args.save_html:
        fig.write_html(args.save_html, include_plotlyjs="cdn")
        print(f"Saved interactive figure to: {args.save_html}")
    else:
        fig.show()

if __name__ == "__main__":
    main()
