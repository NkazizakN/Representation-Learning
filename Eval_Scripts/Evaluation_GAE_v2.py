import os, csv, argparse, importlib
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, silhouette_score
from sklearn.manifold import TSNE
import umap

from data_loader import ARMD, armd_splits, loader, tf, ROOT

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVAL_ROOT = "evaluation"
SEEDS     = [0, 1, 2, 3, 4]
EMB_BS    = 16            # smaller batch: GAE is memory-heavy (dynamic k-NN)


def load_model(model_name):
    mod = importlib.import_module(model_name)
    model = mod.GAE().to(DEVICE)
    state = torch.load(mod.BEST_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def get_embeddings(model):
    full = ARMD(os.path.join(ROOT, "RFMid"), tf(False))
    dl = loader(full, shuffle=False, bs=EMB_BS, nw=4)
    embs, labels = [], []
    for imgs, y, _ in dl:
        imgs = imgs.to(DEVICE)
        _, z = model(imgs)
        embs.append(z.mean(dim=(2, 3)).cpu().numpy())
        labels.append(np.array(y))
    return np.concatenate(embs), np.concatenate(labels), full


def run_probes(embs, labels, full):
    test_idx, rp, rn = armd_splits()
    train_idx = np.array(rp + rn)
    Xtr_full, ytr_full = embs[train_idx], labels[train_idx]
    Xte, yte = embs[test_idx], labels[test_idx]

    # variance comes from bootstrapping the TRAINING SET per seed
    # (resampling train data with replacement), not the classifier RNG —
    # a converged linear fit is insensitive to random_state, which is why
    # seed-only variation gave std=0.
    lin_auc, lin_acc = [], []
    n = len(train_idx)
    clfs = []
    for s in SEEDS:
        rng = np.random.RandomState(s)
        boot = rng.choice(n, size=n, replace=True)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(Xtr_full[boot], ytr_full[boot])
        p = clf.predict_proba(Xte)[:, 1]
        lin_auc.append(roc_auc_score(yte, p))
        lin_acc.append(accuracy_score(yte, clf.predict(Xte)))
        clfs.append(clf)

    out = {
        "lin_auc_mean": float(np.mean(lin_auc)), "lin_auc_std": float(np.std(lin_auc)),
        "lin_acc_mean": float(np.mean(lin_acc)),
    }

    # ── purity breakdown: probe quality on ARMD-only vs ARMD+other test cases ──
    # Averaged probability over the seed ensemble; split positives by co-label group.
    g_all = _armd_colabel_groups(full)               # 0 not-ARMD, 1 ARMD-only, 2 ARMD+other
    g_te = g_all[test_idx]
    P = np.mean([c.predict_proba(Xte)[:, 1] for c in clfs], axis=0)
    neg_mask = (yte == 0)
    for grp, name in [(1, "armd_only"), (2, "armd_other")]:
        pos_mask = (g_te == grp)
        if pos_mask.sum() == 0:
            out[f"{name}_auc"] = float("nan"); out[f"{name}_n"] = 0
            continue
        # AUC for this ARMD subgroup vs ALL test negatives
        sub = pos_mask | neg_mask
        y_sub = yte[sub]
        p_sub = P[sub]
        out[f"{name}_auc"] = float(roc_auc_score(y_sub, p_sub)) if len(set(y_sub)) > 1 else float("nan")
        out[f"{name}_n"] = int(pos_mask.sum())
    return out

def run_silhouette(embs, labels):
    return float(silhouette_score(embs, labels))


# ── 3-colour projection: ARMD-only / ARMD+other / not-ARMD ──
def _armd_colabel_groups(full):
    """
    Returns an array g over full.samples:
        0 = not-ARMD
        1 = ARMD-only      (ARMD=1, no other disease)
        2 = ARMD+other     (ARMD=1, >=1 other disease)
    Reads RFMid CSVs directly to recover co-label counts, which the ARMD
    dataset class does not store. Matched back to samples by image stem.
    """
    import pandas as pd
    csv_names = ["RFMiD_Training_Labels.csv", "RFMiD_Validation_Labels.csv",
                 "RFMiD_Testing_Labels.csv"]
    rfmid_root = os.path.join(ROOT, "RFMid")
    frames = []
    for c in csv_names:
        cp = os.path.join(rfmid_root, c)
        if os.path.isfile(cp):
            d = pd.read_csv(cp)
            d.columns = d.columns.str.lstrip("\ufeff").str.strip()
            d["ID"] = d["ID"].astype(str).str.strip()
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    dcols = [c for c in df.columns if c not in ("ID", "Disease_Risk")]
    other_cols = [c for c in dcols if c != "ARMD"]
    # per-ID: (armd flag, other-disease count)
    info = {}
    for _, r in df.iterrows():
        info[str(r["ID"])] = (int(r["ARMD"]), int(sum(int(r[c]) for c in other_cols)))
    g = np.zeros(len(full.samples), dtype=int)
    for i, (path, _) in enumerate(full.samples):
        stem = os.path.splitext(os.path.basename(path))[0]
        armd, n_other = info.get(stem, (0, 0))
        if armd == 0:
            g[i] = 0
        elif n_other == 0:
            g[i] = 1
        else:
            g[i] = 2
    return g


def plot_projection(embs, full, method, out_path, seed=0):
    assert embs.shape[0] == len(full.samples), \
        f"embedding/sample mismatch: {embs.shape[0]} vs {len(full.samples)}"
    rng = np.random.RandomState(seed)
    g_all = _armd_colabel_groups(full)

    armd_only = np.where(g_all == 1)[0]
    armd_other = np.where(g_all == 2)[0]
    not_armd = np.where(g_all == 0)[0]
    # subsample the large not-ARMD group so it doesn't swamp the plot
    not_keep = rng.choice(not_armd, size=min(400, len(not_armd)), replace=False)

    keep = np.concatenate([not_keep, armd_other, armd_only])
    X = embs[keep]
    g = g_all[keep]

    if method == "tsne":
        proj = TSNE(n_components=2, random_state=seed, init="pca").fit_transform(X)
    else:
        proj = umap.UMAP(n_components=2, random_state=seed).fit_transform(X)

    plt.figure(figsize=(7, 6))
    # draw not-ARMD first (background), then the two ARMD groups on top
    for val, name, color, z in [
        (0, "not-ARMD",   "#cccccc", 1),
        (2, "ARMD+other", "#1f77b4", 2),
        (1, "ARMD-only",  "#d62728", 3),
    ]:
        m = g == val
        plt.scatter(proj[m, 0], proj[m, 1], s=14, c=color, label=name,
                    alpha=0.75, zorder=z)
    plt.legend(); plt.title(method.upper()); plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()
    np.save(out_path.replace(".png", "_coords.npy"), proj)
    np.save(out_path.replace(".png", "_groups.npy"), g)


# ── raw-pixel baseline (floor): probe on downsampled pixels, no learned encoder ──
def run_raw_pixel_floor(full, size=16):
    import torch.nn.functional as F
    test_idx, rp, rn = armd_splits()
    dl = loader(full, shuffle=False, bs=64, nw=4)
    feats, labels = [], []
    for imgs, y, _ in dl:
        x = F.interpolate(imgs, size=(size, size), mode="area")  # B,3,size,size
        feats.append(x.reshape(x.shape[0], -1).numpy())
        labels.append(np.array(y))
    X = np.concatenate(feats); Y = np.concatenate(labels)
    train_idx = np.array(rp + rn)
    Xtr, ytr = X[train_idx], Y[train_idx]
    Xte, yte = X[test_idx], Y[test_idx]
    aucs = []
    n = len(train_idx)
    for s in SEEDS:
        rng = np.random.RandomState(s)
        boot = rng.choice(n, size=n, replace=True)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(Xtr[boot], ytr[boot])
        aucs.append(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))
    return {"raw_pixel_auc_mean": float(np.mean(aucs)),
            "raw_pixel_auc_std": float(np.std(aucs))}



def write_txt(path, d):
    with open(path, "w") as f:
        for k, v in d.items():
            f.write(f"{k}: {v}\n")


def update_summary(model_name, metrics):
    os.makedirs(EVAL_ROOT, exist_ok=True)
    path = os.path.join(EVAL_ROOT, "summary.csv")
    row = {"model": model_name, **metrics}
    rows = []
    if os.path.isfile(path):
        with open(path) as f:
            rows = [r for r in csv.DictReader(f) if r["model"] != model_name]
    rows.append({k: str(v) for k, v in row.items()})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="GAE filename without .py (e.g. GAE_knn_v1)")
    args = ap.parse_args()
    name = args.model

    out_dir = os.path.join(EVAL_ROOT, name)
    for sub in ["linear_probe", "silhouette", "tsne", "umap"]:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    print(f"[{name}] loading model...", flush=True)
    model = load_model(name)

    print(f"[{name}] extracting embeddings...", flush=True)
    embs, labels, full = get_embeddings(model)

    print(f"[{name}] probes...", flush=True)
    probe = run_probes(embs, labels, full)
    write_txt(os.path.join(out_dir, "linear_probe", "results.txt"),
              {"auc_mean": probe["lin_auc_mean"], "auc_std": probe["lin_auc_std"],
               "acc_mean": probe["lin_acc_mean"], "seeds": SEEDS,
               "armd_only_auc": probe.get("armd_only_auc"), "armd_only_n": probe.get("armd_only_n"),
               "armd_other_auc": probe.get("armd_other_auc"), "armd_other_n": probe.get("armd_other_n")})

    print(f"[{name}] silhouette...", flush=True)
    sil = run_silhouette(embs, labels)
    write_txt(os.path.join(out_dir, "silhouette", "results.txt"), {"silhouette": sil})

    print(f"[{name}] raw-pixel floor...", flush=True)
    floor = run_raw_pixel_floor(full)
    write_txt(os.path.join(out_dir, "linear_probe", "raw_pixel_floor.txt"), floor)

    print(f"[{name}] tsne...", flush=True)
    plot_projection(embs, full, "tsne", os.path.join(out_dir, "tsne", "plot.png"))
    print(f"[{name}] umap...", flush=True)
    plot_projection(embs, full, "umap", os.path.join(out_dir, "umap", "plot.png"))

    update_summary(name, {
        "lin_auc_mean": probe["lin_auc_mean"], "lin_auc_std": probe["lin_auc_std"],
        "lin_acc_mean": probe["lin_acc_mean"],
        "armd_only_auc": probe.get("armd_only_auc"), "armd_only_n": probe.get("armd_only_n"),
        "armd_other_auc": probe.get("armd_other_auc"), "armd_other_n": probe.get("armd_other_n"),
        "raw_pixel_auc_mean": floor["raw_pixel_auc_mean"],
        "silhouette": sil,
    })
    print(f"[{name}] done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()