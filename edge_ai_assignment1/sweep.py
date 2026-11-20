"""
sweep.py -- Mission 2 experiments. Does not touch model/.

    python sweep.py --team 22 [--epochs 600]
Prints/saves: normalisation on vs off, a width x depth sweep (train vs val MAE, params),
and sweep_mission2.png / sweep_mission2.csv.
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from app.model import FEATURES, BatteryNet

ap = argparse.ArgumentParser()
ap.add_argument("--team", type=int, required=True)
ap.add_argument("--epochs", type=int, default=600)
a = ap.parse_args()

# same cleaning and split as train.py
df = pd.read_csv(f"data/team_{a.team:02d}_train.csv")
cols = FEATURES + ["days_left"]
bad = (df.battery_v <= 0) | (df.temp_c < -75) | (df.wifi_rssi_dbm >= 0) | ~np.isfinite(df[cols]).all(axis=1)
df = df[~bad]
rng = np.random.default_rng(0)
idx = rng.permutation(len(df)); n_val = len(df) // 5
val, tr = df.iloc[idx[:n_val]], df.iloc[idx[n_val:]]
X, y = tr[FEATURES].to_numpy(np.float32), tr.days_left.to_numpy(np.float32)
Xv, yv = val[FEATURES].to_numpy(np.float32), val.days_left.to_numpy(np.float32)
mu, sd = X.mean(0), X.std(0) + 1e-8
mae = lambda p, t: float(np.mean(np.abs(p - t)))


def fit(hidden, normalise=True, epochs=a.epochs):
    m, s = (mu, sd) if normalise else (np.zeros(5, np.float32), np.ones(5, np.float32))
    Xt, Xvt, yt = torch.tensor((X - m) / s), torch.tensor((Xv - m) / s), torch.tensor(y)
    torch.manual_seed(0)
    net = BatteryNet(hidden)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 128):
            b = perm[i:i + 128]
            loss = torch.nn.functional.mse_loss(net(Xt[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return (sum(p.numel() for p in net.parameters()),
                mae(net(Xt).numpy(), y), mae(net(Xvt).numpy(), yv))


print(f"epochs per run: {a.epochs}\n\n-- normalisation (32,32) --")
for flag in (False, True):
    _, trm, vam = fit((32, 32), flag)
    print(f"normalise={flag!s:5}  train MAE {trm:6.2f}  val MAE {vam:6.2f}")

print("\n-- width / depth sweep --")
results = []
for layers in (1, 2):
    for w in (4, 8, 16, 32, 64):
        h = (w,) * layers
        p, trm, vam = fit(h)
        results.append(dict(hidden=str(h), layers=layers, width=w, params=p, train_mae=round(trm, 2), val_mae=round(vam, 2)))
        print(f"{str(h):10} params {p:5d}  train {trm:6.2f}  val {vam:6.2f}  gap {vam - trm:5.2f}")

with open("sweep_mission2.csv", "w", newline="") as fh:
    w_ = csv.DictWriter(fh, fieldnames=results[0].keys()); w_.writeheader(); w_.writerows(results)
plt.figure(figsize=(7, 4.5))
for layers in (1, 2):
    r = [x for x in results if x["layers"] == layers]
    plt.plot([x["params"] for x in r], [x["val_mae"] for x in r], "o-", label=f"{layers} hidden layer(s): val")
    plt.plot([x["params"] for x in r], [x["train_mae"] for x in r], "x--", alpha=.5, label=f"{layers} hidden layer(s): train")
plt.xscale("log"); plt.xlabel("parameters"); plt.ylabel("MAE (days)"); plt.legend(); plt.grid(alpha=.3)
plt.title(f"Team {a.team}: MAE vs model size"); plt.tight_layout(); plt.savefig("sweep_mission2.png", dpi=150)
print("\nsaved sweep_mission2.csv and sweep_mission2.png")
