# (a) Unif[0, theta]:  theta_hat(p) = qhat_p / p, p = 0.05..0.95
# (b) Unif[0, theta]:  theta_hat(k) = ((k+1)*mean(X^k))^(1/k), k = 2..50
# (c) Exp(theta) (rate): theta_hat(k) = (k! / mean(X^k))^(1/k), k = 2..20


import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class MCConfig:
    n: int
    B: int
    theta_unif: float
    theta_exp: float
    seed: int
    outdir: str


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv(path: str, header: List[str], rows: List[List[float]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def read_csv_xy(path: str, x_col: str, y_col: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        xs, ys = [], []
        for row in r:
            xs.append(float(row[x_col]))
            ys.append(float(row[y_col]))
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def compute_rmse_bias_var(theta_hat: np.ndarray, theta_true: float) -> Tuple[float, float, float]:
    err = theta_hat - theta_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    var = float(np.var(theta_hat, ddof=1))  # выборочная дисперсия
    return rmse, bias, var


def gen_a_unif_quantile(cfg: MCConfig) -> str:
    """
    (a) Unif[0, theta]: theta_hat(k) = k * X_(1/k), where p = 1/k
    Реализуем X_(p) как порядковую статистику X_(ceil(n*p)).
    """
    rng = np.random.default_rng(cfg.seed)

    ks = np.arange(2, 51)
    ps = 1.0 / ks          # p = 1/k

    n, B, theta = cfg.n, cfg.B, cfg.theta_unif

    # X ~ Unif[0, theta]
    X = rng.random((B, n)) * theta
    Xs = np.sort(X, axis=1)

    idxs = (np.ceil(n * ps).astype(int) - 1).clip(0, n - 1)
    qhat = Xs[:, idxs]  # shape (B, len(ks))

    theta_hat = qhat / ps

    rows: List[List[float]] = []
    for j, k in enumerate(ks):
        rmse, bias, var = compute_rmse_bias_var(theta_hat[:, j], theta)
        rows.append([theta, n, B, int(k), float(ps[j]), rmse, bias, var])

    out_path = os.path.join(cfg.outdir, "a_unif_quantile_by_k.csv")
    write_csv(
        out_path,
        header=["theta", "n", "B", "k", "p", "rmse", "bias", "var"],
        rows=rows
    )
    return out_path


def gen_b_unif_moment(cfg: MCConfig) -> str:
    """
    (b) Unif[0, theta]: theta_hat(k) = ((k+1)*mean(X^k))^(1/k), k=2..50
    """
    rng = np.random.default_rng(cfg.seed + 1)

    ks = np.arange(2, 50)
    n, B, theta = cfg.n, cfg.B, cfg.theta_unif

    X = rng.random((B, n)) * theta  # Unif[0,theta]

    rows: List[List[float]] = []
    Xpow = None

    for k in ks:
        if Xpow is None:
            Xpow = X * X
        else:
            Xpow = Xpow * X

        moment = Xpow.mean(axis=1)
        theta_hat = np.exp((math.log(k + 1.0) + np.log(moment)) / float(k))

        rmse, bias, var = compute_rmse_bias_var(theta_hat, theta)
        rows.append([theta, n, B, int(k), rmse, bias, var])

    out_path = os.path.join(cfg.outdir, "b_unif_moment.csv")
    write_csv(out_path,
              header=["theta", "n", "B", "k", "rmse", "bias", "var"],
              rows=rows)
    return out_path


def gen_c_exp_moment(cfg: MCConfig) -> str:
    """
    (c) Exp(theta) with rate theta: f(x)=theta*exp(-theta x), x>=0
        theta_hat(k) = (k! / mean(X^k))^(1/k), k=2..20
    """
    rng = np.random.default_rng(cfg.seed + 2)

    ks = np.arange(2, 20)
    n, B, theta = cfg.n, cfg.B, cfg.theta_exp

    X = rng.exponential(scale=1.0 / theta, size=(B, n))

    rows: List[List[float]] = []
    Xpow = None

    for k in ks:
        if Xpow is None:
            Xpow = X * X
        else:
            Xpow = Xpow * X

        moment = Xpow.mean(axis=1)

        log_k_fact = math.lgamma(int(k) + 1)
        theta_hat = np.exp((log_k_fact - np.log(moment)) / float(k))

        rmse, bias, var = compute_rmse_bias_var(theta_hat, theta)
        rows.append([theta, n, B, int(k), rmse, bias, var])

    out_path = os.path.join(cfg.outdir, "c_exp_moment.csv")
    write_csv(out_path,
              header=["theta", "n", "B", "k", "rmse", "bias", "var"],
              rows=rows)
    return out_path


def plot_from_csv(cfg: MCConfig) -> List[str]:
    """
    Читает CSV и рисует графики rmse vs p/k. Сохраняет PNG.
    """
    out_pngs: List[str] = []

    # (a)
    a_csv = os.path.join(cfg.outdir, "a_unif_quantile_by_k.csv")
    if os.path.exists(a_csv):
        k, rmse = read_csv_xy(a_csv, "k", "rmse")
        plt.figure()
        plt.plot(k, rmse)
        plt.xlabel("k")
        plt.ylabel("СКО")
        plt.title(f"(a) Unif[0,theta], theta_hat=k*X_(1/k); theta={cfg.theta_unif}, n={cfg.n}, B={cfg.B}")
        plt.grid(True)
        out = os.path.join(cfg.outdir, "a_unif_quantile_by_k.png")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        out_pngs.append(out)

    # (b)
    b_csv = os.path.join(cfg.outdir, "b_unif_moment.csv")
    if os.path.exists(b_csv):
        k, rmse = read_csv_xy(b_csv, "k", "rmse")
        plt.figure()
        plt.plot(k, rmse)
        plt.xlabel("k")
        plt.ylabel("СКО")
        plt.title(f"(b) Unif[0,theta], theta_hat=((k+1)*mean(X^k))^(1/k); theta={cfg.theta_unif}, n={cfg.n}, B={cfg.B}")
        plt.grid(True)
        out = os.path.join(cfg.outdir, "b_unif_moment.png")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        out_pngs.append(out)

    # (c)
    c_csv = os.path.join(cfg.outdir, "c_exp_moment.csv")
    if os.path.exists(c_csv):
        k, rmse = read_csv_xy(c_csv, "k", "rmse")
        plt.figure()
        plt.plot(k, rmse)
        plt.xlabel("k")
        plt.ylabel("СКО")
        plt.title(f"(c) Exp(theta) rate, theta_hat=(k!/mean(X^k))^(1/k); theta={cfg.theta_exp}, n={cfg.n}, B={cfg.B}")
        plt.grid(True)
        out = os.path.join(cfg.outdir, "c_exp_moment.png")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        out_pngs.append(out)

    return out_pngs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="Размер выборки n")
    ap.add_argument("--B", type=int, default=10_000, help="Число повторов Монте-Карло B")
    ap.add_argument("--theta", type=float, default=1.0, help="Фиксированная theta (по умолчанию для всех пунктов)")
    ap.add_argument("--seed", type=int, default=13, help="Seed")
    ap.add_argument("--outdir", type=str, default="out_mc", help="Папка для CSV и PNG")
    ap.add_argument("--only-generate", action="store_true", help="Только сгенерировать CSV")
    ap.add_argument("--only-plot", action="store_true", help="Только построить графики из CSV")
    args = ap.parse_args()

    theta_unif = args.theta
    theta_exp = args.theta

    cfg = MCConfig(
        n=args.n,
        B=args.B,
        theta_unif=float(theta_unif),
        theta_exp=float(theta_exp),
        seed=args.seed,
        outdir=args.outdir,
    )

    ensure_outdir(cfg.outdir)

    if not args.only_plot:
        a_path = gen_a_unif_quantile(cfg)
        b_path = gen_b_unif_moment(cfg)
        c_path = gen_c_exp_moment(cfg)
        print("CSV saved:")
        print(" ", a_path)
        print(" ", b_path)
        print(" ", c_path)

    if not args.only_generate:
        pngs = plot_from_csv(cfg)
        print("PNGs saved:")
        for p in pngs:
            print(" ", p)


if __name__ == "__main__":
    main()
