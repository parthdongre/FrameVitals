"""
Quick sanity check for demo_datasets/datalens_master_demo.csv.

Confirms the generated dataset actually exercises every pipeline module by
inspecting:
  - shape, dtypes
  - missingness pattern
  - duplicate rows
  - near-leakage correlation (payment_amount vs monthly_revenue)
  - VIF-pair correlations (mau vs sessions, monthly vs annual revenue)
  - bimodality + skew on selected numerics
  - regex hits inside customer_feedback
  - drift signal between halves split on application_date
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "demo_datasets" / "datalens_master_demo.csv"

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
URL_RE = re.compile(r"https?://[^\s)]+|www\.[^\s)]+")
PHONE_RE = re.compile(r"(?:\+?\d[\d\-.\s()]{6,}\d)")
MONEY_RE = re.compile(r"(?:[$€£¥₹]|USD|EUR|GBP)\s*\d[\d,.]*")
TAG_RE = re.compile(r"(?:^|\s)[@#][\w_]+")


def main() -> None:
    if not CSV.exists():
        print(f"missing: {CSV}")
        sys.exit(1)

    df = pd.read_csv(CSV, parse_dates=["application_date", "last_login_timestamp"])
    n = len(df)
    print(f"shape          {df.shape}")
    print(f"memory MB      {df.memory_usage(deep=True).sum() / 1024**2:.2f}")
    print()

    print("dtypes (first 30):")
    for c in df.columns:
        print(f"  {c:32s}  {str(df[c].dtype):14s}  "
              f"unique={df[c].nunique(dropna=True):>6}  "
              f"missing={df[c].isna().sum()}")
    print()

    print(f"duplicate rows                 {df.duplicated().sum()}")
    print(f"missing % overall              "
          f"{df.isna().sum().sum() / (n * df.shape[1]) * 100:.2f}")
    print()

    pairs = [
        ("monthly_active_users", "sessions_per_month",
         "VIF pair 1 (should be > 0.95)"),
        ("monthly_revenue", "annual_revenue",
         "VIF pair 2 (should be > 0.99)"),
        ("monthly_revenue", "payment_amount",
         "near-leakage (should be high after the lifted-row patch)"),
        ("total_lifetime_value", "revenue_score",
         "engineered leakage (should be > 0.99 → flagged High)"),
        ("feature_adoption_score", "net_promoter_score",
         "Spearman > Pearson (monotone non-linear)"),
        ("monthly_active_users", "has_sso",
         "binary vs numeric (point-biserial)"),
    ]
    print("correlation probes:")
    for a, b, note in pairs:
        sub = df[[a, b]].dropna()
        if sub[b].dtype == bool or sub[b].dtype == object:
            sub[b] = sub[b].astype(int)
        if sub[a].dtype == object:
            continue
        pearson = sub[a].corr(sub[b].astype(float))
        spearman = sub[a].corr(sub[b].astype(float), method="spearman")
        print(f"  {a:25s}  vs  {b:25s}  pearson={pearson:+.3f}  "
              f"spearman={spearman:+.3f}  ·  {note}")
    print()

    print("shape probes:")
    for c in ("monthly_revenue", "feature_adoption_score",
              "support_tickets_per_month", "onboarding_score"):
        s = df[c].dropna()
        skew = s.skew()
        kurt = s.kurtosis()
        print(f"  {c:30s}  γ₁={skew:+.2f}  γ₂={kurt:+.2f}")
    print()

    fb = df["customer_feedback"].dropna().astype(str)
    print(f"customer_feedback rows         {len(fb)}")
    print(f"  email hits                   {fb.str.contains(EMAIL_RE).sum()}")
    print(f"  url hits                     {fb.str.contains(URL_RE).sum()}")
    print(f"  phone hits                   {fb.str.contains(PHONE_RE).sum()}")
    print(f"  monetary hits                {fb.str.contains(MONEY_RE).sum()}")
    print(f"  hash/mention hits            {fb.str.contains(TAG_RE).sum()}")
    print(f"  avg length (chars)           {fb.str.len().mean():.1f}")
    print(f"  avg tokens                   "
          f"{fb.str.split().str.len().mean():.2f}")
    print()

    # Drift split: older half vs newer half
    df = df.sort_values("application_date").reset_index(drop=True)
    cut = len(df) // 2
    a = df.iloc[:cut]
    b = df.iloc[cut:]

    def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
        ref = ref[np.isfinite(ref)]
        cur = cur[np.isfinite(cur)]
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            return float("nan")
        edges[0] = -np.inf
        edges[-1] = np.inf
        rh, _ = np.histogram(ref, bins=edges)
        ch, _ = np.histogram(cur, bins=edges)
        rp = np.where(rh == 0, 1e-6, rh / rh.sum())
        cp = np.where(ch == 0, 1e-6, ch / ch.sum())
        return float(np.sum((cp - rp) * np.log(cp / rp)))

    print("drift between halves (split on application_date):")
    for c in ("monthly_revenue", "feature_adoption_score",
              "monthly_active_users", "annual_revenue"):
        v = psi(a[c].astype(float).to_numpy(), b[c].astype(float).to_numpy())
        bucket = ("stable" if v < 0.10 else
                  "minor" if v < 0.25 else
                  "moderate" if v < 0.50 else
                  "severe")
        print(f"  {c:30s}  PSI={v:.3f}  →  {bucket}")
    # Tier mix shift
    a_mix = a["product_tier"].value_counts(normalize=True)
    b_mix = b["product_tier"].value_counts(normalize=True)
    print()
    print("product_tier mix:")
    for tier in ["free", "starter", "business", "enterprise"]:
        print(f"  {tier:12s}  ref={a_mix.get(tier, 0):.3f}  "
              f"cur={b_mix.get(tier, 0):.3f}")


if __name__ == "__main__":
    main()
