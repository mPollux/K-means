#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analisa o CSV gerado pelo run_bench.sh e produz:
- Tabelas agregadas (medianas)
- Gráficos: Tempo vs Threads, Speedup vs Threads, Pontos/s vs Threads
- Comparativo de schedule (static vs dynamic) e afinamento de chunk (em T fixo)
- Validação: estabilidade do SSE entre repetições e igualdade seq vs omp

Uso:
    python3 analisar_bench.py resultados_YYYYMMDD_HHMMSS.csv

Requisitos:
    pip install pandas matplotlib
"""

import sys
import os
import math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======== Parâmetros do experimento (ajuste se quiser) ========
# Mapeia dataset -> N (pontos) (coerente com os geradores sugeridos)
DATASET_N = {"p": 10_000, "m": 100_000, "g": 1_000_000}

# T escolhido para gráficos de "efeito de schedule/chunk"
THREADS_FIXED_PREFERRED = 8
TOL_SSE = 1e-9  # tolerância para comparar SSEs


def load_csv(path):
    df = pd.read_csv(path)
    # Normaliza strings "NA"
    for col in ["schedule", "chunk"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].replace({"nan": "NA"})
    # Normaliza numéricos
    df["threads"] = pd.to_numeric(df["threads"], errors="coerce")
    df["rep"] = pd.to_numeric(df["rep"], errors="coerce")
    df["iters"] = pd.to_numeric(df["iters"], errors="coerce")
    df["sse_final"] = pd.to_numeric(df["sse_final"], errors="coerce")
    df["time_ms"] = pd.to_numeric(df["time_ms"], errors="coerce")
    # median_ms pode estar NA nas reps (1..5); nas resumo (rep=0) está preenchido
    if "median_ms" in df.columns:
        df["median_ms"] = pd.to_numeric(df["median_ms"], errors="coerce")
    else:
        df["median_ms"] = np.nan
    return df


def pick_threads_fixed_available(df_omp):
    # escolhe T para gráficos de schedule dinamicamente (prefere 8; senão maior disponível)
    candidates = sorted(df_omp["threads"].dropna().unique().tolist())
    if THREADS_FIXED_PREFERRED in candidates:
        return THREADS_FIXED_PREFERRED
    return candidates[-1] if candidates else None


def summarize_baselines(df):
    """Extrai baseline sequencial por dataset (mediana, linha rep=0)"""
    base = (
        df[(df["modo"] == "seq") & (df["rep"] == 0)]
        .groupby("dataset", as_index=False)["median_ms"]
        .min()
        .rename(columns={"median_ms": "tempo_serial_ms"})
    )
    return base


def compute_speedup_tables(df, base):
    """Adiciona coluna speedup usando a mediana do seq do mesmo dataset"""
    df = df.copy()
    df = df.merge(base, on="dataset", how="left")
    # usar median_ms quando rep=0 (resumo). Para linhas individuais, use time_ms.
    df["tempo_ref_ms"] = np.where(df["rep"] == 0, df["median_ms"], df["time_ms"])
    df["speedup"] = df["tempo_serial_ms"] / df["tempo_ref_ms"]
    return df


def plot_time_speedup_throughput(df_omp_medians, dataset, outdir):
    """Plota Tempo vs Threads, Speedup vs Threads, Pontos/s vs Threads (usando median_ms)."""
    os.makedirs(outdir, exist_ok=True)
    # Para “schedule base” (sem chunk) mostramos curvas separadas p/ static e dynamic
    base_mask = (df_omp_medians["dataset"] == dataset) & (df_omp_medians["chunk"] == "NA")
    dsub = df_omp_medians[base_mask].copy()
    if dsub.empty:
        return

    # Ordena
    dsub = dsub.sort_values(["schedule", "threads"])

    # Tempo vs Threads
    plt.figure()
    for sched, grp in dsub.groupby("schedule"):
        plt.plot(grp["threads"], grp["median_ms"], marker="o", label=f"{sched}")
    plt.xlabel("Threads")
    plt.ylabel("Tempo (ms)")
    plt.title(f"Tempo vs Threads — dataset {dataset}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{dataset}_tempo_vs_threads.png"))
    plt.close()

    # Speedup vs Threads
    # speedup (rep=0) já calculado em df_omp_medians["speedup"]
    plt.figure()
    for sched, grp in dsub.groupby("schedule"):
        plt.plot(grp["threads"], grp["speedup"], marker="o", label=f"{sched}")
    plt.xlabel("Threads")
    plt.ylabel("Speedup (mediana)")
    plt.title(f"Speedup vs Threads — dataset {dataset}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{dataset}_speedup_vs_threads.png"))
    plt.close()

    # Pontos/s vs Threads  => (N / (ms/1000)) = N * 1000 / ms
    N = DATASET_N.get(dataset, np.nan)
    if not math.isnan(N):
        dsub["points_per_s"] = N * 1000.0 / dsub["median_ms"]
        plt.figure()
        for sched, grp in dsub.groupby("schedule"):
            plt.plot(grp["threads"], grp["points_per_s"], marker="o", label=f"{sched}")
        plt.xlabel("Threads")
        plt.ylabel("Pontos por segundo")
        plt.title(f"Pontos/s vs Threads — dataset {dataset}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{dataset}_pps_vs_threads.png"))
        plt.close()


def plot_schedule_chunk_bars(df_omp_medians, dataset, outdir):
    """Barra comparando schedules e chunks em um T fixo."""
    os.makedirs(outdir, exist_ok=True)

    Tfixed = pick_threads_fixed_available(df_omp_medians[df_omp_medians["dataset"] == dataset])
    if Tfixed is None:
        return

    d = df_omp_medians[(df_omp_medians["dataset"] == dataset) & (df_omp_medians["threads"] == Tfixed)].copy()
    if d.empty:
        return

    # Mostrar barras agrupadas: eixo X = (schedule, chunk), valor = median_ms (menor é melhor)
    d["label"] = d.apply(lambda r: f"{r['schedule']}" if r["chunk"] == "NA" else f"{r['schedule']},{r['chunk']}", axis=1)
    d = d.sort_values(["schedule", "chunk"])

    plt.figure(figsize=(10, 4))
    plt.bar(d["label"], d["median_ms"])
    plt.ylabel("Tempo (ms)")
    plt.title(f"Schedule/Chunk (T={Tfixed}) — dataset {dataset}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{dataset}_sched_chunk_T{Tfixed}.png"))
    plt.close()


def make_validation_report(df, outpath):
    """Valida SSE:
       (1) estabilidade em repetições por configuração omp/seq
       (2) igualdade seq vs omp (medianas) por dataset dentro da tolerância
       Gera um .txt com o resumo.
    """
    lines = []
    lines.append("VALIDACAO (SSE FINAL)\n")
    lines.append(f"Tolerancia usada: {TOL_SSE:e}\n")

    # (1) variação entre repetições (mesma config) — seq e omp
    def keycols(mode):
        if mode == "seq":
            return ["dataset", "modo"]
        else:
            return ["dataset", "modo", "threads", "schedule", "chunk"]

    for mode in ["seq", "omp"]:
        cfg_cols = keycols(mode)
        df_reps = df[(df["modo"] == mode) & (df["rep"] > 0)].copy()
        if df_reps.empty:
            continue

        lines.append(f"\n(1) Estabilidade entre repeticoes — {mode}\n")
        for cfg, grp in df_reps.groupby(cfg_cols):
            sse_vals = grp["sse_final"].dropna().values
            if len(sse_vals) == 0:
                continue
            sse_min, sse_max = np.min(sse_vals), np.max(sse_vals)
            ok = (sse_max - sse_min) <= TOL_SSE
            lines.append(f"  {cfg}: min={sse_min:.6f} max={sse_max:.6f} | ok={ok}")

    # (2) seq vs omp (rep=0 medianas), por dataset comparando SSE médios
    lines.append("\n(2) Igualdade seq vs omp (por dataset, usando medianas de SSE final)\n")
    seq_med = df[(df["modo"] == "seq") & (df["rep"] > 0)].groupby("dataset")["sse_final"].median()
    omp_med = df[(df["modo"] == "omp") & (df["rep"] > 0)].groupby("dataset")["sse_final"].median()

    for ds in sorted(df["dataset"].dropna().unique()):
        s_seq = seq_med.get(ds, np.nan)
        s_omp = omp_med.get(ds, np.nan)
        if not (np.isnan(s_seq) or np.isnan(s_omp)):
            diff = abs(s_seq - s_omp)
            ok = diff <= TOL_SSE
            lines.append(f"  dataset={ds}: seq_med={s_seq:.6f} omp_med={s_omp:.6f} | |diff|={diff:.3e} | ok={ok}")

    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    if len(sys.argv) < 2:
        print("Uso: python3 analisar_bench.py resultados_YYYYMMDD_HHMMSS.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    outdir = "figs_bench"
    os.makedirs(outdir, exist_ok=True)

    df = load_csv(csv_path)

    # Tabelas de medianas (rep = 0 já traz mediana calculada pelo script bash)
    # Mantemos apenas as linhas-resumo (rep=0) para os gráficos de curvas:
    df_med = df[df["rep"] == 0].copy()

    # Baselines sequenciais (por dataset)
    base = summarize_baselines(df)

    # Adiciona speedup às medianas
    df_med = compute_speedup_tables(df_med, base)

    # ====== Geração de gráficos por dataset ======
    for ds in sorted(df_med["dataset"].dropna().unique()):
        plot_time_speedup_throughput(df_med[df_med["modo"] == "omp"], ds, outdir)
        plot_schedule_chunk_bars(df_med[df_med["modo"] == "omp"], ds, outdir)

    # ====== Relatório de validação (usa reps individuais) ======
    make_validation_report(df, os.path.join(outdir, "validacao_sse.txt"))

    # ====== Exporta tabelas-síntese úteis para o relatório ======
    # 1) Curvas tempo/speedup (só medianas OMP)
    curves = (
        df_med[df_med["modo"] == "omp"]
        .loc[:, ["dataset", "threads", "schedule", "chunk", "median_ms", "speedup"]]
        .sort_values(["dataset", "schedule", "chunk", "threads"])
    )
    curves.to_csv(os.path.join(outdir, "curvas_tempo_speedup.csv"), index=False)

    # 2) Tabela de melhor config por (dataset, threads) (menor tempo)
    best_per_T = (
        df_med[df_med["modo"] == "omp"]
        .sort_values(["dataset", "threads", "median_ms"])
        .groupby(["dataset", "threads"], as_index=False)
        .first()
        .loc[:, ["dataset", "threads", "schedule", "chunk", "median_ms", "speedup"]]
    )
    best_per_T.to_csv(os.path.join(outdir, "melhor_config_por_threads.csv"), index=False)

    print("\nConcluído.")
    print(f"- Figuras em: {outdir}/*.png")
    print(f"- Validação SSE: {outdir}/validacao_sse.txt")
    print(f"- Tabelas: {outdir}/curvas_tempo_speedup.csv e {outdir}/melhor_config_por_threads.csv")


if __name__ == "__main__":
    main()
