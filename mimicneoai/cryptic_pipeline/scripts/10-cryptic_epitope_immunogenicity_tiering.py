#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
10-cryptic_epitope_immunogenicity_tiering.py

Calculate peptide physicochemical features, run the cryptic antigen
immunogenicity model, and assign HLA-I/HLA-II tiers.

This script wraps the later parts of the former Postprocess.ipynb:
  1) calculate peptide features with R Peptides,
  2) infer Immunogenicity Score with the noncoding BiLSTM model,
  3) split HLA-I/HLA-II and assign Tier/Score.

Example (sanitized):
  python 10-cryptic_epitope_immunogenicity_tiering.py \
    -s SAMPLE_ID \
    --input /path/to/SAMPLE_ID/09-CrypticEpitopeCandidates/cryptic_epitopes_annot_extend.csv \
    -o /path/to/SAMPLE_ID/10-CrypticEpitopeTiering \
    --model-path /path/to/model/MimicNeoAI_noncoding.pth \
    --hla-prot-fasta /path/to/hla_prot.fasta \
    --r-home /path/to/conda/env/lib/R \
    --r-lib-path /path/to/conda/env/lib/R/library \
    --threads 35 \
    --batch-size 512
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import sys
import time
from collections import OrderedDict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


R_LIB_PATH: Optional[str] = None


def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def exists(path: str | Path) -> bool:
    return Path(path).exists() and Path(path).stat().st_size > 0


def check_file(path: str | Path, label: Optional[str] = None) -> None:
    if not exists(path):
        name = f" ({label})" if label else ""
        raise FileNotFoundError(f"[ERR] missing or empty{name}: {path}")


def setup_r_environment(r_home: Optional[str], r_lib_path: Optional[str]) -> None:
    global R_LIB_PATH
    if r_home:
        os.environ["R_HOME"] = r_home
    R_LIB_PATH = r_lib_path


def calculate_peptide_features(df: pd.DataFrame, peptide_col: str) -> pd.DataFrame:
    """
    Use R Peptides package to calculate peptide physicochemical features.

    This intentionally mirrors the former Postprocess.ipynb behavior and column
    names: aa_composition, polarity, volume, net_charge, hydrophobicity,
    boman_index, aliphatic_index, isoelectric_point.
    """
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri, r
    from rpy2.robjects.packages import importr

    pandas2ri.activate()
    if R_LIB_PATH:
        r(f'.libPaths(c("{R_LIB_PATH}"))')

    importr("base")
    importr("utils")
    importr("Peptides")

    r_code = f'''
    calculate_features <- function(data, peptide_col) {{
        suppressPackageStartupMessages(library(Peptides))

        calculate_polarity <- function(seq) {{
            polarities <- list(
                hydrophilic = c('A','D','E','N','Q','R','S','T','Y'),
                hydrophobic = c('I','L','M','F','P','V','W')
            )
            sum(sapply(strsplit(seq, '')[[1]], function(aa) {{
                if (aa %in% polarities$hydrophilic) 1
                else if (aa %in% polarities$hydrophobic) -1
                else 0
            }}))
        }}

        aa_volumes <- c(
            A=88.6, C=118.8, D=111.1, E=138.3, F=189.9, G=60.1,
            H=153.2, I=166.6, K=168.6, L=166.6, M=162.9, N=114.1,
            P=115.0, Q=146.2, R=174.0, S=105.9, T=119.0, V=140.0,
            W=227.8, Y=193.6
        )

        calculate_volume <- function(seq) {{
            sum(sapply(strsplit(seq, '')[[1]], function(aa) {{
                if (aa %in% names(aa_volumes)) aa_volumes[aa] else 0
            }}))
        }}

        data$aa_composition <- sapply(data[[peptide_col]], function(x)
            paste(unlist(aaComp(x)), collapse=", "))
        data$polarity <- sapply(data[[peptide_col]], calculate_polarity)
        data$volume <- sapply(data[[peptide_col]], calculate_volume)
        data$net_charge <- sapply(data[[peptide_col]], charge)
        data$hydrophobicity <- sapply(data[[peptide_col]], hydrophobicity)
        data$boman_index <- sapply(data[[peptide_col]], boman)
        data$aliphatic_index <- sapply(data[[peptide_col]], aIndex)
        data$isoelectric_point <- sapply(data[[peptide_col]], pI)

        return(data)
    }}
    '''

    ro.r(r_code)
    calculate_features = ro.globalenv["calculate_features"]

    with (ro.default_converter + pandas2ri.converter).context():
        r_df = ro.conversion.get_conversion().py2rpy(df)
        result_r = calculate_features(r_df, peptide_col)
        result_df = ro.conversion.get_conversion().rpy2py(result_r)

    return result_df


def parallel_calculate_features(peptides: pd.DataFrame, seq_col: str, num_chunks: int = 1) -> pd.DataFrame:
    total_rows = len(peptides)
    if total_rows == 0:
        return peptides.copy()

    num_chunks = max(1, min(num_chunks, total_rows))
    chunk_size = total_rows // num_chunks
    remainder = total_rows % num_chunks

    chunks = []
    start = 0
    for i in range(num_chunks):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(peptides.iloc[start:end])
        start = end

    with Pool(processes=len(chunks)) as pool:
        results = pool.starmap(calculate_peptide_features, [(chunk, seq_col) for chunk in chunks])

    return pd.concat(results, ignore_index=True)


def load_hla(hla_prot_fasta: str | Path) -> pd.DataFrame:
    hla_names = []
    prots = []
    with open(hla_prot_fasta, "r") as f:
        lines = f.readlines()
        switch = 0
        prot = ""
        for line in lines:
            if line[0] == ">":
                hla_names.append(line.split(" ")[1])
                if switch == 1:
                    prots.append(prot)
                    prot = ""
            else:
                prot += line.replace("\n", "")
                switch = 1
    prots.append(prot)
    hla2prot = pd.DataFrame(zip(hla_names, prots), columns=["HLA", "protin"])
    hla2prot["HLA"] = hla2prot["HLA"].str.split(":").str[:2].str.join(":")
    hla2prot = hla2prot.drop_duplicates(subset="HLA", keep="first")
    hla2prot = hla2prot.reset_index(drop=True)
    return hla2prot


def import_torch():
    import torch
    from torch import nn

    return torch, nn


class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []

        counter = collections.Counter()
        for token in tokens:
            if isinstance(token, list):
                counter.update(token)
            else:
                counter[token] += 1

        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        self.idx_to_token = ["<pad>"] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

        for token, freq in self.token_freqs:
            if freq >= min_freq and token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, 0)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]


def build_model():
    torch, nn = import_torch()

    class newBiLSTM(nn.Module):
        def __init__(
            self,
            vocab_size,
            embedding_dim,
            hidden_dim_x1,
            hidden_dim_x2,
            output_dim,
            n_layers,
            bidirectional,
            dropout,
            pad_idx,
            x2_dim,
        ):
            super(newBiLSTM, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
            self.lstm = nn.LSTM(
                embedding_dim,
                hidden_dim_x1,
                num_layers=n_layers,
                bidirectional=bidirectional,
                dropout=dropout,
            )
            self.fc_x2 = nn.Linear(x2_dim, hidden_dim_x2)
            self.fc = nn.Linear(hidden_dim_x1 * 2 + hidden_dim_x2, output_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x1, x2):
            self.lstm.flatten_parameters()
            embedded = self.dropout(self.embedding(x1.T))
            _, (hidden, _) = self.lstm(embedded)
            hidden_x1 = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
            hidden_x2 = self.dropout(self.fc_x2(x2))
            combined = torch.cat((hidden_x1, hidden_x2), dim=1)
            return self.fc(combined)

    return newBiLSTM


def aaindex_vocab(seq, vocab):
    encoded = np.empty([len(seq)])
    for i in range(len(seq)):
        query = seq[i]
        if query == "X":
            query = "-"
        query = query.upper()
        encoded[i] = vocab[query]
    return encoded


def process_chunk(chunk):
    chunk = chunk.reset_index(drop=True)
    features = []
    for i in range(chunk.shape[0]):
        aa_composition = chunk["aa_composition"].iloc[i].split(",")
        aa_composition = [float(value) for value in aa_composition]
        feature = aa_composition + chunk.loc[i, "polarity":"isoelectric_point"].tolist()
        features.append(feature)
    return features


def get_features_parallel(df, n_processes=None):
    torch, _ = import_torch()
    if n_processes is None:
        n_processes = cpu_count()
    n_processes = max(1, min(n_processes, len(df)))

    chunks = np.array_split(df, n_processes)

    with Pool(n_processes) as pool:
        results = list(tqdm(pool.imap(process_chunk, chunks), total=n_processes, desc="Feature tensor chunks"))

    features = [item for sublist in results for item in sublist]
    features_tensor = torch.tensor(features, dtype=torch.float32)
    return features_tensor


def encode_chunk(features_chunk, peptides_chunk, hlas_chunk, hla2prot, vocab, padding_size):
    encoded_batch = []
    flags = []
    batch_size = len(peptides_chunk)

    for i in range(batch_size):
        peptide = peptides_chunk[i]
        hla = hlas_chunk[i]
        flag = 1

        matches = hla2prot[hla2prot["HLA"] == hla]
        if matches.shape[0] == 0:
            flag = 0
            encoded_batch.append(np.zeros(padding_size))
            flags.append(flag)
            continue

        hla_paratope = matches.iloc[0, 1] if matches.shape[0] > 0 else "-"

        peptide_encode = aaindex_vocab(peptide, vocab)
        hla_encode = aaindex_vocab(hla_paratope, vocab)
        feature_array = np.array(features_chunk[i])

        zero_array = np.zeros(12, dtype=np.float32)
        merged = np.concatenate((feature_array, peptide_encode, zero_array, hla_encode))

        current_length = merged.shape[0]
        padding_length = padding_size - current_length
        merged_padded = np.pad(merged, (0, padding_length), "constant")

        encoded_batch.append(merged_padded)
        flags.append(flag)

    return encoded_batch, flags


def encode_wrapper(args):
    return encode_chunk(*args)


def parallel_encode(features, peptides, hlas, hla2prot, vocab, padding_size, workers=4):
    import torch

    workers = max(1, min(workers, len(peptides)))
    batch_size = len(peptides)
    chunk_size = batch_size // workers + (batch_size % workers > 0)

    feature_chunks = [features[i : i + chunk_size] for i in range(0, batch_size, chunk_size)]
    peptide_chunks = [peptides[i : i + chunk_size] for i in range(0, batch_size, chunk_size)]
    hla_chunks = [hlas[i : i + chunk_size] for i in range(0, batch_size, chunk_size)]

    args_iter = (
        (fc, pc, hc, hla2prot, vocab, padding_size) for fc, pc, hc in zip(feature_chunks, peptide_chunks, hla_chunks)
    )

    with Pool(processes=workers) as pool:
        results = []
        for result in tqdm(
            pool.imap(encode_wrapper, args_iter, chunksize=max(1, len(feature_chunks) // workers)),
            total=len(feature_chunks),
            desc="Encoding chunks",
        ):
            results.append(result)

    encoded_arrays = [array for result in results for array in result[0]]
    flags = [flag for result in results for flag in result[1]]

    return torch.from_numpy(np.stack(encoded_arrays)), flags


def try_gpu(i=0):
    torch, _ = import_torch()
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f"cuda:{i}")
    return torch.device("cpu")


def inference(df, thread, device, model_path, hla_prot_fasta, batch_size=512):
    torch, nn = import_torch()

    padding_length = 419
    hla2prot = load_hla(hla_prot_fasta)
    amino = "ARNDCQEGHILKMFPSTWYV-"
    tokens = list(amino)
    vocab = Vocab(tokens, min_freq=1)

    torch.set_num_threads(thread)
    peptides = df["peptide"].tolist()
    peptides = [p.replace(" ", "") for p in peptides]

    def process_hla(hla):
        if "/" in hla:
            return hla.split("/")[-1]
        return hla

    hlas = df["HLA"].apply(process_hla).tolist()

    features = get_features_parallel(df, thread)

    imm_probs = []
    log("inference....")

    newBiLSTM = build_model()
    model = newBiLSTM(
        vocab_size=len(vocab),
        embedding_dim=12,
        hidden_dim_x1=256,
        hidden_dim_x2=16,
        output_dim=2,
        n_layers=2,
        bidirectional=True,
        dropout=0.5,
        pad_idx=vocab["<pad>"],
        x2_dim=25,
    )
    model = nn.DataParallel(model)

    state_dict = torch.load(model_path, map_location=try_gpu())
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace(".module.", ".")
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.to(device=device)

    log("Parallel encoding...")
    encoded_all, flags_all = parallel_encode(features, peptides, hlas, hla2prot, vocab, padding_length, thread)

    log("Infer...")
    model.eval()
    for start in tqdm(range(0, len(peptides), batch_size), desc="Infer batches"):
        end = min(start + batch_size, len(peptides))
        batch_encoded = encoded_all[start:end].to(device)
        batch_flags = flags_all[start:end]

        with torch.no_grad():
            x1 = batch_encoded[:, 25:].int()
            x2 = batch_encoded[:, 0:25].float()
            y = model(x1, x2)
            probs = torch.softmax(y, dim=1)

            for j, flag in enumerate(batch_flags):
                imm_probs.append(float(probs[j, 1]) if flag else 0.5)

    log("done!")
    return imm_probs


def normalize_hla(df, col="HLA Allele"):
    def process(allele):
        allele = str(allele).replace("HLA-", "")
        allele = re.sub(r"(?<=[0-9])-(?=[A-Z])", "/", allele)
        return f"HLA-{allele}"

    df[col] = df[col].apply(process)
    return df


def infer_cryptic(cryptic, model_path, hla_prot_fasta, thread, batch_size):
    cryptic = normalize_hla(cryptic, "HLA Allele")
    log("HLA alleles: " + ", ".join(map(str, cryptic["HLA Allele"].unique())))
    cryptic_select_col = [
        "Epitope Seq",
        "HLA Allele",
        "aa_composition",
        "polarity",
        "volume",
        "net_charge",
        "hydrophobicity",
        "boman_index",
        "aliphatic_index",
        "isoelectric_point",
    ]
    cryptic_infer_input = cryptic[cryptic_select_col].copy()
    cryptic_infer_input = cryptic_infer_input.rename(columns={"Epitope Seq": "peptide", "HLA Allele": "HLA"})
    cryptic_infer_input["HLA"] = cryptic_infer_input["HLA"].str.replace(r"^HLA-", "", regex=True)

    cryptic_infer_input = cryptic_infer_input.drop_duplicates(subset=["HLA", "peptide"])
    cryptic_infer_input.reset_index(drop=True, inplace=True)
    log(f"immunogenicity inference unique HLA-peptide rows: {len(cryptic_infer_input)}")
    imm_probs = inference(cryptic_infer_input, thread, try_gpu(), model_path, hla_prot_fasta, batch_size)
    cryptic_infer_input["Immunogenicity Score"] = imm_probs
    cryptic_infer_input["HLA"] = "HLA-" + cryptic_infer_input["HLA"]
    cryptic_infer_input = cryptic_infer_input.rename(columns={"HLA": "HLA Allele", "peptide": "Epitope Seq"})
    cryptic = cryptic.merge(
        cryptic_infer_input[["Epitope Seq", "HLA Allele", "Immunogenicity Score"]],
        on=["Epitope Seq", "HLA Allele"],
        how="left",
    )
    return cryptic


def is_class_i(allele: str) -> bool:
    hla_i_set = {"A", "B", "C", "E", "F", "G", "H", "J", "K", "L", "V"}
    allele = (allele or "").replace("HLA-", "")
    gene = allele.split("*")[0]
    return gene in hla_i_set


ORDER = ["High", "Moderate", "Weak", "Subthreshold"]


def rank_mhci(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Tier"] = "Subthreshold"

    df.loc[
        (df["Immunogenicity Score"] > 0.5)
        & (df["Best IC50 Score"] < 500)
        & (df["Median IC50 Score"] < 500)
        & (df["Best Percentile"] < 2)
        & (df["Median Percentile"] < 2)
        & (df["TPM_tumor"] > 5)
        & (df["log2FC"] > 4),
        "Tier",
    ] = "Weak"

    df.loc[
        (df["Immunogenicity Score"] > 0.6)
        & (df["Best IC50 Score"] < 300)
        & (df["Median IC50 Score"] < 300)
        & (df["Best Percentile"] < 1)
        & (df["Median Percentile"] < 1)
        & (df["TPM_tumor"] > 7)
        & (df["log2FC"] > 5),
        "Tier",
    ] = "Moderate"

    df.loc[
        (df["Immunogenicity Score"] > 0.8)
        & (df["Best IC50 Score"] < 50)
        & (df["Median IC50 Score"] < 50)
        & (df["Best Percentile"] < 0.5)
        & (df["Median Percentile"] < 0.5)
        & (df["TPM_tumor"] > 10)
        & (df["log2FC"] > 6),
        "Tier",
    ] = "High"

    df["Tier"] = pd.Categorical(df["Tier"], categories=ORDER, ordered=True)
    return df


def rank_mhcii_strict(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Tier"] = "Subthreshold"

    df.loc[
        (df["Immunogenicity Score"] > 0.6)
        & (df["Best IC50 Score"] < 250)
        & (df["Median IC50 Score"] < 250)
        & (df["Best Percentile"] < 1)
        & (df["Median Percentile"] < 1)
        & (df["TPM_tumor"] > 5)
        & (df["log2FC"] > 4),
        "Tier",
    ] = "Weak"

    df.loc[
        (df["Immunogenicity Score"] > 0.7)
        & (df["Best IC50 Score"] < 150)
        & (df["Median IC50 Score"] < 150)
        & (df["Best Percentile"] < 0.5)
        & (df["Median Percentile"] < 0.5)
        & (df["TPM_tumor"] > 7)
        & (df["log2FC"] > 5),
        "Tier",
    ] = "Moderate"

    df.loc[
        (df["Immunogenicity Score"] > 0.9)
        & (df["Best IC50 Score"] < 20)
        & (df["Median IC50 Score"] < 20)
        & (df["Best Percentile"] < 0.1)
        & (df["Median Percentile"] < 0.1)
        & (df["TPM_tumor"] > 10)
        & (df["log2FC"] > 6),
        "Tier",
    ] = "High"

    df["Tier"] = pd.Categorical(df["Tier"], categories=ORDER, ordered=True)
    return df


def assign_tiers(cryptic_epitopes_annot_aafeatures_inferenced: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cryptic = cryptic_epitopes_annot_aafeatures_inferenced.copy()

    mask_i = cryptic["HLA Allele"].astype(str).apply(is_class_i)
    mhci_raw = cryptic[mask_i].copy()
    mhcii_raw = cryptic[~mask_i].copy()

    mhci = rank_mhci(mhci_raw)
    mhcii = rank_mhcii_strict(mhcii_raw)

    mhci["Score"] = mhci["Tier"].astype(str)
    mhcii["Score"] = mhcii["Tier"].astype(str)

    mhci = mhci.sort_values("Tier")
    mhcii = mhcii.sort_values("Tier")
    return mhci, mhcii


def parse_args():
    ap = argparse.ArgumentParser(
        description="Calculate cryptic epitope features, infer immunogenicity, and assign HLA-I/HLA-II tiers."
    )
    ap.add_argument("-s", "--sample", required=True, help="Sample ID")
    ap.add_argument("--input", required=True, help="Step-09 cryptic_epitopes_annot_extend.csv")
    ap.add_argument("-o", "--outdir", required=True, help="Output directory")
    ap.add_argument("--model-path", required=True, help="Noncoding immunogenicity model .pth")
    ap.add_argument("--hla-prot-fasta", required=True, help="HLA protein FASTA used by the model")
    ap.add_argument("--r-home", help="Optional R_HOME for rpy2")
    ap.add_argument("--r-lib-path", help="Optional R library path containing Peptides")
    ap.add_argument("--threads", type=int, default=35, help="Worker/thread count")
    ap.add_argument("--feature-chunks", type=int, help="Number of chunks for R Peptides feature calculation")
    ap.add_argument("--batch-size", type=int, default=512, help="Inference batch size")
    ap.add_argument(
        "--log-file",
        default="10-cryptic_epitope_immunogenicity_tiering.log",
        help="Log filename under --outdir, or an absolute/relative path",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    sample = args.sample
    input_path = Path(args.input)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    for path, label in [
        (input_path, "step-09 cryptic epitope annotation table"),
        (args.model_path, "noncoding immunogenicity model"),
        (args.hla_prot_fasta, "HLA protein FASTA"),
    ]:
        check_file(path, label)

    setup_r_environment(args.r_home, args.r_lib_path)

    log_path = Path(args.log_file)
    if not log_path.is_absolute():
        log_path = outdir / log_path
    ensure_dir(log_path.parent)

    with log_path.open("w") as log_fh:
        def log_both(message: str) -> None:
            log(message)
            log_fh.write(f"[{ts()}] {message}\n")
            log_fh.flush()

        log_both("10-cryptic_epitope_immunogenicity_tiering.py started")
        log_both(f"sample: {sample}")
        log_both(f"input: {input_path}")
        log_both(f"outdir: {outdir}")
        log_both(f"threads: {args.threads}")
        log_both(f"batch_size: {args.batch_size}")

        cryptic_epitopes_annot_extend = pd.read_csv(input_path, low_memory=False)
        log_both(f"input rows: {len(cryptic_epitopes_annot_extend)}")

        feature_chunks = args.feature_chunks or args.threads
        peptides = cryptic_epitopes_annot_extend[["Epitope Seq"]].drop_duplicates(subset="Epitope Seq")
        log_both(f"unique peptides for feature calculation: {len(peptides)}")
        peptides_with_features = parallel_calculate_features(peptides, "Epitope Seq", num_chunks=feature_chunks)
        cryptic_epitopes_annot_aafeatures = pd.merge(
            cryptic_epitopes_annot_extend, peptides_with_features, on="Epitope Seq", how="left"
        )
        cryptic_epitopes_annot_aafeatures.to_csv(outdir / "cryptic_epitopes_annot_aafeatures.csv", index=False)
        log_both(f"feature table rows: {len(cryptic_epitopes_annot_aafeatures)}")

        imm_probs_df = infer_cryptic(
            cryptic_epitopes_annot_aafeatures,
            args.model_path,
            args.hla_prot_fasta,
            args.threads,
            args.batch_size,
        )
        cryptic_epitopes_annot_aafeatures_inferenced = pd.merge(cryptic_epitopes_annot_aafeatures, imm_probs_df)
        cryptic_epitopes_annot_aafeatures_inferenced.to_csv(
            outdir / "cryptic_epitopes_annot_aafeatures_inferenced.csv", index=False
        )
        log_both(f"inferenced table rows: {len(cryptic_epitopes_annot_aafeatures_inferenced)}")

        mhci, mhcii = assign_tiers(cryptic_epitopes_annot_aafeatures_inferenced)
        mhci.to_csv(outdir / "cryptic_peptides_annot_tier_HLA-I.csv", index=False)
        mhcii.to_csv(outdir / "cryptic_peptides_annot_tier_HLA-II.csv", index=False)
        log_both(f"MHCI rows: {len(mhci)} | Tier counts: {mhci['Tier'].value_counts(dropna=False).to_dict()}")
        log_both(f"MHCII rows: {len(mhcii)} | Tier counts: {mhcii['Tier'].value_counts(dropna=False).to_dict()}")
        log_both("10-cryptic_epitope_immunogenicity_tiering.py finished")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        sys.stderr.write(f"{exc}\n")
        sys.exit(1)
