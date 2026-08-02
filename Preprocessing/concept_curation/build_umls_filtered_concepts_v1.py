import argparse
import ast
import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

import spacy
import scispacy
import scispacy.linking
from spacy.language import Language


# -----------------------------
# 0. Optional: force scispaCy to use local linker files
# -----------------------------

def patch_scispacy_cached_path(local_linker_dir: str):
    """
    Force scispaCy UMLS linker to use local files instead of downloading from S3.

    local_linker_dir should contain:
        tfidf_vectors_sparse.npz
        nmslib_index.bin
        tfidf_vectorizer.joblib
        concept_aliases.json
        umls_2022_ab_cat0129.jsonl
        umls_semantic_type_tree.tsv

    This patch is needed because scispaCy internally calls cached_path()
    to download linker resources from S3. We intercept those calls and
    return local file paths instead.
    """
    local_linker_dir = os.path.abspath(local_linker_dir)

    required_files = {
        "tfidf_vectors_sparse.npz": os.path.join(local_linker_dir, "tfidf_vectors_sparse.npz"),
        "nmslib_index.bin": os.path.join(local_linker_dir, "nmslib_index.bin"),
        "tfidf_vectorizer.joblib": os.path.join(local_linker_dir, "tfidf_vectorizer.joblib"),
        "concept_aliases.json": os.path.join(local_linker_dir, "concept_aliases.json"),
        "umls_2022_ab_cat0129.jsonl": os.path.join(local_linker_dir, "umls_2022_ab_cat0129.jsonl"),
        "umls_semantic_type_tree.tsv": os.path.join(local_linker_dir, "umls_semantic_type_tree.tsv"),
    }

    if not os.path.isdir(local_linker_dir):
        raise FileNotFoundError(
            f"Local UMLS linker directory does not exist:\n{local_linker_dir}"
        )

    missing_files = []
    for filename, filepath in required_files.items():
        if not os.path.exists(filepath):
            missing_files.append((filename, filepath))

    if missing_files:
        msg = ["Missing local scispaCy UMLS linker files:"]
        for filename, filepath in missing_files:
            msg.append(f"  - {filename}: expected at {filepath}")
        msg.append("\nPlease put all required files in the same directory.")
        raise FileNotFoundError("\n".join(msg))

    import scispacy.file_cache as file_cache

    # candidate_generation imports cached_path directly, so patching only
    # scispacy.file_cache.cached_path may not be enough.
    import scispacy.candidate_generation as candidate_generation
    import scispacy.linking_utils as linking_utils
    import scispacy.umls_semantic_type_tree as umls_semantic_type_tree

    original_cached_path = file_cache.cached_path

    def cached_path_local(url_or_filename, *args, **kwargs):
        url_str = str(url_or_filename)

        for filename, local_path in required_files.items():
            if filename in url_str:
                print(f"[INFO] Using local scispaCy linker file for {filename}: {local_path}")
                return local_path

        if "ai2-s2-scispacy/data/linkers" in url_str or "/data/linkers/" in url_str:
            raise RuntimeError(
                "Local UMLS linker mode is enabled, but scispaCy requested an unexpected "
                f"linker resource:\n{url_str}\n"
                f"Local linker directory: {local_linker_dir}\n"
                "Please make sure the directory contains the complete linker files "
                "for the installed scispaCy version."
            )

        return original_cached_path(url_or_filename, *args, **kwargs)

    file_cache.cached_path = cached_path_local
    candidate_generation.cached_path = cached_path_local
    linking_utils.cached_path = cached_path_local
    umls_semantic_type_tree.cached_path = cached_path_local

    print(f"[INFO] Patched scispaCy cached_path.")
    print(f"[INFO] Local UMLS linker directory: {local_linker_dir}")


# -----------------------------
# 1. UMLS semantic type settings
# -----------------------------

# UMLS TUI codes useful for H&E WSI morphology / diagnosis.
KEEP_TUIS = {
    # Diagnosis-related
    "T191",  # Neoplastic Process
    "T047",  # Disease or Syndrome
    "T046",  # Pathologic Function
    "T033",  # Finding

    # Morphology / tissue context
    "T025",  # Cell
    "T024",  # Tissue
    "T023",  # Body Part, Organ, or Organ Component
    "T029",  # Body Location or Region
}

# Semantic types usually not directly visible in H&E WSI morphology.
DROP_TUIS = {
    "T028",  # Gene or Genome
    "T116",  # Amino Acid, Peptide, or Protein
    "T121",  # Pharmacologic Substance
    "T109",  # Organic Chemical
    "T061",  # Therapeutic or Preventive Procedure
    "T059",  # Laboratory Procedure
    "T074",  # Medical Device
}

TUI_NAME = {
    "T191": "Neoplastic Process",
    "T047": "Disease or Syndrome",
    "T046": "Pathologic Function",
    "T033": "Finding",
    "T025": "Cell",
    "T024": "Tissue",
    "T023": "Body Part, Organ, or Organ Component",
    "T029": "Body Location or Region",
    "T028": "Gene or Genome",
    "T116": "Amino Acid, Peptide, or Protein",
    "T121": "Pharmacologic Substance",
    "T109": "Organic Chemical",
    "T061": "Therapeutic or Preventive Procedure",
    "T059": "Laboratory Procedure",
    "T074": "Medical Device",
}


# -----------------------------
# 2. Text cleaning
# -----------------------------

def clean_concept(text: str) -> str:
    """Basic cleaning for Quilt-1M ROI concepts."""
    text = str(text).strip().lower()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" ,.;:()[]{}")
    return text


def is_valid_concept(text: str, min_len: int = 3, max_len: int = 120) -> bool:
    """Remove empty, too short, or abnormal concepts."""
    if not text:
        return False
    if len(text) < min_len or len(text) > max_len:
        return False
    if text in {"nan", "none", "null", "[]"}:
        return False
    if not re.search(r"[a-zA-Z]", text):
        return False
    return True


# -----------------------------
# 3. Load concepts from Quilt-1M
# -----------------------------

def load_quilt_concepts(
    csv_path: str,
    roi_col: str = "roi_text",
) -> Tuple[List[str], Counter, Dict[str, List[int]]]:
    """
    Load raw concepts from Quilt-1M lookup csv.

    Returns:
        raw_concepts: unique cleaned concept list
        concept_counter: frequency of each concept
        concept_roi_indices: ROI row indices where each concept occurs
    """
    print(f"[INFO] Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    if roi_col not in df.columns:
        raise ValueError(
            f"Column '{roi_col}' not found in CSV. Available columns: {list(df.columns)}"
        )

    df.dropna(subset=[roi_col], inplace=True)
    df = df[df[roi_col] != "[]"]

    concept_counter = Counter()
    concept_roi_indices = defaultdict(list)

    for row_idx, value in tqdm(
        zip(df.index.tolist(), df[roi_col].tolist()),
        total=len(df),
        desc="Extracting concepts from roi_text",
    ):
        try:
            terms = ast.literal_eval(value)
        except Exception:
            continue

        if not isinstance(terms, list):
            continue

        for term in terms:
            term = clean_concept(term)
            if not is_valid_concept(term):
                continue

            concept_counter[term] += 1
            concept_roi_indices[term].append(int(row_idx))

    raw_concepts = sorted(concept_counter.keys())

    print(f"[INFO] Raw unique concepts: {len(raw_concepts)}")
    print(f"[INFO] Total concept mentions: {sum(concept_counter.values())}")

    return raw_concepts, concept_counter, concept_roi_indices


# -----------------------------
# 4. Load scispaCy + UMLS linker
# -----------------------------

def load_umls_linker(
    model_name: str = "en_core_sci_sm",
    threshold: float = 0.80,
):
    """
    Load scispaCy model and UMLS linker.

    threshold:
        Higher threshold = stricter UMLS mapping.
        Suggested range: 0.75 - 0.90.
    """
    print(f"[INFO] Loading scispaCy model: {model_name}")
    nlp = spacy.load(model_name)

    print("[INFO] Adding UMLS linker...")
    if not Language.has_factory("scispacy_linker"):
        available = ", ".join(sorted(Language.factories.keys()))
        raise RuntimeError(
            "The scispaCy pipe factory 'scispacy_linker' is not registered. "
            "This usually means the installed spaCy/scispaCy versions are incompatible "
            "or scispaCy was not installed correctly.\n"
            f"Detected versions: spaCy=={spacy.__version__}, scispaCy=={scispacy.__version__}\n"
            f"Currently available factories: {available}"
        )

    nlp.add_pipe(
        "scispacy_linker",
        config={
            "resolve_abbreviations": False,
            "linker_name": "umls",
            "threshold": threshold,
            "max_entities_per_mention": 5,
        },
    )

    linker = nlp.get_pipe("scispacy_linker")
    print("[INFO] UMLS linker loaded successfully.")
    return nlp, linker


# -----------------------------
# 5. UMLS normalization
# -----------------------------

def map_one_concept_to_umls(
    doc,
    linker,
    original_term: str,
) -> Optional[dict]:
    """
    Map one concept string to the best UMLS CUI.

    Returns None if no UMLS entity is found.
    """
    best = None

    for ent in doc.ents:
        if not hasattr(ent._, "kb_ents") or len(ent._.kb_ents) == 0:
            continue

        for cui, score in ent._.kb_ents:
            try:
                umls_entity = linker.kb.cui_to_entity[cui]
            except KeyError:
                continue

            umls_types = list(umls_entity.types)

            item = {
                "original_term": original_term,
                "matched_span": ent.text,
                "cui": cui,
                "umls_name": umls_entity.canonical_name,
                "umls_aliases": list(umls_entity.aliases)[:20],
                "umls_definition": umls_entity.definition,
                "umls_types": umls_types,
                "umls_type_names": [TUI_NAME.get(t, t) for t in umls_types],
                "umls_score": float(score),
            }

            if best is None or item["umls_score"] > best["umls_score"]:
                best = item

    return best


def normalize_concepts_with_umls(
    raw_concepts: List[str],
    concept_counter: Counter,
    nlp,
    linker,
    batch_size: int = 128,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run UMLS normalization for all raw concepts.

    Returns:
        mapped_df: concepts successfully mapped to UMLS
        unmatched_df: concepts not mapped to UMLS
    """
    mapped_records = []
    unmatched_records = []

    print("[INFO] Mapping concepts to UMLS...")

    for doc, term in tqdm(
        zip(nlp.pipe(raw_concepts, batch_size=batch_size), raw_concepts),
        total=len(raw_concepts),
        desc="UMLS normalization",
    ):
        item = map_one_concept_to_umls(doc, linker, term)

        if item is None:
            unmatched_records.append(
                {
                    "original_term": term,
                    "freq": int(concept_counter[term]),
                    "reason": "no_umls_mapping",
                }
            )
            continue

        item["freq"] = int(concept_counter[term])
        mapped_records.append(item)

    mapped_df = pd.DataFrame(mapped_records)
    unmatched_df = pd.DataFrame(unmatched_records)

    print(f"[INFO] UMLS mapped concepts: {len(mapped_df)}")
    print(f"[INFO] Unmatched concepts: {len(unmatched_df)}")

    return mapped_df, unmatched_df


# -----------------------------
# 6. Semantic type filtering
# -----------------------------

def should_keep_by_semantic_type(types: List[str]) -> Tuple[bool, str]:
    """
    Decide whether to keep a concept according to UMLS semantic types.

    Rule:
        1. Drop if it has any DROP_TUIS.
        2. Keep if it has any KEEP_TUIS.
        3. Otherwise drop.
    """
    type_set = set(types)

    if type_set & DROP_TUIS:
        return False, "drop_type"

    if type_set & KEEP_TUIS:
        return True, "keep_type"

    return False, "unrelated_type"


def apply_semantic_type_filter(
    mapped_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter UMLS-mapped concepts by semantic types."""
    if mapped_df.empty:
        print("[WARN] mapped_df is empty. No concepts to filter.")
        return mapped_df.copy(), mapped_df.copy()

    keep_flags = []
    reasons = []

    for _, row in mapped_df.iterrows():
        types = row.get("umls_types", [])

        if isinstance(types, str):
            try:
                types = ast.literal_eval(types)
            except Exception:
                types = []

        keep, reason = should_keep_by_semantic_type(types)
        keep_flags.append(keep)
        reasons.append(reason)

    filtered = mapped_df.copy()
    filtered["semantic_filter_keep"] = keep_flags
    filtered["semantic_filter_reason"] = reasons

    kept_df = filtered[filtered["semantic_filter_keep"]].copy()
    dropped_df = filtered[~filtered["semantic_filter_keep"]].copy()

    print(f"[INFO] Kept after semantic type filtering: {len(kept_df)}")
    print(f"[INFO] Dropped after semantic type filtering: {len(dropped_df)}")

    return kept_df, dropped_df


# -----------------------------
# 7. CUI deduplication
# -----------------------------

def deduplicate_by_cui(kept_df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate concepts by UMLS CUI.

    For each CUI, keep the original term with the highest:
        freq * umls_score

    This preserves a human-readable Quilt-1M term while using UMLS for standardization.
    """
    if kept_df.empty:
        print("[WARN] kept_df is empty. No concepts to deduplicate.")
        return kept_df.copy()

    df = kept_df.copy()
    df["dedup_score"] = df["freq"].astype(float) * df["umls_score"].astype(float)

    df = df.sort_values(
        by=["cui", "dedup_score", "freq", "umls_score"],
        ascending=[True, False, False, False],
    )

    dedup_df = df.groupby("cui", as_index=False).head(1).copy()
    dedup_df = dedup_df.sort_values(
        by=["freq", "umls_score"],
        ascending=[False, False],
    ).reset_index(drop=True)

    print(f"[INFO] Concepts after CUI deduplication: {len(dedup_df)}")

    return dedup_df


# -----------------------------
# 8. Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--roi_col", type=str, default="roi_text")
    parser.add_argument("--output_dir", type=str, default="./umls_concepts")
    parser.add_argument("--spacy_model", type=str, default="en_core_sci_sm")
    parser.add_argument("--umls_threshold", type=float, default=0.80)
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--local_umls_linker_dir",
        type=str,
        default=None,
        help=(
            "Local directory containing scispaCy UMLS linker files. "
            "Expected files: tfidf_vectors_sparse.npz, nmslib_index.bin, "
            "tfidf_vectorizer.joblib, concept_aliases.json, "
            "umls_2022_ab_cat0129.jsonl, umls_semantic_type_tree.tsv. "
            "When provided, linker resource downloads are disabled and only local files are used."
        ),
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Save arguments for reproducibility.
    with open(os.path.join(args.output_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    # Step 1: Load raw concepts.
    raw_concepts, concept_counter, concept_roi_indices = load_quilt_concepts(
        csv_path=args.csv_path,
        roi_col=args.roi_col,
    )

    # Save raw frequency table.
    raw_freq_df = pd.DataFrame(
        [{"concept": c, "freq": int(concept_counter[c])} for c in raw_concepts]
    ).sort_values("freq", ascending=False)

    raw_freq_path = os.path.join(args.output_dir, "raw_concepts_frequency.csv")
    raw_freq_df.to_csv(raw_freq_path, index=False)
    print(f"[INFO] Raw concept frequency saved to: {raw_freq_path}")

    # Step 2: Patch scispaCy cache if local linker directory is provided.
    if args.local_umls_linker_dir is not None:
        patch_scispacy_cached_path(args.local_umls_linker_dir)

    # Step 3: Load UMLS linker.
    nlp, linker = load_umls_linker(
        model_name=args.spacy_model,
        threshold=args.umls_threshold,
    )

    # Step 4: UMLS normalization.
    mapped_df, unmatched_df = normalize_concepts_with_umls(
        raw_concepts=raw_concepts,
        concept_counter=concept_counter,
        nlp=nlp,
        linker=linker,
        batch_size=args.batch_size,
    )

    mapped_path = os.path.join(args.output_dir, "concepts_umls_mapped_all.csv")
    unmatched_path = os.path.join(args.output_dir, "concepts_umls_unmatched.csv")

    mapped_df.to_csv(mapped_path, index=False)
    unmatched_df.to_csv(unmatched_path, index=False)

    print(f"[INFO] UMLS mapped concepts saved to: {mapped_path}")
    print(f"[INFO] UMLS unmatched concepts saved to: {unmatched_path}")

    # Step 5: Semantic type filtering.
    kept_df, semantic_dropped_df = apply_semantic_type_filter(mapped_df)

    kept_path = os.path.join(args.output_dir, "concepts_semantic_type_kept.csv")
    dropped_path = os.path.join(args.output_dir, "concepts_semantic_type_dropped.csv")

    kept_df.to_csv(kept_path, index=False)
    semantic_dropped_df.to_csv(dropped_path, index=False)

    print(f"[INFO] Semantic-type kept concepts saved to: {kept_path}")
    print(f"[INFO] Semantic-type dropped concepts saved to: {dropped_path}")

    # Step 6: CUI deduplication.
    dedup_df = deduplicate_by_cui(kept_df)

    # Step 7: Build final table.
    if dedup_df.empty:
        final_df = pd.DataFrame(
            columns=[
                "concept",
                "cui",
                "umls_name",
                "umls_types",
                "umls_type_names",
                "freq",
                "umls_score",
                "dedup_score",
            ]
        )
    else:
        final_df = dedup_df[
            [
                "original_term",
                "cui",
                "umls_name",
                "umls_types",
                "umls_type_names",
                "freq",
                "umls_score",
                "dedup_score",
            ]
        ].copy()

        final_df = final_df.rename(columns={"original_term": "concept"})

    final_path = os.path.join(
        args.output_dir,
        "concepts_umls_cui_dedup_semantic_filtered.csv",
    )
    final_df.to_csv(final_path, index=False)

    # Save concept list only.
    concept_list_path = os.path.join(args.output_dir, "final_concept_list.txt")
    with open(concept_list_path, "w", encoding="utf-8") as f:
        for concept in final_df["concept"].dropna().astype(str).tolist():
            f.write(concept + "\n")

    # Save summary.
    summary = {
        "raw_unique_concepts": len(raw_concepts),
        "total_concept_mentions": int(sum(concept_counter.values())),
        "umls_mapped_concepts": len(mapped_df),
        "umls_unmatched_concepts": len(unmatched_df),
        "semantic_type_kept_before_cui_dedup": len(kept_df),
        "semantic_type_dropped": len(semantic_dropped_df),
        "final_cui_dedup_concepts": len(final_df),
        "umls_threshold": args.umls_threshold,
        "spacy_model": args.spacy_model,
        "local_umls_linker_dir": args.local_umls_linker_dir,
        "keep_tuis": sorted(list(KEEP_TUIS)),
        "drop_tuis": sorted(list(DROP_TUIS)),
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n========== DONE ==========")
    print(json.dumps(summary, indent=2))
    print(f"\n[INFO] Final concept csv saved to: {final_path}")
    print(f"[INFO] Final concept list saved to: {concept_list_path}")
    print(f"[INFO] Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
