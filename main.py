# Bio-Chem Molecule Studio — educational demo
# Requires:            streamlit duckdb rdkit py3Dmol stmol
# Optional (ChemGPT):  torch transformers selfies
#   pip install streamlit duckdb rdkit py3Dmol stmol torch transformers selfies
# Split each "# file.py" block into its own module later.


import importlib
import numpy as np
from models import (
    init_duckdb,
    make_cache_key
)

from metrics import (
    smiles_to_molblock,
    compute_descriptors
)


from generate import (
    _seed_ids,
    load_chemgpt,
    generate_smiles,
    GEN_AVAILABLE, 
    smiles_to_nomenclature
)



# main.py
import pandas as pd
import py3Dmol
from stmol import showmol
import streamlit as st
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, AllChem

st.set_page_config(
    page_title="ChemGPT Studio",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

db_conn = init_duckdb()


def get_last_valid_sub_smiles(smiles: str):
    RDLogger.DisableLog('rdApp.*')
    
    # Check full string first
    if Chem.MolFromSmiles(smiles): 
        return smiles # Return the STRING, not the Mol

    # Work backwards to find the longest valid prefix
    for idx in range(len(smiles) - 1, 0, -1):
        sub_smiles = smiles[:idx]
        if Chem.MolFromSmiles(sub_smiles): 
            return sub_smiles # Return the STRING
            
    return None

SAMPLE_PRESETS = {
    "Caffeine (everyday molecule)": {
        "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "target": "Adenosine receptor (illustrative)",
        "desc": "Small and very drug-like — a friendly reference point.",
    },
    "Aspirin (classic drug)": {
        "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "target": "Cyclooxygenase / COX (illustrative)",
        "desc": "A textbook small-molecule drug; comfortably passes Lipinski's Rule of 5.",
    },
    "Ibuprofen (common NSAID)": {
        "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "target": "Cyclooxygenase / COX (illustrative)",
        "desc": "Another well-behaved oral drug — a good seed or comparison.",
    },
    "Kinase inhibitor (larger candidate)": {
        "smiles": "CN1CCN(CC1)CC2=C(C=C3C(=C2)C(=NC=N3)NC4=CC(=C(C=C4)Cl)Cl)C#C",
        "target": "Protein kinase (illustrative)",
        "desc": "Bigger and more complex — watch how the Lipinski verdict changes.",
    },
    "Ethanol (too-small example)": {
        "smiles": "CCO",
        "target": "",
        "desc": "Deliberately tiny — shows what 'not really drug-like' looks like.",
    },
}

# Session-state defaults so examples can drive the inputs reliably.
st.session_state.setdefault("smiles_input", SAMPLE_PRESETS["Aspirin (classic drug)"]['smiles'])
st.session_state.setdefault("target_input", "Aspirin. " + SAMPLE_PRESETS["Aspirin (classic drug)"]['desc'])


def apply_preset():
    choice = st.session_state["preset_choice"]
    if choice != "Custom input":
        st.session_state["smiles_input"] = SAMPLE_PRESETS[choice]["smiles"]
        st.session_state["target_input"] = SAMPLE_PRESETS[choice]["target"]


def render_results(smiles, target, source):
    """Cache lookup + descriptor cards + Lipinski verdict + local 3D."""
    parsed = compute_descriptors(smiles)
    if parsed is None:
        st.error("That SMILES string could not be parsed. Check the notation and try again.")
        return

    canonical = parsed["canonical"]
    target_context = (target or "").strip()[:60] or "—"
    cache_id = make_cache_key(canonical, target_context)
    row = db_conn.execute(
        "SELECT * FROM molecule_stage WHERE cache_hash = ?", (cache_id,)
    ).fetchone()

    if row:
        desc = {
            "mw": row[4], "logp": row[5], "hbd": row[6], "hba": row[7],
            "tpsa": row[8], "rot_bonds": row[9], "rings": row[10],
            "heavy_atoms": row[11], "lipinski_violations": row[12],
        }
        note = "⚡ loaded from cache"
    else:
        desc = parsed
        db_conn.execute(
            "INSERT OR REPLACE INTO molecule_stage VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                cache_id, canonical, target_context, source,
                desc["mw"], desc["logp"], desc["hbd"], desc["hba"],
                desc["tpsa"], desc["rot_bonds"], desc["rings"],
                desc["heavy_atoms"], desc["lipinski_violations"],
            ),
        )
        note = "computed & cached"

    st.subheader("Results")
    st.caption(f"Properties of the molecule analyzed · source: {source} · canonical form: `{canonical}` · {note}")

    m = st.columns(6)
    m[0].metric("Mol. Weight (g/mol)", desc["mw"])
    m[1].metric("LogP (calc.)", desc["logp"])
    m[2].metric("H-Donors", desc["hbd"])
    m[3].metric("H-Acceptors", desc["hba"])
    m[4].metric("TPSA (Å²)", desc["tpsa"])
    m[5].metric("Rotatable Bonds", desc["rot_bonds"])

    v = desc["lipinski_violations"]
    if v == 0:
        st.success("✅ Meets Lipinski's Rule of 5 (0 violations) — a rough oral-bioavailability heuristic, not proof a molecule is a viable drug.")
    else:
        st.warning(f"⚠️ {v} Lipinski Rule-of-5 violation(s) — a heuristic flag for oral dosing, not a verdict on the molecule.")

    st.markdown("**Interactive 3D structure**")
    st.caption("One quickly generated low-energy 3D conformer (RDKit) — illustrative, not necessarily the bioactive conformation.")
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        molblock = smiles_to_molblock(smiles)
        if molblock:
            view = py3Dmol.view(width=460, height=420)
            view.addModel(molblock, "mol")
            view.setStyle({"stick": {"radius": 0.13, "colorscheme": "Jmol"},
                           "sphere": {"scale": 0.22}})
            view.setBackgroundColor("white")
            view.zoomTo()
            showmol(view, height=420, width=460)
        else:
            st.info("3D coordinates could not be generated for this molecule.")


# --- Sidebar -----------------------------------------------------------------
with st.sidebar:
    st.header("Examples")
    st.selectbox(
        "Load an example molecule:",
        ["Custom input"] + list(SAMPLE_PRESETS.keys()),
        key="preset_choice",
        on_change=apply_preset,
        help="Fills the molecule below — used both for analysis and as a generation seed.",
    )
    if st.session_state.get("preset_choice", "Custom input") != "Custom input":
        st.info(SAMPLE_PRESETS[st.session_state["preset_choice"]]["desc"])

    with st.expander("Metric glossary"):
        st.markdown("""
        - **Molecular Weight** — < 500 g/mol is typical for oral drugs.
        - **LogP** — fat- vs. water-loving; sweet spot ~1–3.
        - **TPSA** — polar surface area; < 140 Å² aids absorption.
        - **H-Donors / Acceptors** — Lipinski's Rule of 5: < 5 donors, < 10 acceptors.
        - **Rotatable Bonds** — molecular flexibility; < 10 preferred.
        """)
    with st.expander("Privacy"):
        st.markdown("""
        - Descriptors and 3D coordinates are computed **locally** (RDKit) — they never leave your machine.
        - ChemGPT generation downloads the model weights once from Hugging Face, then runs locally on CPU.
        - Results are cached in an in-memory DuckDB database, wiped when the app stops.
        - **Note: PubChem receives requests to get the nomenclature of generated and pasted molecules.**
        """)

# --- RDKit guard -------------------------------------------------------------
try:
    Chem.MolFromSmiles("C")
except Exception:
    st.error("This app requires RDKit:  `pip install rdkit`")
    st.stop()

# --- Header ------------------------------------------------------------------
st.title("🧪 ChemGPT Studio")
st.caption("Compute a molecule's physicochemical and drug-likeness properties and view its 3D structure — or sample new molecules with a small transformer (ChemGPT) and analyze those.")
st.caption("🎓 Educational demo · descriptors via RDKit, generation via ChemGPT · not for clinical or regulatory use.")

with st.expander("How to use this & what the numbers mean"):
    st.markdown("""
    1. Pick an **example** (sidebar) or paste a **SMILES** below.
    2. Click **Analyze this molecule** to see its properties + 3D structure — *or* use **Generate** to have ChemGPT sample new molecules seeded from it, then analyze any candidate.

    **SMILES** is a text way to write a molecule, e.g. `CCO` (ethanol) or `c1ccccc1` (benzene). The **Lipinski Rule of 5** is a quick rule of thumb for whether a molecule could work as an oral drug. Generation is the only step that creates new molecules; everything else describes the molecule you analyze.
    """)

# --- Molecule input ----------------------------------------------------------
def handle_analyze_click(): # we need this callback just in case input is invalid...
    # Grab the string directly from session state
    smiles_base = str(st.session_state['smiles_input']).strip() 
    saved_smiles = get_last_valid_sub_smiles(smiles_base)
    
    if saved_smiles is None:
        st.error("Could not find any valid SMILES structure in your input.")
    else:
        # Save the valid string to active state
        st.session_state["active"] = {
            "smiles": saved_smiles,
            "target": st.session_state.get("target_input", ""),
            "source": "pasted",
        }
        
        # If we had to truncate it, warn the user AND update the input box
        if saved_smiles != smiles_base:
            st.warning(f"Invalid SMILES string - truncated to last valid part of SMILES STRING {saved_smiles}.")
            st.warning("NOTE: Chemical Properties May Significantly Differ")
            # This is now safe because the callback runs before the widget renders!
            st.session_state["smiles_input"] = saved_smiles
        else:
            st.toast("Successfully Validated SMILES String! See below for analysis.")

# 2. Draw the UI
in_col, btn_col = st.columns([4, 1])
with in_col:
    st.text_input("Molecule — SMILES string", key="smiles_input",
                  help="e.g. CCO (ethanol) or c1ccccc1 (benzene).")
    st.text_input("Context (optional annotation)", key="target_input",
                  help="Logged with the result; does not change the computed properties.")
with btn_col:
    st.write("")
    st.write("")
    # 3. Attach the callback using on_click (remove the if statement)
    st.button("Analyze this molecule (no generation)", 
              type="primary", 
              use_container_width=True, 
              on_click=handle_analyze_click)
# --- Generation (ChemGPT) ----------------------------------------------------
with st.container(border=True):
    st.markdown("**✨ Generate new candidates with ChemGPT** (optional)")
    st.caption("Seeded from the molecule above (clear the box for unseeded sampling). Generation is *de novo* — not conditioned on the target.")
    if not GEN_AVAILABLE():
        st.warning("Generation needs extra packages:  `pip install torch transformers selfies`")

    g1, g2, g3, g4 = st.columns([2, 1, 1, 1])
    model_name = g1.selectbox("ChemGPT model)",
                              ["ncfrey/ChemGPT-19M", "ncfrey/ChemGPT-4.7M"])
    n_cands = g2.slider("Candidates", 4, 100, 50)
    temp = g3.slider("Temperature", 0.5, 1.5, 1.0, 0.1)
    max_tok = g4.slider("Max tokens", 16, 128, 64, 8)

    if st.button("Generate Molecules", type="secondary",
                 disabled=not GEN_AVAILABLE(), width="stretch"):
        with st.spinner("Loading ChemGPT and sampling molecules (first run downloads the model)…"):
            try:
                st.session_state["gen_candidates"] = generate_smiles(
                    model_name, n_cands, temp, max_tok, st.session_state["smiles_input"]
                )
            except Exception as e:
                st.session_state["gen_candidates"] = []
                st.error(f"Generation failed: {e}")

    cands = st.session_state.get("gen_candidates")
    if cands is not None:
        if not cands:
            st.warning("No valid molecules were generated. Try more candidates or a different temperature.")
        else:
            st.markdown(f"**{len(cands)} valid candidate(s) generated:**")
            st.caption("Chemically valid samples only — not screened for novelty, synthesizability, or stability.")
            rows = []
        
            for smi in cands:
                nomenclature_attempt = "Could not generate nomenclature..."
                d = compute_descriptors(smi)
                if d:
                    try:
                        print("attempting nomenclature...")
                        nomenclature_attempt = smiles_to_nomenclature(smi)
                        print(nomenclature_attempt)
                    except Exception as e:
                        print("Attempted to get nomenclature of generated molecule but failed....")

                    rows.append({"SMILES": smi, "MW": d["mw"], "LogP": d["logp"],
                                 "Lipinski viol.": d["lipinski_violations"], "Nomenclature": nomenclature_attempt})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

            pick_col, pick_btn = st.columns([4, 1])
            choice = pick_col.selectbox("Pick a candidate:", cands, key="cand_choice")
            with pick_btn:
                st.write("")
                st.write("")
                if st.button("Analyze candidate", width="stretch"):
                    st.session_state["active"] = {
                        "smiles": choice,
                        "target": st.session_state["target_input"],
                        "source": "generated",
                    }

# --- Shared results viewer (gated: only renders an active molecule) ----------
if "active" in st.session_state:
    st.markdown("---")
    a = st.session_state["active"]
    render_results(a["smiles"], a["target"], a["source"])

# --- Analysis log ------------------------------------------------------------
st.markdown("---")
st.subheader("Analysis log")
log_df = db_conn.execute(
    """
    SELECT smiles, source, target_context, mw, logp, hbd, hba, tpsa, lipinski_violations
    FROM molecule_stage
    """
).df()

nomenclature_names = [smiles_to_nomenclature(smile_val) for smile_val in log_df['smiles'].to_list()]
log_df['nomenlature'] = nomenclature_names
log_df['target_context'] = np.where(log_df['source'] == "generated", "GENERATED: " + log_df['target_context'], log_df['target_context'])


log_df = log_df.rename(columns={
    "smiles": "SMILES", "nomenclature":"Nomenclature", "source": "Source", "target_context": "Context",
    "mw": "MW", "logp": "LogP", "hbd": "H-Donors", "hba": "H-Acceptors",
    "tpsa": "TPSA", "lipinski_violations": "Lipinski Violations",
})
if not log_df.empty:
    st.dataframe(log_df, width="stretch", hide_index=True)
else:
    st.caption("No molecules analyzed yet.")
