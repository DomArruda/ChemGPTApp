# Bio-Chem Molecule Studio — educational demo
# Requires:            streamlit duckdb rdkit py3Dmol stmol
# Optional (ChemGPT):  torch transformers selfies
#   pip install streamlit duckdb rdkit py3Dmol stmol torch transformers selfies


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

# --- Typography + light styling ----------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stMarkdown, .stMetric,
    input, textarea, button, select,
    [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: -0.015em;
    }
    h1 { font-weight: 700 !important; letter-spacing: -0.025em; }

    div[data-testid="stMetric"] {
        background-color: rgba(120, 120, 120, 0.06);
        border-radius: 8px;
        padding: 10px 4px 6px 4px;
    }
    button[kind="primary"] { font-weight: 600; }
    .stTabs [data-baseweb="tab"] { font-size: 1.02rem; padding: 8px 18px; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

db_conn = init_duckdb()


def get_last_valid_sub_smiles(smiles: str):
    RDLogger.DisableLog('rdApp.*')

    # Check full string first
    if Chem.MolFromSmiles(smiles):
        return smiles  # Return the STRING, not the Mol

    # Work backwards to find the longest valid prefix
    for idx in range(len(smiles) - 1, 0, -1):
        sub_smiles = smiles[:idx]
        if Chem.MolFromSmiles(sub_smiles):
            return sub_smiles  # Return the STRING

    return None


def is_valid_smiles(smiles: str) -> bool:
    RDLogger.DisableLog('rdApp.*')
    if not smiles:
        return False
    return Chem.MolFromSmiles(smiles) is not None


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
st.session_state.setdefault("generate_seed_input", "")
st.session_state.setdefault("generate_context_input", "")


def apply_preset():
    choice = st.session_state["preset_choice"]
    if choice != "Custom input":
        st.session_state["smiles_input"] = SAMPLE_PRESETS[choice]["smiles"]
        st.session_state["target_input"] = SAMPLE_PRESETS[choice]["target"]


def apply_generate_seed_source():
    choice = st.session_state["generate_seed_choice"]
    if choice == "No seed (unseeded sampling)":
        st.session_state["generate_seed_input"] = ""
    elif choice == "Copy from Analyze tab":
        st.session_state["generate_seed_input"] = st.session_state.get("smiles_input", "")
    elif choice in SAMPLE_PRESETS:
        st.session_state["generate_seed_input"] = SAMPLE_PRESETS[choice]["smiles"]
    # "Custom — paste below" leaves the current text box value untouched.


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
        note = "loaded from cache"
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

    source_label = "pasted by you" if source == "pasted" else "generated by ChemGPT"
    st.caption(f"**Source:** {source_label} · **Canonical SMILES:** `{canonical}` · {note}")

    st.markdown("##### Physicochemical properties")
    m = st.columns(6)
    m[0].metric("Mol. Weight (g/mol)", desc["mw"])
    m[1].metric("LogP (calc.)", desc["logp"])
    m[2].metric("H-Donors", desc["hbd"])
    m[3].metric("H-Acceptors", desc["hba"])
    m[4].metric("TPSA (Å²)", desc["tpsa"])
    m[5].metric("Rotatable Bonds", desc["rot_bonds"])

    v = desc["lipinski_violations"]
    if v == 0:
        st.success("**Meets Lipinski's Rule of 5** (0 violations) — a rough oral-bioavailability heuristic, not proof a molecule is a viable drug.")
    else:
        st.warning(f"**{v} Lipinski Rule-of-5 violation(s)** — a heuristic flag for oral dosing, not a verdict on the molecule.")

    st.markdown("##### Interactive 3D structure")
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
        help="Fills the Analyze tab's molecule box below.",
    )
    if st.session_state.get("preset_choice", "Custom input") != "Custom input":
        st.info(SAMPLE_PRESETS[st.session_state["preset_choice"]]["desc"])

    st.divider()
    st.markdown("**What this app does**")
    st.markdown("""
    - **Analyze** describes a molecule you provide — its properties and 3D shape. It never invents new molecules.
    - **Generate** samples brand-new molecules with a small transformer, then lets you analyze any of them.
    """)

    with st.expander("Metric glossary"):
        st.markdown("""
        - **Molecular Weight** — < 500 g/mol is typical for oral drugs.
        - **LogP** — fat- vs. water-loving; sweet spot ~1–3.
        - **TPSA** — polar surface area; < 140 Å² aids absorption.
        - **H-Donors / Acceptors** — Lipinski's Rule of 5: < 5 donors, < 10 acceptors.
        - **Rotatable Bonds** — molecular flexibility; < 10 preferred.
        """)
    with st.expander("Where your data goes"):
        st.markdown("""
        - Descriptors and 3D coordinates are computed **locally** (RDKit) — they never leave your machine.
        - ChemGPT generation downloads the model weights once from Hugging Face, then runs locally on CPU.
        - Results are cached in an in-memory DuckDB database, wiped when the app stops.
        - **PubChem receives requests to look up the name of any generated or pasted molecule.**
        """)

# --- RDKit guard -------------------------------------------------------------
try:
    Chem.MolFromSmiles("C")
except Exception:
    st.error("This app requires RDKit:  `pip install rdkit`")
    st.stop()

# --- Header ------------------------------------------------------------------
st.title("ChemGPT Studio")
st.caption("Analyze a molecule's physicochemical and drug-likeness properties and view it in 3D — or sample new molecules with a small transformer (ChemGPT) and analyze those.")
st.caption("Educational demo · descriptors via RDKit, generation via ChemGPT · not for clinical or regulatory use.")

# --- Session snapshot ----------------------------------------------------------
_stats = db_conn.execute(
    "SELECT COUNT(*), SUM(CASE WHEN source = 'generated' THEN 1 ELSE 0 END) FROM molecule_stage"
).fetchone()
_total_logged, _generated_logged = (_stats[0] or 0), (_stats[1] or 0)

s1, s2, s3 = st.columns(3)
s1.metric("Molecules analyzed", _total_logged)
s2.metric("From generation", _generated_logged)
s3.metric("ChemGPT available", "Yes" if GEN_AVAILABLE() else "No")

st.write("")

# Two clearly separated modes, each self-contained: the action and its results
# live together in the same tab so nothing appears in a distant viewer below.
analyze_tab, generate_tab = st.tabs([
    "Analyze a molecule",
    "Generate new molecules",
])


# =============================================================================
# TAB 1 — ANALYZE: input, then properties + 3D appear directly below
# =============================================================================
def handle_analyze_click():  # callback so we can validate before the widget re-renders
    smiles_base = str(st.session_state['smiles_input']).strip()
    saved_smiles = get_last_valid_sub_smiles(smiles_base)

    if saved_smiles is None:
        st.session_state.pop("analyze_active", None)
        st.error("Could not find any valid SMILES structure in your input.")
    else:
        st.session_state["analyze_active"] = {
            "smiles": saved_smiles,
            "target": st.session_state.get("target_input", ""),
            "source": "pasted",
        }

        if saved_smiles != smiles_base:
            st.warning(f"Invalid SMILES string — truncated to last valid part of SMILES string: {saved_smiles}.")
            st.warning("NOTE: Chemical properties may significantly differ.")
            st.session_state["smiles_input"] = saved_smiles
        else:
            st.toast("Successfully validated SMILES string! See analysis below.")


with analyze_tab:
    st.subheader("Describe a molecule")
    st.caption("Paste a SMILES string, or load an example from the sidebar. This tab only *describes* the molecule you give it — it never generates anything new.")

    with st.container(border=True):
        st.caption("**SMILES** is a text way to write a molecule, e.g. `CCO` (ethanol) or `c1ccccc1` (benzene).")
        in_col, btn_col = st.columns([4, 1])
        with in_col:
            st.text_input("Molecule — SMILES string", key="smiles_input",
                          help="e.g. CCO (ethanol) or c1ccccc1 (benzene).")
            st.text_input("Context (optional annotation)", key="target_input",
                          help="Logged with the result; does not change the computed properties.")
        with btn_col:
            st.write("")
            st.write("")
            st.button("Analyze this molecule",
                      type="primary",
                      use_container_width=True,
                      on_click=handle_analyze_click)

    # Results render right here, immediately under the Analyze controls.
    st.write("")
    if "analyze_active" in st.session_state:
        st.subheader("Results")
        a = st.session_state["analyze_active"]
        with st.container(border=True):
            render_results(a["smiles"], a["target"], a["source"])
    else:
        st.info("Analyze a molecule to see its properties and interactive 3D structure here.")


# =============================================================================
# TAB 2 — GENERATE: own seed input, controls, candidate table, analysis below
# =============================================================================
with generate_tab:
    st.subheader("Sample new candidates with ChemGPT")
    st.caption("ChemGPT samples brand-new molecules — optionally starting from a seed you choose below. Generation is *de novo* — not conditioned on any target. Pick any candidate to see its full properties and 3D structure without leaving this tab.")

    if not GEN_AVAILABLE():
        st.warning("Generation needs extra packages:  `pip install torch transformers selfies`")

    with st.container(border=True):
        st.markdown("**Seed molecule** *(optional)*")
        st.caption("Generation starts from this molecule and mutates it. Leave blank to sample unseeded.")

        seed_source_col, seed_input_col = st.columns([1, 2])
        with seed_source_col:
            st.selectbox(
                "Seed source",
                ["Custom — paste below", "No seed (unseeded sampling)", "Copy from Analyze tab"]
                + list(SAMPLE_PRESETS.keys()),
                key="generate_seed_choice",
                on_change=apply_generate_seed_source,
                help="Quickly fill the seed box, or choose 'Custom' to type your own SMILES.",
            )
        with seed_input_col:
            st.text_input("Seed molecule — SMILES string", key="generate_seed_input",
                          placeholder="e.g. CCO — leave blank for unseeded sampling",
                          label_visibility="visible")

        st.text_input("Context (optional annotation)", key="generate_context_input",
                      help="Logged with any candidate you analyze from this tab.")

        seed = str(st.session_state.get("generate_seed_input", "")).strip()
        if not seed:
            st.caption("No seed — sampling unseeded.")
        elif is_valid_smiles(seed):
            st.caption(f"Seed molecule: `{seed}`  ·  valid")
        else:
            st.caption(f"Seed molecule: `{seed}`  ·  not recognized as valid SMILES — generation may still attempt it")

        st.write("")
        g1, g2, g3, g4 = st.columns(4)
        model_name = g1.selectbox("ChemGPT model",
                                  ["ncfrey/ChemGPT-19M", "ncfrey/ChemGPT-4.7M"])
        n_cands = g2.slider("Candidates", 4, 500, 50)
        temp = g3.slider("Temperature", 0.5, 1.5, 1.0, 0.1)
        max_tok = g4.slider("Max tokens", 16, 128, 64, 8)

        if st.button("Generate molecules", type="primary",
                     disabled=not GEN_AVAILABLE(), use_container_width=True):
            st.session_state.pop("generate_active", None)  # clear stale candidate analysis
            with st.spinner("Loading ChemGPT and sampling molecules (first run downloads the model)…"):
                try:
                    st.session_state["gen_candidates"] = generate_smiles(
                        model_name, n_cands, temp, max_tok, seed
                    )
                except Exception as e:
                    st.session_state["gen_candidates"] = []
                    st.error(f"Generation failed: {e}")

    st.write("")
    cands = st.session_state.get("gen_candidates")
    if cands is not None:
        st.subheader("Candidates")
        if not cands:
            st.warning("No valid molecules were generated. Try more candidates or a different temperature.")
        else:
            with st.container(border=True):
                st.markdown(f"**{len(cands)} valid candidate(s) generated**")
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
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

                pick_col, pick_btn = st.columns([4, 1])
                choice = pick_col.selectbox("Pick a candidate to analyze:", cands, key="cand_choice")
                with pick_btn:
                    st.write("")
                    st.write("")
                    if st.button("Analyze candidate", use_container_width=True):
                        st.session_state["generate_active"] = {
                            "smiles": choice,
                            "target": st.session_state.get("generate_context_input", ""),
                            "source": "generated",
                        }

            # Candidate analysis + 3D renders right here, under the generation controls.
            st.write("")
            if "generate_active" in st.session_state:
                st.subheader("Results")
                g = st.session_state["generate_active"]
                with st.container(border=True):
                    render_results(g["smiles"], g["target"], g["source"])
            else:
                st.info("Pick a candidate and analyze it to see its properties and 3D structure here.")
    else:
        st.info("Set a seed above (optional) and click **Generate molecules** to sample candidates — they'll appear here, ready to analyze.")


# --- Analysis log ------------------------------------------------------------
st.divider()
st.subheader("Analysis log")
st.caption("Every molecule analyzed this session, pasted or generated.")
log_df = db_conn.execute(
    """
    SELECT smiles, source, target_context, mw, logp, hbd, hba, tpsa, lipinski_violations
    FROM molecule_stage
    """
).df()

if not log_df.empty:
    nomenclature_names = [smiles_to_nomenclature(smile_val) for smile_val in log_df['smiles'].to_list()]
    log_df['nomenclature'] = nomenclature_names
    log_df['target_context'] = np.where(log_df['source'] == "generated", "GENERATED: " + log_df['target_context'], log_df['target_context'])

    log_df = log_df.rename(columns={
        "smiles": "SMILES", "nomenclature": "Nomenclature", "source": "Source", "target_context": "Context",
        "mw": "MW", "logp": "LogP", "hbd": "H-Donors", "hba": "H-Acceptors",
        "tpsa": "TPSA", "lipinski_violations": "Lipinski Violations",
    })
    st.dataframe(log_df, use_container_width=True, hide_index=True)
else:
    st.caption("No molecules analyzed yet.")
