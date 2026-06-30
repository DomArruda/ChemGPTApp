# Bio-Chem Molecule Studio — educational demo
# Requires:            streamlit duckdb rdkit py3Dmol stmol
# Optional (ChemGPT):  torch transformers selfies
#   pip install streamlit duckdb rdkit py3Dmol stmol torch transformers selfies
# Split each "# file.py" block into its own module later.

# models.py
import hashlib
import importlib.util
import duckdb
import streamlit as st


@st.cache_resource
def init_duckdb():
    """In-memory DuckDB stage: memoizes analyses and doubles as an audit log."""
    conn = duckdb.connect(database=":memory:", read_only=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS molecule_stage (
            cache_hash          VARCHAR PRIMARY KEY,
            smiles              VARCHAR,
            target_context      VARCHAR,
            source              VARCHAR,
            mw                  DOUBLE,
            logp                DOUBLE,
            hbd                 INTEGER,
            hba                 INTEGER,
            tpsa                DOUBLE,
            rot_bonds           INTEGER,
            rings               INTEGER,
            heavy_atoms         INTEGER,
            lipinski_violations INTEGER
        )
    """)
    return conn


def make_cache_key(canonical_smiles: str, target_context: str) -> str:
    return hashlib.sha256(f"{canonical_smiles}|{target_context}".encode()).hexdigest()


# metrics.py
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, AllChem


def compute_descriptors(smiles: str):
    """Real physicochemical descriptors. Returns None on an unparseable SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    desc = {
        "canonical": Chem.MolToSmiles(mol),
        "mw": round(Descriptors.MolWt(mol), 1),
        "logp": round(Crippen.MolLogP(mol), 2),
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol),
        "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 1),
        "rot_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "rings": rdMolDescriptors.CalcNumRings(mol),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
    }
    desc["lipinski_violations"] = sum(
        [desc["mw"] > 500, desc["logp"] > 5, desc["hbd"] > 5, desc["hba"] > 10]
    )
    return desc


def smiles_to_molblock(smiles: str):
    """Generate 3D coordinates locally with RDKit (no network). None if it fails."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
        if AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True) != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass
    return Chem.MolToMolBlock(mol)


# generate.py  (ChemGPT — a real small transformer, optional)
GEN_AVAILABLE = all(
    importlib.util.find_spec(m) is not None for m in ("torch", "transformers", "selfies")
)


@st.cache_resource(show_spinner=False)
def load_chemgpt(model_name: str):
    """Load tokenizer + causal LM once. First call downloads weights from HF."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return tok, model


def _seed_ids(tok, seed_smiles: str):
    """Build the prompt token ids as a SELFIES fragment."""
    import selfies
    sf = "[C]"
    if seed_smiles.strip():
        mol = Chem.MolFromSmiles(seed_smiles)
        if mol is not None:
            try:
                sf = selfies.encoder(Chem.MolToSmiles(mol))
            except Exception:
                sf = "[C]"
    return tok(sf, return_tensors="pt").input_ids


def generate_smiles(model_name, n, temperature, max_new_tokens, seed_smiles=""):
    """Sample candidate molecules, decode SELFIES -> SMILES, keep RDKit-valid ones."""
    import torch
    import selfies

    tok, model = load_chemgpt(model_name)
    input_ids = _seed_ids(tok, seed_smiles)
    pad_id = tok.pad_token_id
    if pad_id is None:
        pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0

    with torch.no_grad():
        out = model.generate(
            input_ids,
            do_sample=True,
            temperature=float(temperature),
            top_k=50,
            max_new_tokens=int(max_new_tokens),
            num_return_sequences=int(n),
            pad_token_id=pad_id,
        )

    seen, results = set(), []
    for seq in out:
        text = "".join(tok.decode(seq, skip_special_tokens=True).split())
        try:
            smi = selfies.decoder(text)
        except Exception:
            continue
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol)
        if canon and canon not in seen:
            seen.add(canon)
            results.append(canon)
    return results


# main.py
import pandas as pd
import py3Dmol
from stmol import showmol

st.set_page_config(
    page_title="ChemGPT Studio",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

db_conn = init_duckdb()

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
st.subheader("Molecule")
in_col, btn_col = st.columns([4, 1])
with in_col:
    st.text_input("Molecule — SMILES string", key="smiles_input",
                  help="e.g. CCO (ethanol) or c1ccccc1 (benzene).")
    st.text_input("Context (optional annotation)", key="target_input",
                  help="Logged with the result; does not change the computed properties.")
with btn_col:
    st.write("")
    st.write("")
    if st.button("Analyze this molecule (no generation)", type="primary", width="stretch"):
        st.session_state["active"] = {
            "smiles": st.session_state["smiles_input"],
            "target": st.session_state["target_input"],
            "source": "pasted",
        }

# --- Generation (ChemGPT) ----------------------------------------------------
with st.container(border=True):
    st.markdown("**✨ Generate new candidates with ChemGPT** (optional)")
    st.caption("Seeded from the molecule above (clear the box for unseeded sampling). Generation is *de novo* — not conditioned on the target.")
    if not GEN_AVAILABLE:
        st.warning("Generation needs extra packages:  `pip install torch transformers selfies`")

    g1, g2, g3, g4 = st.columns([2, 1, 1, 1])
    model_name = g1.selectbox("ChemGPT model (well under 1B params)",
                              ["ncfrey/ChemGPT-19M", "ncfrey/ChemGPT-4.7M" ])
    n_cands = g2.slider("Candidates", 4, 100, 25)
    temp = g3.slider("Temperature", 0.5, 1.5, 1.0, 0.1)
    max_tok = g4.slider("Max tokens", 16, 128, 64, 8)

    if st.button("Generate Molecules", type="secondary",
                 disabled=not GEN_AVAILABLE, width="stretch"):
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
                d = compute_descriptors(smi)
                if d:
                    rows.append({"SMILES": smi, "MW": d["mw"], "LogP": d["logp"],
                                 "Lipinski viol.": d["lipinski_violations"]})
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
).df().rename(columns={
    "smiles": "SMILES", "source": "Source", "target_context": "Target",
    "mw": "MW", "logp": "LogP", "hbd": "H-Donors", "hba": "H-Acceptors",
    "tpsa": "TPSA", "lipinski_violations": "Lipinski Violations",
})
if not log_df.empty:
    st.dataframe(log_df, width="stretch", hide_index=True)
else:
    st.caption("No molecules analyzed yet.")