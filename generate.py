
# generate.py

import importlib
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, AllChem
import pubchempy
def GEN_AVAILABLE():
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers", "selfies"))


def smiles_to_nomenclature(smiles):
    try:
        # Search PubChem by SMILES
        compounds = pcp.get_compounds(smiles, namespace='smiles')
        if compounds:
            # Retrieve the IUPAC name
            return compounds[0].iupac_name
        return "Molecule not found in PubChem"
    except Exception as e:
        return f"Error: {e}"

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
