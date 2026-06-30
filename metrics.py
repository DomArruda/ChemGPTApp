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