
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

