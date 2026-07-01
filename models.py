# models.py
import hashlib
import importlib.util
import duckdb
import streamlit as st


@st.cache_resource
def init_duckdb():
    """Shared in-memory DuckDB stage. NOTE: @st.cache_resource makes this
    connection process-global and shared across every session, hence the session id..."""
    conn = duckdb.connect(database=":memory:", read_only=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS molecule_stage (
            cache_hash          VARCHAR PRIMARY KEY,
            session_id          VARCHAR,
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


def make_cache_key(canonical_smiles: str, target_context: str, session_id: str = "") -> str:
    """Session_id is folded into the hash so two sessions analyzing the same
    molecule get distinct cache_hash values — no PRIMARY KEY collision and no
    cross-session cache hits."""
    return hashlib.sha256(
        f"{session_id}|{canonical_smiles}|{target_context}".encode()
    ).hexdigest()
