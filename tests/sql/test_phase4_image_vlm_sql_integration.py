"""Explicit integration test for the Phase 4 M2 migration on disposable Supabase."""

import glob
import os
import shutil
import subprocess
from urllib.parse import urlparse

import pytest


pytestmark = pytest.mark.sql_integration


def test_vlm_description_persistence_permissions_and_immutability():
    dsn = os.getenv("PHASE4_VLM_SQL_TEST_DSN")
    if not dsn:
        pytest.skip("Set PHASE4_VLM_SQL_TEST_DSN for the disposable M2 Supabase database")
    psql = os.getenv("PHASE4_PSQL_PATH") or shutil.which("psql")
    if not psql and os.name == "nt":
        installed = glob.glob(r"C:\Program Files\PostgreSQL\*\bin\psql.exe")
        psql = installed[-1] if installed else None
    if not psql:
        pytest.skip("psql is required for the opt-in Phase 4 M2 SQL integration test")

    parsed = urlparse(dsn)
    if parsed.scheme not in {"postgres", "postgresql"}:
        pytest.fail("PHASE4_VLM_SQL_TEST_DSN must be a PostgreSQL connection URI")
    if parsed.hostname not in {"localhost", "127.0.0.1", "::1"} and os.getenv(
        "PHASE4_VLM_SQL_TEST_ALLOW_REMOTE"
    ) != "1":
        pytest.fail("Remote SQL tests require PHASE4_VLM_SQL_TEST_ALLOW_REMOTE=1")

    sql_file = os.path.join(os.path.dirname(__file__), "test_phase4_image_vlm_persistence.sql")
    completed = subprocess.run(
        [psql, dsn, "-X", "-v", "ON_ERROR_STOP=1", "-f", sql_file],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert completed.returncode == 0, (
        "Phase 4 M2 SQL integration failed; first deploy migration 202611150001 "
        f"to the disposable project.\n{completed.stdout}\n{completed.stderr}"
    )
