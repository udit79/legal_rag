"""Opt-in integration test for the local, disposable Supabase/Postgres database."""

import os
import shutil
import subprocess
from urllib.parse import urlparse

import pytest


pytestmark = pytest.mark.sql_integration


def test_image_ocr_atomic_persistence_and_shared_immutability_triggers():
    dsn = os.getenv("PHASE4_SQL_TEST_DSN")
    if not dsn:
        pytest.skip("Set PHASE4_SQL_TEST_DSN to a disposable local Supabase/PostgreSQL database")
    psql = shutil.which("psql")
    if not psql:
        pytest.skip("psql is required for the opt-in SQL integration test")

    parsed = urlparse(dsn)
    if parsed.scheme not in {"postgres", "postgresql"} or parsed.hostname not in {
        "localhost", "127.0.0.1", "::1"
    }:
        pytest.fail("PHASE4_SQL_TEST_DSN must point to a loopback-only disposable database")

    sql_file = os.path.join(
        os.path.dirname(__file__), "test_phase4_image_ocr_immutability.sql"
    )
    completed = subprocess.run(
        [psql, dsn, "-X", "-v", "ON_ERROR_STOP=1", "-f", sql_file],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, (
        "SQL integration script failed. Ensure all Supabase migrations are applied "
        f"to the disposable DB.\n{completed.stdout}\n{completed.stderr}"
    )
