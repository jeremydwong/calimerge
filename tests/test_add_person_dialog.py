"""Tests for the create-user + assign-program flow.

Covers the data-layer behavior the AddPersonDialog depends on. We don't
spin up the Qt dialog headlessly here (that would need pytest-qt fixtures)
— the dialog is a thin shell over create_user / set_user_program / list_programs,
all of which are exercised directly.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def temp_workouts_db(monkeypatch, tmp_path):
    """Point workouts.db helpers at an isolated DB so tests don't pollute the
    real one. We rebind the resolver function rather than mutating module-level
    constants, matching how the rest of config.py is structured."""
    from calimerge import config

    isolated = tmp_path / "test_workouts.db"
    monkeypatch.setattr(config, "workouts_db_path", lambda: isolated)
    config.init_workouts_db(isolated)
    return isolated


def test_create_user_and_assign_program(temp_workouts_db):
    from calimerge.config import (
        create_user, get_user, list_programs,
        set_user_program, get_user_by_id,
    )

    user = create_user("alice", mass_kg=68.5)
    assert user["username"] == "alice"
    assert user["mass_kg"] == pytest.approx(68.5)

    fetched = get_user("alice")
    assert fetched is not None
    assert fetched["id"] == user["id"]

    progs = list_programs()
    fga = next(p for p in progs if p["name"] == "fga")
    set_user_program(user["id"], fga["id"])

    refreshed = get_user_by_id(user["id"])
    assert refreshed is not None
    assert refreshed["active_program_id"] == fga["id"]
    assert refreshed["program_started_at"] is not None


def test_create_user_no_mass(temp_workouts_db):
    from calimerge.config import create_user, get_user

    user = create_user("bob", mass_kg=None)
    fetched = get_user("bob")
    assert fetched["mass_kg"] is None


def test_duplicate_username_raises(temp_workouts_db):
    """The dialog's accept handler does its own get_user check first to avoid
    this exception, but the data layer should still complain — we don't want
    silent overwrites if the dialog logic is ever bypassed."""
    from calimerge.config import create_user
    import sqlite3

    create_user("carol")
    with pytest.raises(sqlite3.IntegrityError):
        create_user("carol")


def test_list_programs_includes_seeded_three(temp_workouts_db):
    from calimerge.config import list_programs

    progs = list_programs()
    names = {p["name"] for p in progs}
    assert {"vivifrail", "calisthenics", "fga"} <= names


def test_dialog_module_imports():
    """Smoke-import: the dialog module shouldn't crash on import. This catches
    typos that would break the runtime open-from-File-menu path even if the
    dialog itself isn't constructed in the test."""
    from calimerge.gui import add_person_dialog
    assert hasattr(add_person_dialog, "AddPersonDialog")
