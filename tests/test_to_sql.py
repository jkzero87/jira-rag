"""to_sql(): parameterized WHERE clause from a filter form."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from jira_rag.query_parse import to_sql


def form(**fields):
    base = {"priority": None, "issue_type": None, "open": None, "resolution": None,
            "created_from": None, "created_to": None, "text_terms": []}
    base.update(fields)
    return base


def test_empty_form_has_no_where():
    assert to_sql(form(text_terms=["Kubernetes"])) == ("", [])


def test_all_filters_parameterized():
    where, params = to_sql(form(priority=["Blocker"], issue_type=["Bug", "Task"],
                                open=True, created_from="2026-06-01",
                                created_to="2027-01-01"))
    assert where == ("priority IN (%s) AND issue_type IN (%s, %s) AND resolution IS NULL"
                     " AND created >= (%s::timestamp AT TIME ZONE 'UTC')"
                     " AND created < (%s::timestamp AT TIME ZONE 'UTC')")
    assert params == ["Blocker", "Bug", "Task", "2026-06-01", "2027-01-01"]


def test_closed_and_resolution():
    where, params = to_sql(form(open=False, resolution=["Won't Fix", "Later"]))
    assert where == "resolution IS NOT NULL AND resolution IN (%s, %s)"
    assert params == ["Won't Fix", "Later"]


def test_bare_string_priority():
    assert to_sql(form(priority="Critical")) == ("priority IN (%s)", ["Critical"])
