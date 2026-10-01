"""Tiny assert-based tests for the guard in query_parse.parse().

Run with  python -m pytest tests  (or standalone: python3 tests/test_guard.py)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from jira_rag.query_parse import _apply_question_name_guard

# Helper: run the guard on (question, form) and return the guarded form.
def guard(question, form):
    return _apply_question_name_guard(question, form)


def test_subtask_not_task():
    # "Which blocker sub-tasks are open?" — form has issue_type ["Sub-task","Task"].
    # "sub-task" appears -> keep Sub-task.  "task" must NOT be satisfied by the
    # "sub-task" mention -> drop Task.
    form = {
        "priority": None, "issue_type": ["Sub-task", "Task"], "open": None,
        "resolution": None, "created_from": None, "created_to": None,
        "text_terms": [],
    }
    out = guard("Which blocker sub-tasks are open?", dict(form))
    assert out["issue_type"] == ["Sub-task"], out["issue_type"]


def test_planning_efforts_task_dropped():
    # "planning efforts for Spark Connect" — no "task"/"sub-task" word -> Task dropped.
    form = {
        "priority": None, "issue_type": ["Task"], "open": None,
        "resolution": None, "created_from": None, "created_to": None,
        "text_terms": [],
    }
    out = guard("Which critical-priority planning efforts for Spark Connect are still open?", dict(form))
    assert out["issue_type"] is None, out["issue_type"]


def test_critical_bugs_kept():
    # "critical bugs" — both "Critical" and "Bug" appear -> both kept.
    form = {
        "priority": ["Critical"], "issue_type": ["Bug"], "open": None,
        "resolution": None, "created_from": None, "created_to": None,
        "text_terms": [],
    }
    out = guard("Which critical bugs filed in 2026 are still open?", dict(form))
    assert out["priority"] == ["Critical"], out["priority"]
    assert out["issue_type"] == ["Bug"], out["issue_type"]


def test_zookeeper_upgrades_improvement_dropped():
    # "ZooKeeper upgrades" — "Improvement" not in question -> dropped.
    form = {
        "priority": None, "issue_type": ["Improvement"], "open": None,
        "resolution": None, "created_from": None, "created_to": None,
        "text_terms": [],
    }
    out = guard("Were any ZooKeeper upgrades proposed but then rejected or postponed?", dict(form))
    assert out["issue_type"] is None, out["issue_type"]


def main():
    tests = [
        test_subtask_not_task,
        test_planning_efforts_task_dropped,
        test_critical_bugs_kept,
        test_zookeeper_upgrades_improvement_dropped,
    ]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"\nAll {len(tests)} guard tests passed.")


if __name__ == "__main__":
    main()
