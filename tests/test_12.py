import pytest
from definition_ccbed655178849b79d5227ac18ab5c0e import assign_grade_group

@pytest.mark.parametrize("grade, expected", [
    ("A", "Prime"),
    ("B", "Prime"),
    ("C", "Subprime"),
    ("G", "Subprime"),
    ("Z", "Subprime"),  # Edge case: Invalid grade should still be Subprime
])
def test_assign_grade_group(grade, expected):
    assert assign_grade_group(grade) == expected
