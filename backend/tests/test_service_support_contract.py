from __future__ import annotations

from datetime import date

import pytest

from app.models import ReferenceLevel, ReferenceProfile
from app.services import _build_feature_supports, _build_sequence_supports


def _profile(support: dict[str, object]) -> ReferenceProfile:
    return ReferenceProfile(
        fitted_through=date(2010, 5, 31),
        support_json=support,
    )


def test_feature_person_support_accepts_ml_v2_aliases() -> None:
    person, _, _ = _build_feature_supports(
        {
            ReferenceLevel.PERSON: _profile(
                {
                    "active_days": 60,
                    "span_days": 90,
                    "mean_feature_coverage": 0.95,
                    "minimum_nonzero_feature_support": 40,
                    "last_active_day": "2010-05-30",
                }
            )
        },
        role_known=True,
    )
    assert person is not None
    assert person.active_days_current_role == 60
    assert person.coverage == pytest.approx(0.95)
    assert person.min_feature_observations == 40
    assert person.last_active_gap_days == 1


def test_sequence_person_support_uses_last_sequence_day() -> None:
    person, _, _ = _build_sequence_supports(
        {
            ReferenceLevel.PERSON: _profile(
                {
                    "sequence_days": 60,
                    "transitions": 1_500,
                    "span_days": 90,
                    "last_active_day": "2010-05-31",
                    "last_sequence_day": "2010-05-29",
                }
            )
        },
        role_known=True,
    )
    assert person is not None
    assert person.sequence_days_current_role == 60
    assert person.last_active_gap_days == 2
