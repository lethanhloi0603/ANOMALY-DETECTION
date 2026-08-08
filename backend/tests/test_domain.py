from __future__ import annotations

import json
from datetime import date

import pytest

from app.domain.fusion import fuse_scores
from app.domain.readiness import (
    Branch,
    FeatureGlobalSupport,
    FeaturePersonSupport,
    FeatureRoleSupport,
    ReferenceLevel,
    SequenceGlobalSupport,
    SequencePersonSupport,
    SequenceRoleSupport,
    load_framework_config,
    select_feature_reference,
    select_sequence_reference,
)
from app.domain.universe import EmployeePeriod, eligible_employee_days

SCORE_DATE = date(2010, 6, 1)
AS_OF = date(2010, 5, 31)


def test_employee_day_universe_keeps_zero_event_days_and_effective_bounds() -> None:
    universe = eligible_employee_days(
        [
            EmployeePeriod(
                employee_id="U001",
                valid_from=date(2010, 6, 1),
                valid_to=date(2010, 6, 3),
            ),
            EmployeePeriod(
                employee_id="U002",
                valid_from=date(2010, 6, 2),
                valid_to=None,
            ),
        ],
        start=date(2010, 6, 1),
        end=date(2010, 6, 3),
    )

    assert universe == [
        ("U001", date(2010, 6, 1)),
        ("U001", date(2010, 6, 2)),
        ("U002", date(2010, 6, 2)),
        ("U002", date(2010, 6, 3)),
    ]


def feature_person(**changes):
    values = {
        "support_as_of": AS_OF,
        "active_days": 60,
        "span_days": 90,
        "active_days_current_role": 60,
        "coverage": 0.90,
        "min_feature_observations": 40,
        "last_active_gap_days": 30,
    }
    values.update(changes)
    return FeaturePersonSupport(**values)


def feature_role(**changes):
    values = {
        "support_as_of": AS_OF,
        "role_known": True,
        "peer_users": 15,
        "peer_user_days": 300,
        "recent_user_days": 100,
        "coverage": 0.90,
        "min_feature_observations": 200,
    }
    values.update(changes)
    return FeatureRoleSupport(**values)


def feature_global(**changes):
    values = {
        "support_as_of": AS_OF,
        "users": 200,
        "user_days": 10_000,
        "coverage": 0.90,
        "train_fitted": True,
    }
    values.update(changes)
    return FeatureGlobalSupport(**values)


def sequence_person(**changes):
    values = {
        "support_as_of": AS_OF,
        "sequence_days": 60,
        "transitions": 1_500,
        "span_days": 90,
        "sequence_days_current_role": 60,
        "last_active_gap_days": 30,
    }
    values.update(changes)
    return SequencePersonSupport(**values)


def sequence_role(**changes):
    values = {
        "support_as_of": AS_OF,
        "role_known": True,
        "peer_users": 15,
        "sequence_days": 300,
        "transitions": 10_000,
        "recent_transitions": 2_000,
    }
    values.update(changes)
    return SequenceRoleSupport(**values)


def sequence_global(**changes):
    values = {
        "support_as_of": AS_OF,
        "users": 200,
        "sequence_days": 10_000,
        "transitions": 100_000,
        "train_fitted": True,
    }
    values.update(changes)
    return SequenceGlobalSupport(**values)


def test_feature_selects_person_at_exact_document_thresholds():
    decision = select_feature_reference(
        score_date=SCORE_DATE,
        person=feature_person(),
        role=None,
        global_support=None,
    )

    assert decision.branch is Branch.FEATURE
    assert decision.selected_level is ReferenceLevel.PERSON
    assert decision.can_score
    assert decision.reason_codes == ()
    assert [item.level for item in decision.evaluations] == [ReferenceLevel.PERSON]


def test_feature_person_fails_then_role_is_selected_with_all_reasons_in_order():
    decision = select_feature_reference(
        score_date=SCORE_DATE,
        person=feature_person(
            active_days=59,
            span_days=89,
            active_days_current_role=59,
            coverage=0.89,
            min_feature_observations=39,
            last_active_gap_days=31,
        ),
        role=feature_role(),
        global_support=None,
    )

    assert decision.selected_level is ReferenceLevel.ROLE
    assert decision.reason_codes == (
        "P_ACTIVE_DAYS_LOW",
        "P_SPAN_DAYS_LOW",
        "P_ROLE_TENURE_LOW",
        "P_COVERAGE_LOW",
        "P_FEATURE_SUPPORT_LOW",
        "P_STALE",
    )


def test_feature_role_unknown_and_insufficient_falls_back_to_global():
    decision = select_feature_reference(
        score_date=SCORE_DATE,
        person=None,
        role=feature_role(
            role_known=False,
            peer_users=14,
            peer_user_days=299,
            recent_user_days=99,
            coverage=0.89,
            min_feature_observations=199,
        ),
        global_support=feature_global(),
    )

    assert decision.selected_level is ReferenceLevel.GLOBAL
    assert decision.reason_codes == (
        "P_SUPPORT_MISSING",
        "R_UNKNOWN",
        "R_USERS_LT_15",
        "R_USERDAYS_LT_300",
        "R_RECENT_LOW",
        "R_COVERAGE_LOW",
        "R_FEATURE_SUPPORT_LT_200",
    )


def test_feature_global_insufficient_returns_no_score():
    decision = select_feature_reference(
        score_date=SCORE_DATE,
        person=None,
        role=None,
        global_support=feature_global(
            users=199,
            user_days=9_999,
            coverage=0.89,
            train_fitted=False,
        ),
    )

    assert decision.selected_level is ReferenceLevel.NO_SCORE
    assert not decision.can_score
    assert decision.reason_codes == (
        "P_SUPPORT_MISSING",
        "R_SUPPORT_MISSING",
        "G_USERS_LT_200",
        "G_USERDAYS_LT_10000",
        "G_COVERAGE_LOW",
        "G_NOT_TRAIN_FITTED",
        "F_INSUFFICIENT_DATA",
    )


def test_support_as_of_must_be_strictly_before_score_date():
    decision = select_feature_reference(
        score_date=SCORE_DATE,
        person=feature_person(support_as_of=SCORE_DATE),
        role=feature_role(),
        global_support=feature_global(),
    )

    assert decision.selected_level is ReferenceLevel.ROLE
    assert decision.reason_codes == ("P_SUPPORT_NOT_PAST",)


def test_feature_configuration_override_is_loaded_from_json(tmp_path):
    config_path = tmp_path / "framework.v5.json"
    config_path.write_text(
        json.dumps(
            {
                "version": "test.v2",
                "readiness": {
                    "feature": {
                        "person": {"min_active_days": 31},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    decision = select_feature_reference(
        score_date=SCORE_DATE,
        person=feature_person(active_days=30),
        role=feature_role(),
        global_support=feature_global(),
        config_path=config_path,
    )

    assert decision.selected_level is ReferenceLevel.ROLE
    assert decision.reason_codes == ("P_ACTIVE_DAYS_LOW",)
    assert decision.config_version == "test.v2"


def test_repository_config_aliases_are_honored(tmp_path):
    config_path = tmp_path / "framework.v5.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "alias.v1",
                "readiness": {
                    "feature": {
                        "person": {
                            "min_active_days_in_current_role": 31,
                            "min_feature_coverage_ratio": 0.95,
                        }
                    },
                    "sequence": {
                        "person": {
                            "min_sequence_days_in_current_role": 21,
                            "max_stale_gap_days": 20,
                        }
                    },
                },
                "fusion": {
                    "weights": {"feature": 3, "sequence": 1},
                },
            }
        ),
        encoding="utf-8",
    )

    feature_decision = select_feature_reference(
        score_date=SCORE_DATE,
        person=feature_person(
            active_days_current_role=30,
            coverage=0.94,
        ),
        role=feature_role(),
        global_support=feature_global(),
        config_path=config_path,
    )
    sequence_decision = select_sequence_reference(
        score_date=SCORE_DATE,
        seq_len=2,
        person=sequence_person(
            sequence_days_current_role=20,
            last_active_gap_days=21,
        ),
        role=sequence_role(),
        global_support=sequence_global(),
        config_path=config_path,
    )
    fusion = fuse_scores(
        q_feature=0.8,
        q_sequence=0.2,
        config_path=config_path,
    )

    assert feature_decision.selected_level is ReferenceLevel.ROLE
    assert feature_decision.reason_codes == (
        "P_ROLE_TENURE_LOW",
        "P_COVERAGE_LOW",
    )
    assert feature_decision.config_version == "alias.v1"
    assert sequence_decision.selected_level is ReferenceLevel.ROLE
    assert sequence_decision.reason_codes == (
        "SP_ROLE_TENURE_LOW",
        "SP_STALE",
    )
    assert fusion.risk == pytest.approx(0.65)
    assert fusion.config_version == "alias.v1"


def test_missing_config_path_uses_framework_defaults(tmp_path):
    config = load_framework_config(tmp_path / "does-not-exist.json")

    assert config["version"] == "framework.v5"
    assert config["readiness"]["sequence"]["current_day"]["min_seq_len"] == 2


def test_sequence_length_below_two_is_immediate_no_score():
    decision = select_sequence_reference(
        score_date=SCORE_DATE,
        seq_len=1,
        person=sequence_person(),
        role=sequence_role(),
        global_support=sequence_global(),
    )

    assert decision.branch is Branch.SEQUENCE
    assert decision.selected_level is ReferenceLevel.NO_SCORE
    assert decision.reason_codes == ("S_CURRENT_LEN_LT_2",)
    assert len(decision.evaluations) == 1


def test_sequence_selects_person_at_exact_document_thresholds():
    decision = select_sequence_reference(
        score_date=SCORE_DATE,
        seq_len=2,
        person=sequence_person(),
        role=None,
        global_support=None,
    )

    assert decision.selected_level is ReferenceLevel.PERSON
    assert decision.reason_codes == ()


def test_sequence_person_fails_then_role_is_selected_deterministically():
    decision = select_sequence_reference(
        score_date=SCORE_DATE,
        seq_len=2,
        person=sequence_person(
            support_as_of=SCORE_DATE,
            sequence_days=59,
            transitions=1_499,
            span_days=89,
            sequence_days_current_role=59,
            last_active_gap_days=31,
        ),
        role=sequence_role(),
        global_support=None,
    )

    assert decision.selected_level is ReferenceLevel.ROLE
    assert decision.reason_codes == (
        "SP_SUPPORT_NOT_PAST",
        "SP_DAYS_LOW",
        "SP_TRANSITIONS_LOW",
        "SP_SPAN_DAYS_LOW",
        "SP_ROLE_TENURE_LOW",
        "SP_STALE",
    )


def test_sequence_role_fails_then_global_is_selected():
    decision = select_sequence_reference(
        score_date=SCORE_DATE,
        seq_len=2,
        person=None,
        role=sequence_role(
            role_known=False,
            peer_users=14,
            sequence_days=299,
            transitions=9_999,
            recent_transitions=1_999,
        ),
        global_support=sequence_global(),
    )

    assert decision.selected_level is ReferenceLevel.GLOBAL
    assert decision.reason_codes == (
        "SP_SUPPORT_MISSING",
        "SR_UNKNOWN",
        "SR_USERS_LT_15",
        "SR_DAYS_LT_300",
        "SR_TRANS_LT_10000",
        "SR_RECENT_LOW",
    )


def test_sequence_global_fails_then_no_score():
    decision = select_sequence_reference(
        score_date=SCORE_DATE,
        seq_len=2,
        person=None,
        role=None,
        global_support=sequence_global(
            users=199,
            sequence_days=9_999,
            transitions=99_999,
            train_fitted=False,
        ),
    )

    assert decision.selected_level is ReferenceLevel.NO_SCORE
    assert decision.reason_codes == (
        "SP_SUPPORT_MISSING",
        "SR_SUPPORT_MISSING",
        "SG_USERS_LT_200",
        "SG_DAYS_LT_10000",
        "SG_TRANS_LT_100000",
        "SG_NOT_TRAIN_FITTED",
        "S_INSUFFICIENT_DATA",
    )


def test_negative_sequence_length_is_invalid():
    with pytest.raises(ValueError, match="non-negative"):
        select_sequence_reference(
            score_date=SCORE_DATE,
            seq_len=-1,
            person=None,
            role=None,
            global_support=None,
        )


def test_fusion_uses_both_calibrated_scores():
    result = fuse_scores(q_feature=0.97, q_sequence=0.93)

    assert result.risk == pytest.approx(0.95)
    assert result.feature_weight == pytest.approx(0.5)
    assert result.sequence_weight == pytest.approx(0.5)
    assert result.available_branches == ("FEATURE", "SEQUENCE")
    assert result.reason_codes == ()


@pytest.mark.parametrize(
    ("q_feature", "q_sequence", "risk", "weights", "reason"),
    [
        (0.8, None, 0.8, (1.0, 0.0), "SEQUENCE_NO_SCORE"),
        (None, 0.7, 0.7, (0.0, 1.0), "FEATURE_NO_SCORE"),
    ],
)
def test_fusion_normalizes_weight_when_one_branch_is_missing(
    q_feature, q_sequence, risk, weights, reason
):
    result = fuse_scores(q_feature=q_feature, q_sequence=q_sequence)

    assert result.risk == pytest.approx(risk)
    assert (result.feature_weight, result.sequence_weight) == weights
    assert result.reason_codes == (reason,)


def test_fusion_returns_no_score_when_both_branches_are_missing():
    result = fuse_scores(q_feature=None, q_sequence=None)

    assert result.risk is None
    assert result.is_no_score
    assert not result.can_alert
    assert result.reason_codes == ("FUSION_NO_BRANCH_SCORE",)


def test_fusion_uses_configured_weights_and_deep_merge():
    result = fuse_scores(
        q_feature=0.8,
        q_sequence=0.2,
        config={
            "version": "weighted.v1",
            "fusion": {
                "feature_weight": 3,
                "sequence_weight": 1,
            },
        },
    )

    assert result.risk == pytest.approx(0.65)
    assert result.feature_weight == pytest.approx(0.75)
    assert result.sequence_weight == pytest.approx(0.25)
    assert result.config_version == "weighted.v1"


@pytest.mark.parametrize("bad_score", [-0.01, 1.01, float("nan")])
def test_fusion_rejects_uncalibrated_or_non_finite_scores(bad_score):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        fuse_scores(q_feature=bad_score, q_sequence=0.5)
