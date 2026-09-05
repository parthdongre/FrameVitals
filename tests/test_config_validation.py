import pytest

from framevitals.config import AnalysisConfig, resolve_config


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"workers": True}, "workers"),
        ({"workers": 2.5}, "workers"),
        ({"artifacts": 1}, "artifacts"),
        ({"target": 42}, "target"),
        ({"max_sample_rows": True}, "max_sample_rows"),
        ({"max_relationship_pairs": 2.5}, "max_relationship_pairs"),
        ({"disabled_modules": "ai"}, "disabled_modules"),
        ({"disabled_modules": ("ai", 3)}, "disabled_modules"),
    ],
)
def test_analysis_config_rejects_runtime_type_coercion(kwargs, message):
    with pytest.raises(ValueError, match=message):
        AnalysisConfig(**kwargs)


def test_mapping_integer_controls_do_not_silently_truncate_floats():
    with pytest.raises(ValueError, match="workers"):
        resolve_config({"resources": {"workers": 2.5}})

    with pytest.raises(ValueError, match="max_sample_rows"):
        resolve_config({"resources": {"max_sample_rows": 2.5}})
