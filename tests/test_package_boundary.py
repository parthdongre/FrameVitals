from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_framevitals_has_no_legacy_module_imports():
    package_dir = (
        ROOT
        / "src"
        / "framevitals"
    )

    for path in package_dir.glob("*.py"):
        source = path.read_text(
            encoding="utf-8"
        )

        assert "from modules." not in source, (
            f"{path.name} imports through "
            "the legacy modules namespace."
        )

        assert "import modules." not in source, (
            f"{path.name} imports through "
            "the legacy modules namespace."
        )


def test_distribution_only_packages_framevitals():
    pyproject_path = (
        ROOT
        / "pyproject.toml"
    )

    with pyproject_path.open(
        "rb"
    ) as handle:
        pyproject = tomllib.load(
            handle
        )

    packages = (
        pyproject[
            "tool"
        ][
            "setuptools"
        ][
            "packages"
        ]
    )

    assert packages == [
        "framevitals",
    ]
