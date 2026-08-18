from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import run_app


def launcher_args(**overrides):
    values = {
        "setup_only": False,
        "train": False,
        "skip_install": True,
        "data": None,
        "artifact_dir": Path("artifacts"),
        "host": "127.0.0.1",
        "port": 5000,
        "debug": False,
        "verbose": False,
        "_bootstrapped": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_repository_root_is_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert run_app.repository_root() == Path(run_app.__file__).resolve().parent


def test_venv_python_platform_layout(tmp_path, monkeypatch):
    expected = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    assert run_app.venv_python(tmp_path) == tmp_path / expected


def test_build_reexec_command_preserves_argument_boundaries(tmp_path):
    command = run_app.build_reexec_command(tmp_path / "python with spaces", ["--data", "a b.csv"])
    assert command[0] == str(tmp_path / "python with spaces")
    assert command[2:4] == ["--data", "a b.csv"]
    assert command[-1] == "--_bootstrapped"


def test_ready_environment_does_not_reexec(tmp_path, monkeypatch):
    python_path = run_app.venv_python(tmp_path / ".venv")
    python_path.parent.mkdir(parents=True)
    python_path.touch()
    monkeypatch.setattr(run_app, "is_current_interpreter", lambda _: True)
    assert run_app.ensure_environment(launcher_args(), [], tmp_path) is None


def test_skip_install_avoids_dependency_command(tmp_path, monkeypatch):
    python_path = run_app.venv_python(tmp_path / ".venv")
    python_path.parent.mkdir(parents=True)
    python_path.touch()
    monkeypatch.setattr(run_app, "is_current_interpreter", lambda _: True)
    monkeypatch.setattr(run_app, "install_dependencies", lambda *args: pytest.fail("install called"))
    run_app.ensure_environment(launcher_args(skip_install=True), [], tmp_path)


def test_dependency_stamp_is_verified_offline(tmp_path, monkeypatch):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("example-package>=1\n", encoding="utf-8")
    environment = tmp_path / ".venv"
    environment.mkdir()
    (environment / ".requirements.sha256").write_text(
        run_app._requirements_digest(requirements) + "\n", encoding="utf-8"
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_app.subprocess, "run", fake_run)
    run_app.install_dependencies(Path("python"), tmp_path)
    assert "--dry-run" in calls[0]
    assert "--no-index" in calls[0]


def test_bootstrapped_interpreter_mismatch_blocks_loop(tmp_path, monkeypatch):
    python_path = run_app.venv_python(tmp_path / ".venv")
    python_path.parent.mkdir(parents=True)
    python_path.touch()
    monkeypatch.setattr(run_app, "is_current_interpreter", lambda _: False)
    with pytest.raises(run_app.LauncherError, match="loop"):
        run_app.ensure_environment(launcher_args(_bootstrapped=True), [], tmp_path)


def test_setup_only_stops_after_policy_validation(tmp_path):
    assert run_app.run(launcher_args(setup_only=True), tmp_path) == 0


def test_absent_artifact_and_csv_has_actionable_error(tmp_path):
    with pytest.raises(run_app.LauncherError) as error:
        run_app.run(launcher_args(artifact_dir=tmp_path / "artifacts"), tmp_path)
    message = str(error.value)
    assert "accepted_loans.csv" in message
    assert "Manual training:" in message
    assert "does not create a synthetic production model" in message


def test_corrupted_artifact_blocks_training(tmp_path, monkeypatch):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "model.joblib").write_bytes(b"not-joblib")
    data = tmp_path / "data.csv"
    data.write_text("real,csv\n", encoding="utf-8")
    monkeypatch.setattr("src.train.train_from_csv", lambda *_: pytest.fail("training called"))
    with pytest.raises(run_app.LauncherError, match="invalid"):
        run_app.run(launcher_args(artifact_dir=artifact_dir, data=data), tmp_path)


def test_training_failure_never_starts_flask(tmp_path, monkeypatch):
    data = tmp_path / "data.csv"
    data.write_text("real,csv\n", encoding="utf-8")
    monkeypatch.setattr("src.train.train_from_csv", lambda *_: (_ for _ in ()).throw(ValueError("bad data")))
    monkeypatch.setattr("flask.Flask.run", lambda *_args, **_kwargs: pytest.fail("Flask started"))
    with pytest.raises(run_app.LauncherError, match="Training failed"):
        run_app.run(launcher_args(artifact_dir=tmp_path / "artifacts", data=data), tmp_path)


def test_artifact_present_allows_full_root_flow(artifact_path, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("flask.Flask.run", lambda _app, **kwargs: calls.append(kwargs))
    result = run_app.run(launcher_args(artifact_dir=artifact_path.parent), tmp_path)
    assert result == 0
    assert calls == [{"host": "127.0.0.1", "port": 5000, "debug": False, "use_reloader": False}]


def test_absent_artifact_with_csv_trains_then_runs(artifact_path, tmp_path, monkeypatch):
    output = tmp_path / "trained"
    data = tmp_path / "real.csv"
    data.write_text("placeholder", encoding="utf-8")
    trained = []

    def fake_train(_data, output_dir):
        from shutil import copy2

        from src.training.artifact import artifact_metadata_path

        output_dir.mkdir(parents=True)
        copy2(artifact_path, output_dir / "model.joblib")
        copy2(artifact_metadata_path(artifact_path), output_dir / "metadata.json")
        trained.append(True)

    monkeypatch.setattr("src.train.train_from_csv", fake_train)
    monkeypatch.setattr("flask.Flask.run", lambda *_args, **_kwargs: None)
    assert run_app.run(launcher_args(artifact_dir=output, data=data), tmp_path) == 0
    assert trained


def test_train_flag_forces_training(artifact_path, tmp_path, monkeypatch):
    data = tmp_path / "real.csv"
    data.write_text("placeholder", encoding="utf-8")
    calls = []

    def fake_train(_data, _output):
        calls.append(True)

    monkeypatch.setattr("src.train.train_from_csv", fake_train)
    monkeypatch.setattr("flask.Flask.run", lambda *_args, **_kwargs: None)
    run_app.run(launcher_args(train=True, artifact_dir=artifact_path.parent, data=data), tmp_path)
    assert calls == [True]


def test_help_does_not_bootstrap(capsys):
    with pytest.raises(SystemExit) as exit_info:
        run_app.main(["--help"])
    assert exit_info.value.code == 0
    assert "--setup-only" in capsys.readouterr().out


def test_main_propagates_reexec_status(monkeypatch):
    monkeypatch.setattr(run_app, "ensure_environment", lambda *_: 17)
    assert run_app.main(["--skip-install"]) == 17
