from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import venv
from collections.abc import Sequence
from pathlib import Path

MIN_PYTHON = (3, 10)
BOOTSTRAP_ENV = "ENGINEER_WORK_BOOTSTRAPPED"
CANONICAL_DATA_PATH = Path("data") / "accepted_loans.csv"
DATA_CANDIDATES = (CANONICAL_DATA_PATH, Path("data") / "accepted_2007_to_2018Q4.csv")


class LauncherError(RuntimeError):
    """An expected launcher failure that should be shown without a traceback."""


def repository_root(anchor: Path | None = None) -> Path:
    candidate = (anchor or Path(__file__)).resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for directory in (candidate, *candidate.parents):
        if (directory / "pyproject.toml").is_file() and (directory / "src").is_dir():
            return directory
    raise LauncherError(f"Repository root not found from: {candidate}")


def venv_python(venv_dir: Path) -> Path:
    relative = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    return venv_dir / relative


def is_current_interpreter(python_path: Path) -> bool:
    try:
        return Path(sys.executable).resolve() == python_path.resolve()
    except OSError:
        return False


def build_reexec_command(python_path: Path, argv: Sequence[str]) -> list[str]:
    return [str(python_path), str(Path(__file__).resolve()), *argv, "--_bootstrapped"]


def _requirements_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install_dependencies(python_path: Path, root: Path, verbose: bool = False) -> None:
    requirements = root / "requirements.txt"
    stamp = root / ".venv" / ".requirements.sha256"
    digest = _requirements_digest(requirements)
    if stamp.is_file() and stamp.read_text(encoding="utf-8").strip() == digest:
        verification = subprocess.run(
            [str(python_path), "-m", "pip", "install", "--dry-run", "--no-index", "-r", str(requirements)],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        if verification.returncode == 0:
            if verbose:
                print("Dependencies are already synchronized.")
            return
    command = [str(python_path), "-m", "pip", "install", "-r", str(requirements)]
    result = subprocess.run(command, cwd=root, check=False)
    if result.returncode:
        raise LauncherError(f"Dependency installation failed (exit code {result.returncode}).")
    stamp.write_text(digest + "\n", encoding="utf-8")


def ensure_environment(args: argparse.Namespace, argv: Sequence[str], root: Path) -> int | None:
    if sys.version_info < MIN_PYTHON:
        required = ".".join(map(str, MIN_PYTHON))
        raise LauncherError(f"Python {required}+ is required; found {sys.version.split()[0]}.")

    environment_dir = root / ".venv"
    python_path = venv_python(environment_dir)
    if not python_path.is_file():
        if args._bootstrapped or os.environ.get(BOOTSTRAP_ENV) == "1":
            raise LauncherError("Virtual environment bootstrap loop detected; .venv Python is missing.")
        print(f"Creating virtual environment: {environment_dir}")
        venv.EnvBuilder(with_pip=True).create(environment_dir)
    if not python_path.is_file():
        raise LauncherError(f"Virtual environment Python was not created: {python_path}")

    if not args.skip_install:
        install_dependencies(python_path, root, args.verbose)

    if not is_current_interpreter(python_path):
        if args._bootstrapped or os.environ.get(BOOTSTRAP_ENV) == "1":
            raise LauncherError("Virtual environment re-exec loop detected.")
        environment = os.environ.copy()
        environment[BOOTSTRAP_ENV] = "1"
        command = build_reexec_command(python_path, argv)
        if args.verbose:
            print(f"Restarting inside .venv: {command}")
        return subprocess.run(command, cwd=root, env=environment, check=False).returncode
    return None


def find_training_data(root: Path, requested: Path | None) -> Path | None:
    if requested:
        path = requested if requested.is_absolute() else root / requested
        return path.resolve() if path.is_file() else None
    for relative in DATA_CANDIDATES:
        candidate = root / relative
        if candidate.is_file():
            return candidate
    return None


def missing_inputs_message(root: Path, requested: Path | None = None) -> str:
    expected = requested or CANONICAL_DATA_PATH
    if not expected.is_absolute():
        expected = root / expected
    return (
        "No valid model artifact and no training CSV were found.\n"
        f"Missing CSV: {expected}\n"
        f"Place a real LendingClub CSV at: {root / CANONICAL_DATA_PATH}\n"
        f"Manual training: {venv_python(root / '.venv')} -m src.train "
        f"--data {root / CANONICAL_DATA_PATH} --output {root / 'artifacts'}\n"
        "The application does not create a synthetic production model or invent P(Default)."
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bootstrap, validate, train, and run the Credit Default Risk DSS.")
    parser.add_argument("--setup-only", action="store_true", help="Prepare .venv and dependencies, then exit.")
    parser.add_argument("--train", action="store_true", help="Force model training before startup.")
    parser.add_argument("--skip-install", action="store_true", help="Do not install or update dependencies.")
    parser.add_argument("--data", type=Path, help=f"Training CSV (default discovery starts at {CANONICAL_DATA_PATH}).")
    parser.add_argument(
        "--artifact-dir", type=Path, default=Path("artifacts"), help="Artifact directory (default: artifacts)."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Flask bind host (default: 127.0.0.1).")
    parser.add_argument("--port", default=5000, type=int, help="Flask bind port (default: 5000).")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode without the reloader.")
    parser.add_argument("--verbose", action="store_true", help="Show diagnostic details and tracebacks.")
    parser.add_argument("--_bootstrapped", action="store_true", help=argparse.SUPPRESS)
    return parser


def run(args: argparse.Namespace, root: Path) -> int:
    from src.config import DEFAULT_POLICY_PATH, validate_policy_config

    policy = validate_policy_config(DEFAULT_POLICY_PATH)
    if args.setup_only:
        print("Environment setup is complete.")
        return 0

    from src.train import train_from_csv
    from src.training.artifact import artifact_metadata_path, load_artifact, validate_artifact

    artifact_dir = args.artifact_dir if args.artifact_dir.is_absolute() else root / args.artifact_dir
    artifact_path = artifact_dir / "model.joblib"
    metadata_path = artifact_metadata_path(artifact_path)

    artifact = None
    if not args.train:
        try:
            artifact = load_artifact(artifact_path, policy)
        except RuntimeError as exc:
            if artifact_path.exists() or metadata_path.exists():
                raise LauncherError(f"Model artifact is invalid: {exc}") from exc

    if args.train or artifact is None:
        data_path = find_training_data(root, args.data)
        if data_path is None:
            raise LauncherError(missing_inputs_message(root, args.data))
        print(f"Training model from: {data_path}")
        try:
            train_from_csv(data_path, artifact_dir)
            artifact = load_artifact(artifact_path, policy)
        except Exception as exc:
            raise LauncherError(f"Training failed; Flask was not started: {exc}") from exc

    if not metadata_path.is_file():
        raise LauncherError(f"Model metadata is missing: {metadata_path}")
    validate_artifact(artifact, policy)

    from web_app.app import create_app

    app = create_app(artifact_path=artifact_path, policy_path=DEFAULT_POLICY_PATH)
    print(f"Starting DSS at http://{args.host}:{args.port} (debug={args.debug})")
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(effective_argv)
    try:
        root = repository_root()
        reexec_code = ensure_environment(args, effective_argv, root)
        if reexec_code is not None:
            return reexec_code
        return run(args, root)
    except Exception as exc:
        if args.verbose:
            raise
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
