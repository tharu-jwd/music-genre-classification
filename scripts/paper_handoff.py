"""Download a run's lightweight results ZIP, or combine teammates' ZIPs locally."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_experiments import SEEDS, load_manifest, selected_runs

FILES = ("results.json", "evaluation_test.json", "test_predictions.npz")


def download_seed(manifest, seed, output, *, volume_name="music-genre-runs", run_name=None):
    import modal

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new ZIP filename: {output}")
    volume = modal.Volume.from_name(volume_name)
    output.parent.mkdir(parents=True, exist_ok=True)
    # A temporary ZIP cannot be mistaken for a complete teammate handoff.
    temporary = output.with_suffix(output.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(f"Previous incomplete download exists: {temporary}")
    try:
        with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for run in selected_runs(manifest, seed=seed, run_id=run_name):
                for filename in FILES:
                    relative = f"{run['run_id']}/{filename}"
                    print(f"Downloading {relative}", flush=True)
                    with archive.open(relative, "w", force_zip64=True) as target:
                        for chunk in volume.read_file(f"/{relative}"):
                            target.write(chunk)
        temporary.rename(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"Send this ZIP to the person doing analysis: {output}")


def import_archives(manifest, archives, output):
    output = Path(output)
    allowed = {f"{run['run_id']}/{filename}" for run in manifest["runs"] for filename in FILES}
    members = set()
    # Validate every archive before writing any files. Accept only exact protocol paths.
    for path in archives:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if not names or len(names) != len(set(names)):
                raise ValueError(f"Empty archive or duplicate entries: {path}")
            for name in names:
                if name not in allowed or name in members:
                    raise ValueError(f"Unexpected or overlapping result: {name}")
                if (output / name).exists():
                    raise FileExistsError(f"Result already exists: {output / name}")
                members.add(name)
            run_ids = {name.split('/')[0] for name in names}
            expected = {f"{run_id}/{filename}" for run_id in run_ids for filename in FILES}
            if set(names) != expected:
                raise ValueError(f"Each run must contain its report and test predictions: {path}")
    for path in archives:
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                target = output / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source, target.open("xb") as destination:
                    import shutil
                    shutil.copyfileobj(source, destination)
    print(f"Imported {len(members) // len(FILES)} evaluated runs into {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("download", "import"))
    parser.add_argument("--manifest", type=Path, default=Path("experiments/paper21-modal.json"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--run-name", help="Download exactly this configuration/seed run")
    parser.add_argument("--archives", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--volume", default="music-genre-runs")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    if args.action == "download":
        if (args.seed is None) == (args.run_name is None) or args.archives:
            parser.error("download requires exactly one of --run-name or --seed, and no --archives")
        download_seed(manifest, args.seed, args.out, volume_name=args.volume, run_name=args.run_name)
    else:
        if not args.archives or args.seed is not None or args.run_name is not None:
            parser.error("import requires --archives and does not accept --seed or --run-name")
        import_archives(manifest, args.archives, args.out)


if __name__ == "__main__":
    main()
