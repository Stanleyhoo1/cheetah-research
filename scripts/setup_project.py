"""
Point cheetah-dlc/config.yaml at this copy of the repo.

DeepLabCut finds labeled-data/, training-datasets/ and dlc-models-pytorch/ through the
project_path line in config.yaml, so it has to match where the repo is on this machine.
Run this once after cloning, and again after any git pull that changes config.yaml.

Only the project_path line is changed; the rest of the file (which is tracked in git) stays
byte-for-byte the same. DeepLabCut would also fix the path on its own the next time it reads
the file, but it saves the path exactly as it was given (often a relative one such as
"cheetah-dlc"), so this sets the full path once, up front.

Usage (any Python 3, DeepLabCut not needed):
    python scripts/setup_project.py
    python scripts/setup_project.py --config path/to/config.yaml
"""
import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LINE = re.compile(r"^project_path:[^\n]*$", re.M)


def yaml_value(path):
    """The path as a YAML value, quoted only if plain YAML would misread it."""
    s = str(path)
    if re.search(r":\s|\s#|^[\s'\"!&*?|>%@`{}\[\],-]|\s$", s):
        return "'" + s.replace("'", "''") + "'"
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "cheetah-dlc" / "config.yaml"),
                    help="project settings file (default: cheetah-dlc/config.yaml in this repo)")
    args = ap.parse_args()

    # Same rule DeepLabCut uses: the project is the folder config.yaml is in (absolute, not resolved).
    config = Path(args.config).expanduser().absolute()
    if not config.is_file():
        sys.exit(f"No config.yaml at {config}")
    project = config.parent

    with open(config, newline="") as f:
        text = f.read()
    found = LINE.findall(text)
    if len(found) != 1:
        sys.exit(f"Expected one 'project_path:' line in {config}, found {len(found)}.")
    old = found[0].split(":", 1)[1].strip().strip("'\"")

    if Path(old) == project:
        print(f"project_path is already {project}")
        return
    text = LINE.sub(lambda m: f"project_path: {yaml_value(project)}", text, count=1)
    with open(config, "w", newline="") as f:
        f.write(text)
    print(f"project_path: {old}\n           -> {project}")


if __name__ == "__main__":
    main()
