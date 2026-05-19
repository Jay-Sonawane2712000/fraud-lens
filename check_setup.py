from pathlib import Path


REQUIRED_FOLDERS = [
    "data/raw",
    "data/processed",
    "modules",
    "models/saved",
    "dashboard",
    "notebooks",
    "outputs/features",
    "outputs/figures",
    "outputs/reports",
    "tests",
]

REQUIRED_FILES = [
    "requirements.txt",
    "README.md",
    ".gitignore",
]


def main() -> None:
    project_root = Path.cwd()
    print(f"Project path: {project_root}")

    missing_folders = [
        folder for folder in REQUIRED_FOLDERS if not (project_root / folder).is_dir()
    ]
    missing_files = [
        file_name for file_name in REQUIRED_FILES if not (project_root / file_name).is_file()
    ]

    print("\nFolder check:")
    for folder in REQUIRED_FOLDERS:
        status = "OK" if (project_root / folder).is_dir() else "MISSING"
        print(f"  {status}: {folder}")

    print("\nFile check:")
    for file_name in REQUIRED_FILES:
        status = "OK" if (project_root / file_name).is_file() else "MISSING"
        print(f"  {status}: {file_name}")

    print("\nRaw data folder check:")
    print(f"  {'OK' if (project_root / 'data/raw').is_dir() else 'MISSING'}: data/raw")

    if missing_folders or missing_files:
        print("\nPhase 0 setup is incomplete.")
        if missing_folders:
            print(f"Missing folders: {', '.join(missing_folders)}")
        if missing_files:
            print(f"Missing files: {', '.join(missing_files)}")
        raise SystemExit(1)

    print("\nSuccess: Phase 0 project structure is ready.")


if __name__ == "__main__":
    main()
