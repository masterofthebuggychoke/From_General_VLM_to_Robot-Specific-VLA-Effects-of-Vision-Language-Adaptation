"""Create the downloadable code snapshot from the standalone repository folder."""
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "code"
    destination = root / "site" / "downloads" / "code.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as tmp:
        temp = Path(tmp) / "code.zip"
        with ZipFile(temp, "w", compression=ZIP_DEFLATED, compresslevel=8) as archive:
            for path in sorted(source.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts and ".git" not in path.parts and path.suffix != ".pyc":
                    archive.write(path, path.relative_to(root))
        temp.replace(destination)
    print(destination)


if __name__ == "__main__":
    main()
