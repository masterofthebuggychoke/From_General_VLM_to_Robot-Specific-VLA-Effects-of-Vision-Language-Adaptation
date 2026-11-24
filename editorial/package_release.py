"""Build a repository-root GitHub upload archive from an assembled project checkout."""
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile


def main():
    root = Path(__file__).resolve().parents[1]
    output = root.parent / "SmolVLA-GitHub-project.zip"
    included = [root / name for name in (".github", "README.md", "site", "code", "editorial", "report")]
    paths = []
    for item in included:
        paths.extend(item.rglob("*") if item.is_dir() else [item])
    with TemporaryDirectory(dir=output.parent) as tmp:
        staged = Path(tmp) / output.name
        with ZipFile(staged, "w", compression=ZIP_DEFLATED, compresslevel=7) as archive:
            for path in sorted(paths):
                if not path.is_file() or path.suffix == ".pyc" or "__pycache__" in path.parts:
                    continue
                # Visitors download site/report.pdf; the manuscript can regenerate the loose PDF.
                if path.parent == root / "report" and path.suffix == ".pdf":
                    continue
                archive.write(path, path.relative_to(root))
        staged.replace(output)
    print(output)


if __name__ == "__main__":
    main()
