# based on https://gist.github.com/scivision/ad241e9cf0474e267240e196d7545eca

import os
import sys
import tarfile
import tempfile
from pathlib import Path

try:
    import zstandard  # pip install zstandard
except ModuleNotFoundError:
    zstandard = None


def extract_zst(archive: Path, out_path: Path):
    """
    Extract content of zst file to a target file system directory.

    works on Windows, Linux, MacOS, etc.

    Parameters
    ----------
    archive: pathlib.Path or str
      .zst file to extract
    out_path: pathlib.Path or str
      directory to extract files and directories to
    """
    if zstandard is None:
        raise ImportError("pip install zstandard")

    archive = Path(archive).expanduser()
    out_path = Path(out_path).expanduser().resolve()
    # need .resolve() in case intermediate relative dir doesn't exist

    dctx = zstandard.ZstdDecompressor()

    with tempfile.TemporaryFile(suffix=".tar") as ofh:
        with archive.open("rb") as ifh:
            dctx.copy_stream(ifh, ofh)
        ofh.seek(0)
        with tarfile.open(fileobj=ofh) as z:
            z.extractall(out_path)


def compress_zst(in_path: Path, archive: Path):
    """
    Compress a directory into a .tar.zst file.

    Certain hidden files are excluded, including .git directories and
    macOS's .DS_Store files.

    Parameters
    ----------
    in_path: pathlib.Path or str
      directory to compress
    archive: pathlib.Path or str
      .tar.zst file to compress into
    """
    if zstandard is None:
        raise ImportError("pip install zstandard")
    dctx = zstandard.ZstdCompressor(level=9, threads=-1, write_checksum=True)
    with tempfile.TemporaryFile(suffix=".tar") as ofh:
        with tarfile.open(fileobj=ofh, mode="w") as z:
            for dirpath, dirnames, filenames in os.walk(in_path):
                if os.path.basename(dirpath) == ".git":
                    continue
                for n in range(len(dirnames) - 1, -1, -1):
                    if dirnames[n] == ".git" or dirnames[n].startswith("---"):
                        dirnames.pop(n)
                for f in filenames:
                    if f.startswith(".git") or f == ".DS_Store" or f.startswith("---"):
                        continue
                    finame = Path(os.path.join(dirpath, f))
                    arcname = finame.relative_to(in_path)
                    print(f"> {arcname}")
                    z.add(finame, arcname=arcname)
        ofh.seek(0)
        with archive.open("wb") as ifh:
            dctx.copy_stream(ofh, ifh)


if __name__ == "__main__":
    x = Path(sys.argv[1])
    name = x.name
    if name.endswith(".tar.zst"):
        y = x.with_name(name[:-8])
        if x.exists():
            if not y.exists():
                print(f"extracting from: {x}")
                extract_zst(x, y)
            else:
                print(f"not extracting, existing target: {y}")
        else:
            print(f"not extracting, does not exist: {x}")
    else:
        y = x.with_name(name + ".tar.zst")
        if x.exists():
            if not y.exists():
                print(f"compressing to tar.zst: {x}")
                compress_zst(x, y)
            else:
                print(f"not compressing, existing tar.zst: {x}")
        else:
            print(f"not compressing, does not exist: {x}")
