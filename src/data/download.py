from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from tqdm import tqdm


def url_filename(url: str) -> str:
    name = Path(urlparse(url).path).name
    if not name:
        raise ValueError(f"Could not infer filename from URL: {url}")
    return name


def download_file(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urlopen(url) as response:
        total = response.headers.get("Content-Length")
        total_bytes = int(total) if total is not None else None
        with tmp.open("wb") as out, tqdm(
            total=total_bytes,
            unit="B",
            unit_scale=True,
            desc=dest.name,
        ) as progress:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                progress.update(len(chunk))
    tmp.replace(dest)
    return dest


def ensure_raw_files(urls: list[str], dest_dir: Path) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for url in urls:
        dest = dest_dir / url_filename(url)
        if dest.exists() and dest.stat().st_size > 0:
            paths.append(dest)
            continue
        paths.append(download_file(url, dest))
    return paths
