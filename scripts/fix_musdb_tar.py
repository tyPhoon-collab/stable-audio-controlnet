"""musdb18hq の WebDataset tar を修復するユーティリティ。

- 拡張子直前のキー名が短縮されてしまったトラックを所定の名前にリネームする
- リネーム後に同一トラックのステムが連続するよう並び替えた tar を再生成する

使用例:
  python fix_musdb_tar.py --input /app/data/musdb18hq/test.tar \
      --output /app/data/musdb18hq/test_fixed.tar \
      --rename M="MERC Music - Knockout"
"""

import argparse
import io
import tarfile
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

STEM_ORDER = {"bass": 0, "drums": 1, "other": 2, "vocals": 3}
DEFAULT_RENAME = {"M": "MERC Music - Knockout"}


def parse_rename_args(values: Iterable[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--rename に不正な形式の値があります: '{value}'")
        key, new_name = value.split("=", 1)
        key = key.strip()
        new_name = new_name.strip()
        if not key or not new_name:
            raise ValueError(f"--rename に空文字列は使えません: '{value}'")
        mapping[key] = new_name
    return mapping


def split_entry_name(name: str) -> Tuple[str, str, str]:
    """<曲名>.<ステム>.<拡張子> に分割する。"""
    parts = name.rsplit(".", 2)
    if len(parts) != 3:
        raise ValueError(f"期待しないファイル名です: {name}")
    song, stem, ext = parts
    return song, stem, ext


def collect_entries(tar_path: Path) -> List[Tuple[tarfile.TarInfo, bytes]]:
    entries: List[Tuple[tarfile.TarInfo, bytes]] = []
    with tarfile.open(tar_path, "r") as tar:
        for member in tar:
            if not member.isfile():
                continue
            data = tar.extractfile(member)
            if data is None:
                continue
            entries.append((member, data.read()))
    return entries


def build_grouped_entries(
    entries: List[Tuple[tarfile.TarInfo, bytes]],
    rename_map: Dict[str, str],
) -> Tuple[OrderedDict, Dict[str, List[str]]]:
    order = OrderedDict()
    grouped = defaultdict(list)

    for member, payload in entries:
        song, stem, ext = split_entry_name(member.name)
        new_song = rename_map.get(song, song)

        if new_song not in order:
            order[new_song] = None
        grouped[new_song].append((stem, ext, payload, member))

    return order, grouped


def write_fixed_tar(
    output_path: Path,
    order: OrderedDict,
    grouped,
) -> None:
    with tarfile.open(output_path, "w") as tar:
        for song in order:
            entries = grouped[song]
            entries.sort(key=lambda item: STEM_ORDER.get(item[0], 99))

            for stem, ext, payload, src_info in entries:
                name = f"{song}.{stem}.{ext}"
                info = tarfile.TarInfo(name=name)
                info.size = len(payload)
                info.mode = src_info.mode
                info.mtime = src_info.mtime
                tar.addfile(info, io.BytesIO(payload))


def warn_incomplete_tracks(grouped) -> None:
    for song, entries in grouped.items():
        stems = {stem for stem, _, _, _ in entries}
        missing = [s for s in STEM_ORDER if s not in stems]
        if missing:
            print(f"⚠️  {song} は {missing} が欠けています")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix musdb WebDataset tar ordering and names"
    )
    parser.add_argument("--input", required=True, help="修正対象の tar ファイル")
    parser.add_argument("--output", required=True, help="修正後の tar ファイル出力先")
    parser.add_argument(
        "--rename",
        action="append",
        default=[],
        help="短縮されたキー名を '旧名=新名' 形式で指定 (複数指定可)",
    )
    parser.add_argument(
        "--no-default-rename",
        action="store_true",
        help="組み込みのデフォルトマッピング (M→MERC Music - Knockout) を無効化",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    rename_map = {} if args.no_default_rename else dict(DEFAULT_RENAME)
    rename_map.update(parse_rename_args(args.rename))

    print(f"📦 読み込み中: {input_path}")
    entries = collect_entries(input_path)
    print(f"   読み込んだエントリ数: {len(entries)}")

    order, grouped = build_grouped_entries(entries, rename_map)
    warn_incomplete_tracks(grouped)

    print(f"💾 書き出し: {output_path}")
    write_fixed_tar(output_path, order, grouped)
    print("✅ 完了")


if __name__ == "__main__":
    main()
