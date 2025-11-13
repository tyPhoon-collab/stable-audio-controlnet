"""
ジャンル情報マッピングのテスト
"""

from pathlib import Path

import pytest

from main.data.common_mapping import GenreMapping


@pytest.fixture(autouse=True)
def reset_genre_mapping():
    GenreMapping._instance = None
    yield
    GenreMapping._instance = None


def test_genre_mapping_loads_csv():
    """実データのCSVを読み込めることを確認"""
    csv_path = "/app/data/tracklist.csv"
    assert Path(csv_path).exists(), f"CSV file not found: {csv_path}"

    mapper = GenreMapping()
    mapping = mapper.load_mapping(csv_path)

    # CSVに複数のエントリが存在することを確認
    assert len(mapping) > 0, "CSV should have at least one entry"
    assert "A Classic Education - NightOwl" in mapping
    assert "Actions - Devil's Words" in mapping

    # ジャンルが正しく読み込まれていることを確認
    assert mapping["A Classic Education - NightOwl"] == "Singer/Songwriter"
    assert mapping["Actions - Devil's Words"] == "Pop/Rock"


def test_genre_mapping_get_genre():
    """get_genre メソッドが正しく動作することを確認"""
    csv_path = "/app/data/tracklist.csv"
    mapper = GenreMapping()
    mapper.load_mapping(csv_path)

    # 存在するトラックのジャンルを取得
    genre = mapper.get_genre("A Classic Education - NightOwl")
    assert genre == "Singer/Songwriter"

    # 存在しないトラックは デフォルト値を返す
    genre = mapper.get_genre("NonExistent Track")
    assert genre == "Unknown"

    # カスタムデフォルト値を指定
    genre = mapper.get_genre("NonExistent Track", default="Unknown Genre")
    assert genre == "Unknown Genre"


def test_genre_mapping_matching_with_tar_keys():
    """tarファイルのキーとCSV内のキーの対応付けをテスト"""
    import tarfile

    tar_path = "/app/data/musdb18hq/train.tar"
    csv_path = "/app/data/tracklist.csv"

    assert Path(tar_path).exists(), f"Tar file not found: {tar_path}"
    assert Path(csv_path).exists(), f"CSV file not found: {csv_path}"

    # tarから全トラックキーを抽出
    tar_keys = set()
    with tarfile.open(tar_path, "r") as tar:
        for member in tar:
            base_name = member.name.rsplit(".", 2)[0]
            tar_keys.add(base_name)

    # CSVを読み込み
    mapper = GenreMapping()
    csv_mapping = mapper.load_mapping(csv_path)
    csv_keys = set(csv_mapping.keys())

    # 対応付けを確認
    matched_direct = tar_keys & csv_keys
    unmatched = tar_keys - csv_keys

    print(
        f"\ntar keys: {len(tar_keys)}, CSV keys: {len(csv_keys)}, "
        f"direct matched: {len(matched_direct)}, unmatched: {len(unmatched)}"
    )

    # 実際にいくつか確認
    test_track = "A Classic Education - NightOwl"
    assert test_track in tar_keys, f"{test_track} should be in tar"
    assert test_track in csv_keys, f"{test_track} should be in CSV"
    genre = mapper.get_genre(test_track)
    assert genre != "", f"Should get genre for {test_track}"
    assert genre == "Singer/Songwriter"

    # 対応付け結果を表示
    if unmatched:
        print("\n⚠️  Unmatched tar keys:")
        for key in list(unmatched)[:5]:
            print(f"    {key}")
    else:
        print("\n✅ All tar keys directly match CSV keys")


def test_genre_coverage_all_tar_files():
    """train.tar と test_ordered.tar の両方でジャンルカバレッジをチェック"""
    import tarfile

    tar_files = [
        ("/app/data/musdb18hq/train.tar", "train"),
        ("/app/data/musdb18hq/test_ordered.tar", "test_ordered"),
    ]
    csv_path = "/app/data/tracklist.csv"

    assert Path(csv_path).exists(), f"CSV file not found: {csv_path}"

    # CSVを一度読み込み
    mapper = GenreMapping()
    csv_mapping = mapper.load_mapping(csv_path)
    csv_keys = set(csv_mapping.keys())

    all_results = {}
    total_unmatched = 0

    for tar_path, tar_name in tar_files:
        if not Path(tar_path).exists():
            print(f"⚠️  {tar_path} not found, skipping")
            continue

        # tarから全トラックキーを抽出
        tar_keys = set()
        with tarfile.open(tar_path, "r") as tar:
            for member in tar:
                base_name = member.name.rsplit(".", 2)[0]
                tar_keys.add(base_name)

        # 対応付けを確認
        matched = tar_keys & csv_keys
        unmatched = tar_keys - csv_keys

        all_results[tar_name] = {
            "tar_count": len(tar_keys),
            "matched": len(matched),
            "unmatched": len(unmatched),
            "unmatched_keys": unmatched,
        }

        total_unmatched += len(unmatched)

    # 結果を表示
    print("\n=== Genre Coverage Report ===")
    for tar_name, result in all_results.items():
        coverage = (
            100 * result["matched"] / result["tar_count"]
            if result["tar_count"] > 0
            else 0
        )
        print(
            f"\n{tar_name}:"
            f"\n  Total tracks: {result['tar_count']}"
            f"\n  Matched: {result['matched']}"
            f"\n  Unmatched: {result['unmatched']}"
            f"\n  Coverage: {coverage:.1f}%"
        )

        if result["unmatched"] > 0:
            print("  Unmatched keys:")
            for key in sorted(result["unmatched_keys"])[:10]:
                print(f"    - {key}")
            if result["unmatched"] > 10:
                print(f"    ... and {result['unmatched'] - 10} more")

    # 全体サマリー
    print("\n=== Summary ===")
    print(f"CSV entries: {len(csv_keys)}")
    print(f"Total unmatched across all tar files: {total_unmatched}")

    # アサーション: すべてのトラックがカバーされていることを確認
    assert total_unmatched == 0, (
        f"Found {total_unmatched} unmatched tracks across tar files. "
        "All tar tracks should have genres in CSV."
    )
    print("\n✅ All tar tracks have matching genre info in CSV")
