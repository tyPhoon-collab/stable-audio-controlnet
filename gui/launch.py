#!/usr/bin/env python3
"""
Stable Audio ControlNet Chord GUI - Docker起動用スクリプト

Docker環境での実行に最適化されたシンプルな起動スクリプト
"""

import os
import sys


def main():
    """メイン関数"""
    print("🎵 Stable Audio ControlNet - Chord Progression GUI (Docker)")
    print("=" * 60)

    try:
        # プロジェクトディレクトリをPythonパスに追加
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, project_root)

        # アプリケーションをインポート・起動
        from app import main as app_main

        app_main()

    except Exception as e:
        print(f"❌ アプリケーションの起動エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
