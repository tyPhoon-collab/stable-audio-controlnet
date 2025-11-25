#!/usr/bin/env python3
"""Discord通知CLI - コマンドラインから Discordへメッセージを送信"""

import argparse
import sys
from pathlib import Path

# 親ディレクトリをPythonパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from main.notifier import DiscordNotifier


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Discord Webhook経由でメッセージを送信する",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 通常のメッセージを送信
  python -m scripts.notify_discord "訓練が完了しました"

  # 成功メッセージを送信（✅プリフィックス付き）
  python -m scripts.notify_discord "訓練が完了しました" --type success

  # エラーメッセージを送信（❌プリフィックス付き）
  python -m scripts.notify_discord "エラーが発生しました" --type error

  # Webhook URLを指定
  python -m scripts.notify_discord "メッセージ" --webhook "https://discord.com/api/webhooks/..."

環境変数:
  DISCORD_WEBHOOK_URL: デフォルト Webhook URL
        """,
    )

    parser.add_argument(
        "message",
        help="送信するメッセージ",
    )

    parser.add_argument(
        "--type",
        choices=["normal", "success", "error", "info"],
        default="normal",
        help="メッセージタイプ（デフォルト: normal）",
    )

    parser.add_argument(
        "--webhook",
        help="Discord Webhook URL（デフォルト: DISCORD_WEBHOOK_URL 環境変数から取得）",
    )

    args = parser.parse_args()

    try:
        notifier = DiscordNotifier(args.webhook)
    except ValueError as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1

    # メッセージタイプに応じて送信
    success = False
    if args.type == "success":
        success = notifier.send_success(args.message)
    elif args.type == "error":
        success = notifier.send_error(args.message)
    elif args.type == "info":
        success = notifier.send_info(args.message)
    else:
        success = notifier.send(args.message)

    if success:
        print("✅ メッセージを送信しました")
        return 0
    else:
        print("❌ メッセージの送信に失敗しました", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
