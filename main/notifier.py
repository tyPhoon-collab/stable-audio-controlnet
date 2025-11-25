"""Discord通知ユーティリティ"""

import os

import requests
from dotenv import load_dotenv

# .envファイルを読み込む
load_dotenv()


class DiscordNotifier:
    """Discord Webhook経由でメッセージを送信するクラス"""

    def __init__(self, webhook_url: str | None = None) -> None:
        """
        Args:
            webhook_url: Discord Webhook URL。
                        指定されない場合は DISCORD_WEBHOOK_URL 環境変数から取得
        """
        self.webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")

        if not self.webhook_url:
            raise ValueError(
                "Discord Webhook URL is not set. "
                "Provide it as an argument or set DISCORD_WEBHOOK_URL environment variable."
            )

    def send(self, message: str) -> bool:
        """
        Discord にメッセージを送信

        Args:
            message: 送信するメッセージ

        Returns:
            成功した場合は True、失敗した場合は False
        """
        if not self.webhook_url:
            print("Discord Webhook URL is not configured")
            return False

        data = {"content": message}
        try:
            response = requests.post(self.webhook_url, json=data)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            print(f"Failed to send message to Discord: {e}")
            return False

    def send_error(self, error_message: str) -> bool:
        """エラーメッセージを送信（プリフィックス付き）"""
        return self.send(f"❌ エラー: {error_message}")

    def send_success(self, message: str) -> bool:
        """成功メッセージを送信（プリフィックス付き）"""
        return self.send(f"✅ {message}")

    def send_info(self, message: str) -> bool:
        """情報メッセージを送信（プリフィックス付き）"""
        return self.send(f"ℹ️ {message}")


def get_notifier(webhook_url: str | None = None) -> "DiscordNotifier | None":
    """
    Discord通知機能が利用可能な場合、DiscordNotifierインスタンスを返す

    Args:
        webhook_url: Discord Webhook URL（オプション）

    Returns:
        DiscordNotifier インスタンス、または利用不可の場合は None
    """
    try:
        return DiscordNotifier(webhook_url)
    except ValueError:
        return None
