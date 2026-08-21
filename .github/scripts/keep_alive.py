"""Streamlit Community Cloud のアプリを起こしておくスクリプト。

12時間アクセスがないとアプリは休止し、訪問者は復帰ボタンを押して20〜30秒待たされる。
note記事からの読者を入口で失わないよう、GitHub Actions から定期的に実行する。

curl や requests では起こせない点に注意。Streamlit は
  JSの実行 → /_stcore/stream への WebSocket 接続
まで進んで初めて Python プロセスが起動するため、実ブラウザでの訪問が必要になる。
"""
import os
import sys

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

# ?keepalive=1 はこのアクセスが自動巡回であることの目印。アプリ側がこれを見て
# GA4への計測を止める(付けないと1日6件の「利用者」として数えられてしまう)
APP_URL = os.environ.get("APP_URL", "https://tomonavi-forecast.streamlit.app/?keepalive=1")

# 休止中に出る復帰ボタン。「Yes, get this app back up!」の部分一致で拾う
WAKE_BUTTON_TEXT = "get this app back up"

# 起動判定に使う文字列。Streamlit Cloud はアプリ本体を iframe に入れて配信するため、
# 親ページの body からは本文を読めない。一方タイトルは st.set_page_config の値が反映され、
# 休止中は "Streamlit"、起動後は "需要予測アプリ · Streamlit" になるので判定に使える
READY_MARKER = "需要予測アプリ"

# 休止画面自体もJSで描画されるため、ボタンの有無はこの秒数まで待って判断する
BUTTON_WAIT_MS = 20_000
# 復帰ボタンを押した後の待ち時間。起動には環境の再構築が走るので長めにとる
WAKE_READY_WAIT_MS = 180_000
# もともと起きていた場合の待ち時間。すぐ表示されるはずなので短くてよい
AWAKE_READY_WAIT_MS = 60_000


def log(message: str, *, error: bool = False) -> None:
    """進捗を即時に出す。

    CI ではログがパイプに流れるため既定では出力が溜め込まれ、途中で打ち切られた際に
    どこまで進んだか分からなくなる。原因を追えるよう1行ずつ吐き出す。
    """
    print(message, file=sys.stderr if error else sys.stdout, flush=True)


def main() -> int:
    # ローカル(Windows)実行でも日本語ログが化けないようにする
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    with sync_playwright() as p:
        # 公式Playwrightイメージは root で動くため、サンドボックスを切らないと
        # Chromium が起動できない
        browser = p.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page()
        log(f"訪問: {APP_URL}")
        page.goto(APP_URL, wait_until="domcontentloaded", timeout=60_000)
        log("ページ読み込み完了。復帰ボタンの有無を確認する")

        # 休止中なら復帰ボタンを押す。起きていればボタンは現れないのでタイムアウトで抜ける
        button = page.get_by_text(WAKE_BUTTON_TEXT, exact=False).first
        try:
            button.wait_for(state="visible", timeout=BUTTON_WAIT_MS)
            log("休止中だったので復帰ボタンを押す")
            button.click()
            was_asleep = True
        except PlaywrightTimeoutError:
            log("復帰ボタンは出なかった(すでに起動中とみられる)")
            was_asleep = False

        # 起動を確認する。待つ長さと、確認できなかったときの扱いは状況で変える
        timeout_ms = WAKE_READY_WAIT_MS if was_asleep else AWAKE_READY_WAIT_MS
        log(f"起動確認を待つ(最大 {timeout_ms // 1000} 秒)")
        try:
            page.wait_for_function(
                "marker => document.title.includes(marker)",
                arg=READY_MARKER,
                timeout=timeout_ms,
            )
            log(f"起動を確認: タイトルが {page.title()!r} になった")
            status = 0
        except PlaywrightTimeoutError:
            if was_asleep:
                # 復帰要求はサーバ側に届いており、ブラウザを閉じても起動は続く。
                # 起動が遅いだけのことが多いので、警告に留めて成功扱いにする
                log("復帰は要求済み。起動確認は時間内に取れなかった(処理は継続中とみられる)")
                status = 0
            else:
                # 起きているはずなのに表示されない=異常。気づけるように失敗で返す
                log("起動中のはずが画面を確認できなかった", error=True)
                status = 1

        browser.close()
        log(f"終了(status={status})")
        return status


if __name__ == "__main__":
    # 例外で落ちると GitHub Actions には「exit code 1」としか残らず、原因が追えない。
    # 実行環境の情報と併せて記録してから終了する
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        import platform
        import traceback

        log("=== 想定外のエラーで終了 ===", error=True)
        log(f"python: {sys.version}", error=True)
        log(f"platform: {platform.platform()}", error=True)
        try:
            import playwright

            log(f"playwright: {playwright.__version__}", error=True)
        except Exception:
            log("playwright: バージョンを取得できず", error=True)
        log(traceback.format_exc(), error=True)
        raise SystemExit(1)
