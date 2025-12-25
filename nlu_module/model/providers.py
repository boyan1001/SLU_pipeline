import importlib
import time
import json

from pathlib import Path

_MODULE = None
_RATE_NS = None
_RATE_LOCK = None
_CONC_SEM = None
_PROVIDER = None
_MODEL = None

def _resolve_module_name(provider: str) -> str:
    mapping = {
        "openai": "gpt_use",
        "gpt": "gpt_use",
        "gemini": "gemini_use",
        "ollama": "ollama_use",
    }
    return mapping.get(provider.lower(), provider.lower())

# Initialize the provider module and rate limiting resources
def init_worker(shared_ns, lock, sem, provider: str, model: str):
    """
    在每個 worker process 裡初始化：
    - 載入 .env（支援幾個常見路徑）
    - 設定全域的 rate limit 資源 (_RATE_NS, _RATE_LOCK, _CONC_SEM)
    - 匯入對應的 provider module
    - 將 rate limit 資源塞給舊版 provider module（如果有用到）
    """
    global _MODULE, _RATE_NS, _RATE_LOCK, _CONC_SEM, _PROVIDER, _MODEL

    from dotenv import load_dotenv
    import os
    import sys

    # 嘗試載入 .env
    env_paths = [
        Path.cwd() / ".env",                    # 當前目錄
        Path(__file__).parent.parent / ".env",  # 專案根目錄
    ]
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            break

    # 簡單檢查 Gemini key
    if provider.lower() == "gemini" and not os.getenv("GENAI_API_KEY"):
        print("[providers][worker] Warning: GENAI_API_KEY not found in worker process", file=sys.stderr)

    # 建立 / 掛載共享限流資源
    _RATE_NS = shared_ns
    _RATE_LOCK = lock
    _CONC_SEM = sem
    _PROVIDER = provider or "openai"
    _MODEL = model

    # 確保 namespace 至少有基本欄位，避免 None / 缺欄位
    if not hasattr(_RATE_NS, "rpm"):
        # rpm = None 代表「不啟用 RPM 限流」，但仍可用 semaphore 控制併發
        _RATE_NS.rpm = None
    if not hasattr(_RATE_NS, "count") or _RATE_NS.count is None:
        _RATE_NS.count = 0
    if not hasattr(_RATE_NS, "window_start") or _RATE_NS.window_start is None:
        _RATE_NS.window_start = time.time()

    module_name = _resolve_module_name(_PROVIDER)

    # Import provider module
    try:
        _MODULE = importlib.import_module(f"nlu_module.model.{module_name}")
    except Exception as e:
        import traceback
        print(f"[providers][import error] failed to import module nlu_module.model.{module_name}: {e}", file=sys.stderr)
        traceback.print_exc()
        _MODULE = None
        return

    # 把 rate limit 相關物件塞給舊版 provider，如果他們有用的話
    for name, val in (
        ("_RATE_NS", _RATE_NS),
        ("_RATE_LOCK", _RATE_LOCK),
        ("_CONC_SEM", _CONC_SEM),
    ):
        try:
            setattr(_MODULE, name, val)
        except Exception:
            pass

    # 如果 provider module 自己有 _init_worker，就呼叫他
    if hasattr(_MODULE, "_init_worker"):
        try:
            _MODULE._init_worker(shared_ns, lock, sem, model)
        except TypeError:
            # 舊版簽名不帶 model
            _MODULE._init_worker(shared_ns, lock, sem)


def _rate_limit_acquire():
    """
    進入一次 LLM 呼叫前的限流控制：
    - 如果 _RATE_NS 或 semaphore 沒設好 => 直接略過（不做任何限流）
    - 如果 rpm is None => 只控制併發，不做 RPM 限流
    - 否則 => 同時控制併發與每 60 秒的 request 數
    """
    global _RATE_NS, _RATE_LOCK, _CONC_SEM

    # 還沒初始化或沒用到限流資源，就直接跳過
    if _RATE_NS is None or _CONC_SEM is None:
        return

    # 先拿併發 semaphore
    _CONC_SEM.acquire()

    # 沒鎖就不做 RPM 管控，只做併發控制
    if _RATE_LOCK is None:
        return

    rpm = getattr(_RATE_NS, "rpm", None)

    # rpm=None => 不做 RPM 限流，僅限制併發
    if rpm is None or rpm <= 0:
        return

    # 防止 count 未初始化
    if getattr(_RATE_NS, "count", None) is None:
        _RATE_NS.count = 0
    if getattr(_RATE_NS, "window_start", None) is None:
        _RATE_NS.window_start = time.time()

    # 真正的 RPM 限流邏輯
    while True:
        now = time.time()
        with _RATE_LOCK:
            # 視窗超過 60 秒就 reset
            if now - _RATE_NS.window_start >= 60.0:
                _RATE_NS.window_start = now
                _RATE_NS.count = 0

            # 還沒超過 rpm，可以直接執行
            if _RATE_NS.count < rpm:
                _RATE_NS.count += 1
                break

            # 超過 rpm，就睡到下一個視窗
            sleep_s = max(0.0, 60.0 - (now - _RATE_NS.window_start))

        # 為了讓 sleep 有機會被打斷，最多睡 1 秒再回圈
        time.sleep(min(1.0, sleep_s))


def _rate_limit_release():
    """在一次 LLM 呼叫結束後釋放併發 semaphore。"""
    global _RATE_NS, _CONC_SEM
    if _RATE_NS is not None and _CONC_SEM is not None:
        try:
            _CONC_SEM.release()
        except Exception:
            # 就算 release 失敗也不要連帶把 main flow 弄崩
            pass


def ask(prompt: str, task: str, llm_config: dict = None):
    """
    公用入口：
    - 檢查 task 是否為 intent_detection / slot_filling
    - 根據 llm_config 決定 provider / model
    - 執行 rate limit（如果有設定）
    - 呼叫 provider module 的 ask_* 或 ask()
    """
    if task not in {"intent_detection", "slot_filling"}:
        raise ValueError("Invalid task type. Must be 'intent_detection' or 'slot_filling'.")

    global _MODULE, _PROVIDER, _MODEL

    # llm_config 可能是 list（例如 config/llm_setting.json）
    if isinstance(llm_config, list):
        import sys
        print("[providers][info] llm_config is a list; using first entry as active config", file=sys.stderr)
        if len(llm_config) == 0:
            llm_config = {}
        else:
            # 如果已經有 _PROVIDER，盡量挑 provider 一致的那個
            if _PROVIDER:
                matched = None
                for c in llm_config:
                    try:
                        if c.get("provider") == _PROVIDER:
                            matched = c
                            break
                    except Exception:
                        continue
                llm_config = matched or llm_config[0]
            else:
                llm_config = llm_config[0]

    # 如果 worker 還沒透過 init_worker 初始化，就在這裡 lazy import
    if _MODULE is None:
        import sys, traceback

        provider = llm_config.get("provider", "openai") if llm_config else "openai"
        model = llm_config.get("model", "gpt-5-mini") if llm_config else "gpt-5-mini"

        module_name = _resolve_module_name(provider)
        try:
            _MODULE = importlib.import_module(f"nlu_module.model.{module_name}")
            _PROVIDER = provider
            _MODEL = model
        except Exception as e:
            print(f"[providers][import error] failed to import module nlu_module.model.{module_name}: {e}", file=sys.stderr)
            traceback.print_exc()
            return None

    # 進入一次 LLM 呼叫的限流區
    _rate_limit_acquire()

    try:
        # 優先找 ask_{provider}，找不到就 fallback 到 ask()
        fn = getattr(_MODULE, f"ask_{_PROVIDER.lower()}", None)
        if fn is None:
            fn = getattr(_MODULE, "ask", None)
        if fn is None:
            raise NotImplementedError(
                f"Provider module '{_MODULE.__name__}' missing 'ask_{_PROVIDER}' and 'ask'"
            )

        try:
            return fn(prompt, task, llm_config)
        except Exception as e:
            import sys, traceback
            print(f"[providers][runtime error] provider={_PROVIDER} model={_MODEL} error={e}", file=sys.stderr)
            traceback.print_exc()
            return None
    finally:
        # 結束時記得釋放併發 semaphore
        _rate_limit_release()
