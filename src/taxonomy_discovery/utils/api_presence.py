"""API-backed presence judge (e.g. GPT-5.x) with a FrozenJudge-compatible presence_grid.

The white-box FrozenJudge scores presence from digit-token logits, so it needs a LOCAL model.
This gives an INDEPENDENT frontier judge for the --eval-judge role: it re-scores the final
taxonomy's presence via structured 0-N ratings over the API, so reported fidelity / recovered
AUC cannot reflect the local (selection) judge's idiosyncrasy. Presence protocol: for each
(concept, response) ask the model to rate 0..score_max how strongly the reply exhibits the
behavior, parse the integer, normalize to [0,1] -- the API analogue of mode="scale".

Key is read from OPENAI_API_KEY (env, else parsed from .env; no python-dotenv dependency).
Judges concept x response pairs concurrently with a thread pool; caches per pair.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def _load_key(env_path: str = ".env", var: str = "OPENAI_API_KEY") -> str:
    if os.environ.get(var):
        return os.environ[var]
    for p in (env_path, ".env", os.path.expanduser("~/.env")):
        if p and os.path.exists(p):
            for line in open(p):
                line = line.strip()
                if line.startswith(var + "=") and not line.startswith("#"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(f"{var} not found in environment or .env")


class BudgetExceeded(RuntimeError):
    """--max-calls tripped. Raised INSTEAD of spending more; everything judged so far is
    already on disk in the pair cache, so the next run resumes from exactly here."""


class ApiJudge:
    def __init__(self, model: str, env_path: str = ".env", max_workers: int = 16, score_max: int = 10,
                 max_completion_tokens: int = 2048, reasoning_effort: str = "minimal", retries: int = 5,
                 max_text_chars: int = 1200, verbose: bool = False,
                 base_url: str | None = None, key_var: str = "OPENAI_API_KEY",
                 cache_path: str | None = None, max_calls: int | None = None):
        # base_url + key_var support any OpenAI-compatible endpoint, e.g. DeepSeek:
        #   ApiJudge("deepseek-v4-flash", base_url="https://api.deepseek.com",
        #            key_var="DEEPSEEK_API_KEY", reasoning_effort=None)
        from openai import OpenAI                                  # imported lazily so the dep is optional
        self.model = model
        self.client = OpenAI(api_key=_load_key(env_path, key_var), base_url=base_url)
        self.max_workers = max_workers; self.score_max = score_max
        self.max_completion_tokens = max_completion_tokens; self.reasoning_effort = reasoning_effort
        self.retries = retries; self.max_text_chars = max_text_chars; self.verbose = verbose
        self._use_effort = reasoning_effort is not None           # auto-disabled if the model rejects it
        self._cache: dict = {}; self._lock = threading.Lock()
        # PERSISTENT PAIR CACHE: append-only JSONL, one line per judged (concept, text). Every
        # paid call lands on disk immediately, so a crash/pause/balance-abort loses at most the
        # in-flight requests and a rerun re-pays for nothing. Append-only survives a truncated
        # final line (skipped on load), which a rewritten-whole-file cache would not.
        self.n_calls = 0; self.n_hits = 0
        self.max_calls = max_calls
        self._cache_path = cache_path; self._cf = None
        if cache_path:
            if os.path.exists(cache_path):
                bad = 0
                for line in open(cache_path, encoding="utf-8"):
                    try:
                        r = json.loads(line)
                        self._cache[r["k"]] = float(r["v"])
                    except Exception:  # noqa: BLE001,PERF203 (partial last line after a kill)
                        bad += 1
                print(f"[api-judge] pair cache {cache_path}: {len(self._cache)} judged pairs "
                      f"reused" + (f" ({bad} partial lines skipped)" if bad else ""), flush=True)
            else:
                os.makedirs(os.path.dirname(os.path.abspath(cache_path)) or ".", exist_ok=True)
            self._cf = open(cache_path, "a", encoding="utf-8")

    def _key(self, concept: str, text: str) -> str:
        # the MODEL is part of the key: two judges must never share a cached score
        h = hashlib.sha1(f"{self.model}\x00{concept}\x00{text}".encode()).hexdigest()
        return h

    def close(self):
        if self._cf:
            self._cf.flush(); self._cf.close(); self._cf = None

    def _create(self, prompt: str):                               # adaptive: tolerate models w/o reasoning_effort
        kw = dict(model=self.model, messages=[{"role": "user", "content": prompt}],
                  max_completion_tokens=self.max_completion_tokens)
        if self._use_effort:
            kw["reasoning_effort"] = self.reasoning_effort
        try:
            return self.client.chat.completions.create(**kw)
        except Exception as e:                                    # unknown-param -> drop reasoning_effort, retry once
            if self._use_effort and "reasoning_effort" in str(e):
                self._use_effort = False; kw.pop("reasoning_effort", None)
                return self.client.chat.completions.create(**kw)
            raise

    def _score_one(self, concept: str, text: str) -> float:
        key = self._key(concept, text)
        with self._lock:
            if key in self._cache:
                self.n_hits += 1
                return self._cache[key]
            if self.max_calls is not None and self.n_calls >= self.max_calls:
                raise BudgetExceeded(
                    f"--max-calls {self.max_calls} reached ({self.n_calls} paid calls this run, "
                    f"{self.n_hits} cache hits). Everything judged is cached at "
                    f"{self._cache_path or '<no cache file>'}; rerun to continue.")
            self.n_calls += 1
        prompt = (
            "You are annotating an AI assistant's reply for one specific response behavior.\n\n"
            f"{text[:self.max_text_chars]}\n\n"
            f"Behavior: \"the assistant tends to {concept}\"\n\n"
            f"On an integer scale from 0 to {self.score_max}, how strongly does the assistant's reply "
            f"exhibit THIS behavior? 0 = not at all, {self.score_max} = strongly and unmistakably. "
            "Judge only what the reply DOES, not its topic. Respond with ONLY the integer.")
        val = 0.0
        for a in range(self.retries):
            try:
                out = (self._create(prompt).choices[0].message.content or "").strip()
                m = re.findall(r"\d+", out)
                val = min(int(m[-1]), self.score_max) / self.score_max if m else 0.0
                break
            except Exception as e:  # noqa: BLE001
                # NON-retryable: an exhausted balance poisons every remaining pair with a
                # silent 0.0 -- fail the whole run instead (a zero-filled grid is worse
                # than a crash; callers early-save their generations). DeepSeek also ties
                # the CONCURRENCY cap to remaining balance, so a 429 that mentions balance
                # is the same failure wearing a rate-limit costume.
                msg = str(e)
                if "402" in msg or "Insufficient Balance" in msg or (
                        "429" in msg and "balance" in msg.lower()):
                    raise RuntimeError(
                        f"API balance exhausted ({e}); aborting instead of zero-filling"
                    ) from e
                if "429" in msg and a < self.retries - 1:         # plain rate limit: longer
                    time.sleep(min(5 * (a + 1), 60))              # backoff than transient
                    continue                                      # errors get
                if a == self.retries - 1:
                    if self.verbose:
                        print(f"[api-judge] gave up on a pair ({type(e).__name__}: {e})", flush=True)
                    val = 0.0
                else:
                    time.sleep(min(2 ** a, 30))                   # backoff for rate limits / transient errors
        with self._lock:
            self._cache[key] = val
            if self._cf:                                          # flush per pair: a kill -9
                self._cf.write(json.dumps({"k": key, "v": val}) + "\n")   # between calls still
                self._cf.flush()                                  # keeps every paid score
        return val

    def presence_grid(self, concepts, texts, text_batch: int = 48, grain: str = "response",
                      mode: str = "scale", verbose: bool = False):   # signature matches FrozenJudge
        pairs = [(ci, ti) for ci in range(len(concepts)) for ti in range(len(texts))]
        out = np.zeros((len(concepts), len(texts)), dtype=np.float32)
        done = [0]

        def work(p):
            ci, ti = p
            return p, self._score_one(concepts[ci], texts[ti])
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            for (ci, ti), v in ex.map(work, pairs):
                out[ci, ti] = v
                done[0] += 1
                if (verbose or self.verbose) and done[0] % 500 == 0:
                    print(f"[api-judge] {done[0]}/{len(pairs)} pairs", flush=True)
        return out
