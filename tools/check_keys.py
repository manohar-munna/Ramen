"""Tries every key in GEMINI_API_KEYS and reports which ones work.

Two probes per key. Listing models is free and says whether the credential is accepted
at all; one tiny generation says whether it still has allowance, which is the thing that
actually stops a run. Keys are never printed in full.

    python tools/check_keys.py [-m MODEL]
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.getcwd())

import enhancer  # noqa: E402

LIST_URL = "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1"


def mask(key: str) -> str:
    return key[:6] + "..." + key[-4:] if len(key) > 12 else "(short)"


def accepted(key: str, timeout: int = 45):
    """Whether the API recognises this credential at all."""
    req = urllib.request.Request(LIST_URL, headers={"x-goog-api-key": key})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            json.load(resp)
        return True, ""
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        return False, "%d %s" % (e.code, enhancer.short_reason(
            enhancer.EnhancementError("x: " + body)))
    except Exception as e:
        return False, enhancer.redact_keys(str(e))[:80]


def has_allowance(key: str, model: str, timeout: int = 60):
    """Whether a real generation still goes through on it."""
    try:
        enhancer.call_gemini("Reply with: OK", api_key=key, model=model,
                             timeout=timeout, attempts=1)
        return True, ""
    except enhancer.EnhancementError as e:
        return False, enhancer.short_reason(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default=enhancer.DEFAULT_MODEL)
    args = ap.parse_args()

    keys = enhancer.api_keys()
    if not keys:
        print("No keys. Put GEMINI_API_KEYS=key1,key2,... in .env")
        return 2

    print("%d key(s) in .env, probing against %s\n" % (len(keys), args.model))
    usable = []
    for i, key in enumerate(keys, 1):
        ok, why = accepted(key)
        if not ok:
            print("%d. %-18s rejected outright - %s" % (i, mask(key), why))
            continue
        ok2, why2 = has_allowance(key, args.model)
        if ok2:
            usable.append(key)
            print("%d. %-18s WORKS" % (i, mask(key)))
        else:
            print("%d. %-18s accepted, but %s" % (i, mask(key), why2))

    print("\n%d of %d key(s) can generate on %s." % (len(usable), len(keys), args.model))
    if usable:
        print("The engine tries them in this order and skips a spent one for the rest "
              "of the run.")
    return 0 if usable else 1


if __name__ == "__main__":
    raise SystemExit(main())
