import os


def _maybe_apply_sglang_patch() -> None:
    if os.environ.get("ENABLE_SGLANG_DSV4_A100_PATCH", "0") != "1":
        return

    from dsv4_a100_patch import apply_patch

    apply_patch()


_maybe_apply_sglang_patch()
