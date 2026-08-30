"""Application settings, sourced from environment / .env.

Nothing in this app hardcodes a path, threshold or normalisation constant --
everything routes through the Settings singleton below.

Note on precedence: values that are ALSO recorded inside the checkpoint
(input size, normalisation, class names, threshold) are treated as fallbacks
here. `model_loader` prefers what the checkpoint itself declares, so serving
config cannot silently drift away from how the model was trained.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import torch
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository layout anchor: .../backend/app/config.py -> .../backend
BACKEND_ROOT: Path = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime configuration. Field names map to upper-case env vars."""

    # pydantic v2 reserves the `model_` prefix for its own API; several of our
    # settings legitimately start with it, so the protection is disabled.
    model_config = SettingsConfigDict(
        # Anchored to backend/, not the process CWD, so `uvicorn app.main:app`
        # finds the same .env no matter which directory it was launched from.
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
    )

    # --- model -------------------------------------------------------------
    model_path: Path = Field(
        default=BACKEND_ROOT / "models" / "EfficientNetB0_Dataset02_OralCancer_EXP4A.pkl",
        description="Path to the trained checkpoint (.pkl / .pth).",
    )
    device: Literal["auto", "cpu", "cuda"] = "auto"

    # --- decision layer ----------------------------------------------------
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    # Index 1 == Cancer. VERIFIED against the checkpoint's own metadata:
    #   class_names: {0: 'NON CANCER', 1: 'CANCER'}
    # model_loader re-checks this at startup and logs loudly on a mismatch,
    # because an inverted mapping is the one bug that fails silently.
    positive_class_index: int = Field(default=1, ge=0, le=1)

    # EXP-4A emits a single raw logit (verified: dummy forward spans
    # [-1406.8, +242.1], far outside [0,1]), so an activation IS required.
    model_outputs_probability: bool = False

    # --- preprocessing (fallbacks; checkpoint metadata wins) ---------------
    input_size: int = 224
    normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Eval-time geometry. "resize" -> Resize((S, S));
    # "resize_crop" -> Resize(int(S / CENTER_CROP_RATIO)) + CenterCrop(S).
    # EXP-4A trained on an already-resized dataset (Dataset02_resized,
    # image_size=224), so a direct square resize is the faithful choice.
    # TODO: verify against the training notebook's *validation* transform.
    resize_mode: Literal["resize", "resize_crop"] = "resize"
    center_crop_ratio: float = Field(default=0.875, gt=0.0, le=1.0)  # 224/256

    # --- uploads -----------------------------------------------------------
    max_upload_mb: float = Field(default=10.0, gt=0.0)
    max_batch_files: int = Field(default=10, ge=1)

    # --- model delivery ----------------------------------------------------
    # Weights are too large and too project-specific to commit. When MODEL_PATH
    # is absent at startup and this is set, the checkpoint is fetched once and
    # cached at MODEL_PATH. Point it at a direct-download URL (S3, R2, a GitHub
    # release asset) -- NOT a Google Drive / Dropbox preview page, which returns
    # HTML rather than the file.
    model_url: str = ""
    # Optional integrity check for that download. Get it with:
    #   sha256sum backend/models/<your>.pkl
    model_sha256: str = ""

    # Bearer token for a MODEL_URL that requires authentication -- a private
    # Hugging Face repo, a private object store, or a private GitHub release
    # asset. Sent as `Authorization: Bearer <token>`. Leave empty for a public
    # or presigned URL. Set this in the host's dashboard, never in git.
    model_auth_token: str = ""

    # --- server ------------------------------------------------------------
    # Render/Railway/Fly inject the port to bind. Read here so `python -m app`
    # and the Docker CMD agree; the Procfile passes it to uvicorn directly.
    port: int = 8000

    # Vite's default port is 5173, but it silently falls back to 5174, 5175...
    # when the port is already taken (a second `npm run dev`, or a dev server
    # left running from an earlier session). The page then loads fine while
    # every API call is blocked by CORS, which surfaces to the user as
    # "cannot reach the server" even though the backend is healthy. Listing the
    # fallback ports removes that failure mode in local development.
    allowed_origins: str = (
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174,http://localhost:5175,http://127.0.0.1:5175"
    )

    # Regex alternative for origins that cannot be enumerated -- chiefly
    # Vercel preview deployments, which get a new hostname per commit, e.g.
    #   ALLOWED_ORIGIN_REGEX=https://oral-cancer-app-.*-yourteam\.vercel\.app
    # Anchored at both ends by CORSMiddleware, so it must match the WHOLE origin.
    allowed_origin_regex: str = ""

    log_level: str = "INFO"

    @field_validator("model_path")
    @classmethod
    def _resolve_model_path(cls, value: Path) -> Path:
        """Resolve a relative MODEL_PATH against backend/, never the CWD.

        Without this, `MODEL_PATH=models/foo.pkl` only works when the server is
        started from inside backend/. Launched from anywhere else the file is
        silently not found, the model does not load, and /api/health still
        answers 200 with model_loaded=false -- which reads as "the backend is
        running but the app says it is not ready".
        """
        expanded = value.expanduser()
        if expanded.is_absolute():
            return expanded
        return (BACKEND_ROOT / expanded).resolve()

    @property
    def origins(self) -> list[str]:
        """CORS origins as a list. `*` disables the allow-list entirely."""
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def cors_kwargs(self) -> dict[str, object]:
        """CORS settings, including the regex only when one is configured.

        Passing allow_origin_regex="" would compile to a pattern that matches
        every origin, so an empty value must be omitted entirely.
        """
        kwargs: dict[str, object] = {"allow_origins": self.origins}
        if self.allowed_origin_regex.strip():
            kwargs["allow_origin_regex"] = self.allowed_origin_regex.strip()
        return kwargs

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @property
    def resolved_device(self) -> torch.device:
        """Honour DEVICE, falling back to CPU when CUDA was asked for but is absent."""
        if self.device == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cpu":
            return torch.device("cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
