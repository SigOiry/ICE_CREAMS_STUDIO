#!/usr/bin/env python3
"""Flet desktop UI for apply, train, and validation ICE CREAMS workflows."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import webbrowser
import warnings
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import flet as ft

try:
    import flet_map as ftm
except Exception:
    ftm = None

from apply_ICECREAMS import (
    classify_s2_scene,
    discover_scene_batch_info,
    move_scene_input_to_status_folder,
)
from train_icecreams import train_model
from validate_icecreams import (
    DEFAULT_TARGET_CLASS,
    VALIDATION_MODE_MULTICLASS,
    VALIDATION_MODE_PRESENCE_ABSENCE,
    validate_model,
)
from ice_creams_feature_modes import (
    DEFAULT_FEATURE_MODE,
    FEATURE_MODE_HIGH_SPATIAL_ACCURACY,
    FEATURE_MODE_HIGH_SPECTRAL_COMPLEXITY,
    feature_mode_label,
    normalize_feature_mode,
)
from ice_creams_model_families import (
    DEFAULT_SPECTRAL_CNN_USE_STANDARDIZED_REFLECTANCE,
    MODEL_FAMILY_SPECTRAL_1D_CNN,
    MODEL_FAMILY_TABULAR_DENSE,
    model_family_label,
    spectral_cnn_sequence_input_label,
)

# Keep terminal output clean on current Flet versions.
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    import pyi_splash  # type: ignore
except Exception:
    pyi_splash = None


PROJECT_ROOT = Path(__file__).resolve().parent
APP_VERSION = "1.0.23"
UPDATE_REPO_OWNER = "SigOiry"
UPDATE_REPO_NAME = "ICE_CREAMS_STUDIO"
UPDATE_REPO_BRANCH = "main"
UPDATE_MANIFEST_PATH = "templates/Output/latest.json"
UPDATE_RAW_BASE_URL = (
    f"https://raw.githubusercontent.com/"
    f"{UPDATE_REPO_OWNER}/{UPDATE_REPO_NAME}/{UPDATE_REPO_BRANCH}"
)
UPDATE_MANIFEST_URL = f"{UPDATE_RAW_BASE_URL}/{UPDATE_MANIFEST_PATH}"
APP_USER_AGENT = f"ICE_CREAMS_Studio/{APP_VERSION}"
LIQUID_ACCENT = "#4F8CFF"
LIQUID_SURFACE = "#F7FBFF"
LIQUID_SURFACE_ALT = "#EAF3FF"
LIQUID_TEXT = "#0F253D"
LIQUID_SUBTEXT = "#2E4B69"
LIQUID_MUTED = "#607E9E"
# Frosted-glass blur is visually nice but expensive on some Windows GPUs, so
# keep it opt-in and ship the desktop UI in the cheaper compositing mode.
ENABLE_DECORATIVE_BLUR = str(os.environ.get("ICECREAMS_ENABLE_DECORATIVE_BLUR", "")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
GLASS_PANEL_BLUR = (40, 40) if ENABLE_DECORATIVE_BLUR else None
MODAL_SCRIM_BLUR = (18, 18) if ENABLE_DECORATIVE_BLUR else None
SHOW_AMBIENT_BACKGROUND_BLOBS = ENABLE_DECORATIVE_BLUR
UI_REFRESH_MIN_INTERVAL_SECONDS = 0.05
MODAL_SCRIM_BG = ft.Colors.with_opacity(0.38, "#10263D")
MODAL_SCRIM_BG_STRONG = ft.Colors.with_opacity(0.46, "#10263D")
MODAL_PANEL_BG = ft.Colors.with_opacity(0.92, "#F8FBFF")
MODAL_PANEL_GRADIENT_TOP = ft.Colors.with_opacity(0.98, "#FFFFFF")
MODAL_PANEL_GRADIENT_BOTTOM = ft.Colors.with_opacity(0.94, "#E7F1FF")
MODAL_PANEL_BORDER = ft.Colors.with_opacity(0.82, "#FFFFFF")
MODAL_PANEL_DIVIDER = ft.Colors.with_opacity(0.18, "#16314C")
MODAL_BACKGROUND_OPACITY = 0.82


def _close_startup_splash() -> None:
    """Close PyInstaller splash screen when the first UI frame is ready."""
    if pyi_splash is None:
        return
    try:
        pyi_splash.close()
    except Exception:
        return


def _hide_windows_console_if_requested() -> None:
    """Hide Windows console for packaged runs or when explicitly requested."""
    if os.name != "nt":
        return

    hide_requested = str(os.environ.get("ICECREAMS_HIDE_CONSOLE", "")).strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if not hide_requested and not bool(getattr(sys, "frozen", False)):
        return

    try:
        import ctypes

        console_hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if console_hwnd:
            ctypes.windll.user32.ShowWindow(console_hwnd, 0)
    except Exception:
        return


def _resolve_initial_directory(current_value: str | None = None) -> str:
    """Pick a sensible directory for file chooser dialogs."""
    if current_value:
        current_path = Path(current_value)
        if current_path.exists():
            return str(current_path if current_path.is_dir() else current_path.parent)
    return str(PROJECT_ROOT)


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _compact_path(path_value: str | None, max_len: int = 88) -> str:
    """Shorten long filesystem paths for compact status displays."""
    if not path_value:
        return "-"
    if len(path_value) <= max_len:
        return path_value
    head = max_len // 2 - 2
    tail = max_len - head - 3
    return f"{path_value[:head]}...{path_value[-tail:]}"


def _normalise_web_url(url_value: str | None) -> str:
    """Return a browser-safe URL, defaulting to https when scheme is omitted."""
    if not url_value:
        return ""
    cleaned = str(url_value).strip().strip("<>()[]")
    if not cleaned:
        return ""
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", cleaned):
        cleaned = f"https://{cleaned}"
    return cleaned


def _resolve_folder_target(path_value: str | None) -> Path | None:
    """Resolve the folder that should be opened for a file or folder path."""
    if not path_value:
        return None

    candidate = Path(path_value)
    if candidate.exists():
        return candidate if candidate.is_dir() else candidate.parent

    return candidate.parent if candidate.suffix else candidate


def _parse_version_tuple(version_value: object) -> tuple[int, ...]:
    version_text = str(version_value or "").strip()
    numeric_parts = [int(part) for part in re.findall(r"\d+", version_text)]
    return tuple(numeric_parts) if numeric_parts else (0,)


def _is_newer_version(candidate_version: object, current_version: object) -> bool:
    return _parse_version_tuple(candidate_version) > _parse_version_tuple(current_version)


def _manifest_string(manifest: dict[str, object], key: str) -> str:
    return str(manifest.get(key, "") or "").strip()


def _github_lfs_binary_url(raw_url: str, *, use_media_host: bool = True) -> str:
    """Convert a raw.githubusercontent.com URL into a URL that serves the real binary."""
    cleaned_url = _normalise_web_url(raw_url)
    match = re.match(
        r"^https://raw\.githubusercontent\.com/([^/]+)/([^/]+)/([^/]+)/(.+)$",
        cleaned_url,
        re.IGNORECASE,
    )
    if not match:
        return ""

    owner, repo, git_ref, asset_path = match.groups()
    if use_media_host:
        return f"https://media.githubusercontent.com/media/{owner}/{repo}/{git_ref}/{asset_path}"
    return f"https://github.com/{owner}/{repo}/raw/{git_ref}/{asset_path}"


def _candidate_installer_urls(installer_url: str) -> list[str]:
    """Return installer download URLs, including GitHub LFS-safe fallbacks when needed."""
    primary_url = _normalise_web_url(installer_url)
    if not primary_url:
        return []

    candidates = [primary_url]
    for fallback_url in (
        _github_lfs_binary_url(primary_url, use_media_host=True),
        _github_lfs_binary_url(primary_url, use_media_host=False),
    ):
        if fallback_url and fallback_url not in candidates:
            candidates.append(fallback_url)
    return candidates


def _looks_like_git_lfs_pointer(file_prefix: bytes) -> bool:
    """Detect Git LFS pointer files returned instead of the real binary payload."""
    if not file_prefix:
        return False
    return file_prefix.startswith(b"version https://git-lfs.github.com/spec/v1")


def _resolve_manifest_installer_url(manifest: dict[str, object]) -> str:
    installer_url = _normalise_web_url(_manifest_string(manifest, "installer_url"))
    if installer_url:
        github_raw_binary_url = _github_lfs_binary_url(installer_url, use_media_host=False)
        return github_raw_binary_url or installer_url

    installer_path = _manifest_string(manifest, "installer_path").lstrip("/")
    if not installer_path:
        return ""
    return f"https://github.com/{UPDATE_REPO_OWNER}/{UPDATE_REPO_NAME}/raw/{UPDATE_REPO_BRANCH}/{installer_path}"


def _fetch_update_manifest() -> dict[str, object]:
    request = Request(
        UPDATE_MANIFEST_URL,
        headers={
            "User-Agent": APP_USER_AGENT,
            "Cache-Control": "no-cache",
        },
    )
    with urlopen(request, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Update manifest is not a JSON object.")
    return payload


def _download_update_installer(
    installer_url: str,
    destination_path: Path,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    download_errors: list[str] = []

    for candidate_url in _candidate_installer_urls(installer_url):
        try:
            request = Request(
                candidate_url,
                headers={
                    "User-Agent": APP_USER_AGENT,
                    "Cache-Control": "no-cache",
                },
            )
            hasher = hashlib.sha256()
            bytes_downloaded = 0
            total_bytes = 0
            with urlopen(request, timeout=30) as response:
                total_bytes = int(response.headers.get("Content-Length", "0") or 0)
                with destination_path.open("wb") as installer_stream:
                    first_chunk = response.read(1024 * 1024)
                    if _looks_like_git_lfs_pointer(first_chunk):
                        raise ValueError(
                            "The installer URL returned a Git LFS pointer file instead of the actual installer."
                        )
                    if first_chunk:
                        installer_stream.write(first_chunk)
                        hasher.update(first_chunk)
                        bytes_downloaded += len(first_chunk)
                        if progress_callback is not None:
                            progress_callback(bytes_downloaded, total_bytes)

                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        installer_stream.write(chunk)
                        hasher.update(chunk)
                        bytes_downloaded += len(chunk)
                        if progress_callback is not None:
                            progress_callback(bytes_downloaded, total_bytes)

            return {
                "path": str(destination_path),
                "size_bytes": bytes_downloaded,
                "sha256": hasher.hexdigest().upper(),
            }
        except Exception as exc:  # noqa: BLE001 - try alternate GitHub asset URLs before failing.
            download_errors.append(f"{candidate_url} -> {exc}")
            try:
                destination_path.unlink(missing_ok=True)
            except Exception:
                pass

    if download_errors:
        raise ValueError(download_errors[-1])
    raise ValueError("No installer download URL was available.")


def _launch_windows_update_installer(installer_path: Path) -> None:
    if os.name != "nt":
        raise OSError("Automatic installer launch is only supported on Windows.")

    update_dir = installer_path.parent
    update_dir.mkdir(parents=True, exist_ok=True)
    script_path = update_dir / f"run_update_{int(time.time())}.cmd"
    current_executable = ""
    if getattr(sys, "frozen", False):
        try:
            current_executable = str(Path(sys.executable).resolve())
        except Exception:
            current_executable = sys.executable

    script_lines = [
        "@echo off",
        "setlocal",
        "timeout /t 2 /nobreak >nul",
        f'start "" /wait "{installer_path}" /SP- /VERYSILENT /SUPPRESSMSGBOXES /NOCANCEL /CLOSEAPPLICATIONS /FORCECLOSEAPPLICATIONS',
        "set EXIT_CODE=%ERRORLEVEL%",
        'if not "%EXIT_CODE%"=="0" exit /b %EXIT_CODE%',
    ]
    if current_executable:
        script_lines.append(f'if exist "{current_executable}" start "" "{current_executable}"')
    script_lines.extend(
        [
            'del "%~f0"',
            "exit /b 0",
        ]
    )
    script_path.write_text("\r\n".join(script_lines) + "\r\n", encoding="ascii")

    creationflags = 0
    for flag_name in ("CREATE_NEW_PROCESS_GROUP", "DETACHED_PROCESS", "CREATE_NO_WINDOW"):
        creationflags |= int(getattr(subprocess, flag_name, 0))

    subprocess.Popen(
        ["cmd.exe", "/c", str(script_path)],
        creationflags=creationflags,
        close_fds=True,
        cwd=str(update_dir),
    )


def _glass_panel(
    content: ft.Control,
    expand: bool = False,
    padding: int = 24,
    *,
    variant: str = "card",
    accent: str | None = None,
) -> ft.Container:
    """Create a liquid-glass card used throughout the UI."""
    panel_accent = accent or LIQUID_ACCENT
    is_modal = variant == "modal"
    is_sidebar = variant == "sidebar"
    if is_modal:
        content = ft.Column(
            spacing=18,
            tight=True,
            controls=[
                ft.Row(
                    spacing=12,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=[
                        ft.Container(
                            width=56,
                            height=5,
                            border_radius=999,
                            bgcolor=panel_accent,
                        ),
                        ft.Container(
                            expand=True,
                            height=1,
                            bgcolor=MODAL_PANEL_DIVIDER,
                        ),
                    ],
                ),
                content,
            ],
        )
    uses_opaque_surface = is_modal or is_sidebar

    return ft.Container(
        content=content,
        expand=expand,
        padding=padding,
        border_radius=30 if uses_opaque_surface else 28,
        blur=None if uses_opaque_surface else GLASS_PANEL_BLUR,
        bgcolor=MODAL_PANEL_BG if uses_opaque_surface else ft.Colors.with_opacity(0.52, LIQUID_SURFACE),
        gradient=ft.LinearGradient(
            begin=ft.Alignment(-1.0, -1.0),
            end=ft.Alignment(1.0, 1.0),
            colors=(
                [
                    MODAL_PANEL_GRADIENT_TOP,
                    ft.Colors.with_opacity(0.96, "#F2F8FF"),
                    MODAL_PANEL_GRADIENT_BOTTOM,
                ]
                if uses_opaque_surface
                else [
                    ft.Colors.with_opacity(0.62, "#FFFFFF"),
                    ft.Colors.with_opacity(0.40, LIQUID_SURFACE_ALT),
                    ft.Colors.with_opacity(0.18, "#DCEAFF"),
                ]
            ),
        ),
        border=ft.border.all(
            1,
            MODAL_PANEL_BORDER if uses_opaque_surface else ft.Colors.with_opacity(0.56, "#FFFFFF"),
        ),
        shadow=(
            [
                ft.BoxShadow(
                    blur_radius=30,
                    spread_radius=0,
                    color=ft.Colors.with_opacity(0.18, "#10263D"),
                    offset=(0, 14),
                ),
                ft.BoxShadow(
                    blur_radius=8,
                    spread_radius=0,
                    color=ft.Colors.with_opacity(0.10, panel_accent),
                    offset=(0, 2),
                ),
            ]
            if uses_opaque_surface
            else [
                ft.BoxShadow(
                    blur_radius=18,
                    spread_radius=0,
                    color=ft.Colors.with_opacity(0.10, "#284D72"),
                    offset=(0, 6),
                ),
                ft.BoxShadow(
                    blur_radius=22,
                    spread_radius=-6,
                    color=ft.Colors.with_opacity(0.62, "#FFFFFF"),
                    offset=(0, 1),
                ),
            ]
        ),
    )


def _log_entry(message: str, accent: str) -> ft.Container:
    """Create a single activity log row."""
    return ft.Container(
        padding=ft.padding.symmetric(horizontal=14, vertical=10),
        border_radius=18,
        bgcolor=ft.Colors.with_opacity(0.64, LIQUID_SURFACE_ALT),
        border=ft.border.all(1, ft.Colors.with_opacity(0.26, accent)),
        content=ft.Row(
            spacing=12,
            vertical_alignment=ft.CrossAxisAlignment.START,
            controls=[
                ft.Container(
                    width=10,
                    height=10,
                    border_radius=999,
                    bgcolor=accent,
                    margin=ft.margin.only(top=5),
                ),
                ft.Text(
                    f"{_timestamp()}  {message}",
                    size=13,
                    color=LIQUID_TEXT,
                ),
            ],
        ),
    )


def _labeled_picker(
    title: str,
    field: ft.TextField,
    button: ft.Control,
    helper: str,
) -> ft.Container:
    """Build a consistent file or folder picker row."""
    return ft.Container(
        padding=ft.padding.symmetric(horizontal=18, vertical=16),
        border_radius=22,
        bgcolor=ft.Colors.with_opacity(0.62, LIQUID_SURFACE_ALT),
        border=ft.border.all(1, ft.Colors.with_opacity(0.30, ft.Colors.WHITE)),
        content=ft.Column(
            spacing=10,
            controls=[
                ft.Text(
                    title,
                    size=14,
                    weight=ft.FontWeight.W_600,
                    color=LIQUID_TEXT,
                ),
                ft.ResponsiveRow(
                    columns=12,
                    run_spacing=12,
                    spacing=12,
                    controls=[
                        ft.Container(
                            col={"xs": 12, "md": 8},
                            content=field,
                        ),
                        ft.Container(
                            col={"xs": 12, "sm": 6, "lg": 4},
                            content=button,
                        ),
                    ],
                ),
                ft.Text(
                    helper,
                    size=11,
                    color=LIQUID_MUTED,
                ),
            ],
        ),
    )


def _frosted_button_style(bgcolor: str, color: str) -> ft.ButtonStyle:
    """Create a glossy liquid-glass button style while preserving base colors."""
    return ft.ButtonStyle(
        bgcolor={
            ft.ControlState.DEFAULT: ft.Colors.with_opacity(0.84, bgcolor),
            ft.ControlState.HOVERED: ft.Colors.with_opacity(0.96, bgcolor),
            ft.ControlState.DISABLED: ft.Colors.with_opacity(0.50, LIQUID_SURFACE_ALT),
        },
        color={
            ft.ControlState.DEFAULT: LIQUID_TEXT,
            ft.ControlState.DISABLED: ft.Colors.with_opacity(0.48, LIQUID_TEXT),
        },
        elevation={ft.ControlState.DEFAULT: 0, ft.ControlState.HOVERED: 1},
        side={
            ft.ControlState.DEFAULT: ft.BorderSide(1, ft.Colors.with_opacity(0.42, ft.Colors.WHITE)),
            ft.ControlState.HOVERED: ft.BorderSide(1, ft.Colors.with_opacity(0.66, ft.Colors.WHITE)),
        },
        shape=ft.RoundedRectangleBorder(radius=18),
        padding=ft.padding.symmetric(horizontal=16, vertical=12),
        overlay_color=ft.Colors.TRANSPARENT,
        animation_duration=0,
        enable_feedback=False,
    )


def _workflow_intro_panel(title: str, subtitle: str, steps: list[str]) -> ft.Container:
    """Create a compact workflow briefing panel with explicit step sequence."""
    step_chips: list[ft.Control] = []
    for index, step_text in enumerate(steps, start=1):
        step_chips.append(
            ft.Container(
                padding=ft.padding.symmetric(horizontal=12, vertical=8),
                border_radius=999,
                bgcolor=ft.Colors.with_opacity(0.72, LIQUID_SURFACE_ALT),
                border=ft.border.all(1, ft.Colors.with_opacity(0.30, ft.Colors.WHITE)),
                content=ft.Text(
                    f"{index}. {step_text}",
                    size=11,
                    color=LIQUID_SUBTEXT,
                ),
            )
        )

    return _glass_panel(
        padding=18,
        content=ft.Column(
            spacing=10,
            controls=[
                ft.Text(
                    title,
                    size=20,
                    weight=ft.FontWeight.W_700,
                    color=LIQUID_TEXT,
                ),
                ft.Text(
                    subtitle,
                    size=12,
                    color=LIQUID_SUBTEXT,
                ),
                ft.Row(
                    wrap=True,
                    spacing=8,
                    run_spacing=8,
                    controls=step_chips,
                ),
            ],
        ),
    )


def main(page: ft.Page) -> None:
    """Configure and render the Flet application."""
    _hide_windows_console_if_requested()

    page.title = "ICE CREAMS Studio"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 0
    page.spacing = 0
    page.bgcolor = "#DCEBFA"
    page.scroll = ft.ScrollMode.HIDDEN
    page.theme = ft.Theme(
        color_scheme_seed=LIQUID_ACCENT,
        font_family="SF Pro Display",
        use_material3=True,
    )

    # Start maximized (fullscreen-like) while preserving window controls so users
    # can re-enter fullscreen/maximized mode after switching to windowed mode.
    try:
        page.window.resizable = True
        page.window.maximizable = True
        page.window.maximized = True
        page.window.full_screen = False
    except AttributeError:
        if hasattr(page, "window_resizable"):
            page.window_resizable = True
        if hasattr(page, "window_maximizable"):
            page.window_maximizable = True
        if hasattr(page, "window_maximized"):
            page.window_maximized = True
        if hasattr(page, "window_full_screen"):
            page.window_full_screen = False

    default_apply_output_dir = PROJECT_ROOT / "outputs"
    default_training_source = PROJECT_ROOT / "Data" / "Input" / "Training"
    default_validation_source = PROJECT_ROOT / "Data" / "Input" / "Validation"
    default_validation_output_dir = PROJECT_ROOT / "outputs"
    default_models_dir = PROJECT_ROOT / "models"
    about_assets_dir = PROJECT_ROOT / "about"
    about_icons_dir = PROJECT_ROOT / "icons"
    existing_models = (
        sorted(default_models_dir.rglob("*.pkl"))
        if default_models_dir.exists()
        else []
    )

    def _normalise_model_name(name: str) -> str:
        return "".join(ch for ch in name.lower() if ch.isalnum())

    preferred_model_key = _normalise_model_name("ICE_CREAMS_V1.3.0")
    default_apply_model = next(
        (
            model_path
            for model_path in existing_models
            if _normalise_model_name(model_path.stem) == preferred_model_key
        ),
        existing_models[0] if existing_models else None,
    )
    about_info_path = about_assets_dir / "Info.txt"
    about_info_text = ""
    try:
        about_info_text = about_info_path.read_text(encoding="utf-8")
    except OSError:
        about_info_text = ""

    about_lines = [line.strip() for line in about_info_text.splitlines() if line.strip()]
    about_project_statement = (
        about_lines[0]
        if len(about_lines) >= 1
        else "ICE CREAMS is developed within the BiCOME project supported by ESA."
    )
    about_ui_statement = (
        about_lines[1]
        if len(about_lines) >= 2
        else "This UI was designed by Simon Oiry for the ICE CREAMS algorithm developed by Bede Davies."
    )

    about_links: dict[str, str] = {
        "repo": "https://github.com/BedeFfinian/ICE_CREAMS",
        "simon_website": "https://oirysimon.com",
        "simon_github": "https://github.com/SigOiry",
        "simon_researchgate": "https://www.researchgate.net/profile/Simon-Oiry",
        "bede_website": "https://bedeffinianrowedavies.com",
        "bede_github": "https://github.com/BedeFfinian",
        "bede_researchgate": "https://www.researchgate.net/profile/Bede-Davies-2",
    }
    current_about_profile = ""
    for raw_line in about_info_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith("simon"):
            current_about_profile = "simon"
            continue
        if lowered.startswith("bede"):
            current_about_profile = "bede"
            continue
        if ":" not in line:
            continue

        key_part, value_part = line.split(":", 1)
        key_name = key_part.strip(" -").lower()
        url_value = _normalise_web_url(value_part.strip())
        if not url_value:
            continue

        if "link toward the algorithm" in lowered:
            about_links["repo"] = url_value
            continue
        if current_about_profile not in {"simon", "bede"}:
            continue
        if "website" in key_name:
            about_links[f"{current_about_profile}_website"] = url_value
        elif "github" in key_name:
            about_links[f"{current_about_profile}_github"] = url_value
        elif "researchgate" in key_name:
            about_links[f"{current_about_profile}_researchgate"] = url_value

    simon_photo_path = about_assets_dir / "Simon.png"
    bede_photo_path = about_assets_dir / "Bede.png"
    project_logo_path = about_icons_dir / "Icecream logo.png"
    bicome_logo_path = about_assets_dir / "BiCOME Logo.png"
    esa_logo_path = about_assets_dir / "ESA_logo.png"
    nantes_logo_path = about_assets_dir / "Nantes-Universite.png"
    if not nantes_logo_path.exists():
        nantes_logo_path = about_assets_dir / "Nantes-Université.png"

    state = {
        "busy": False,
        "operation_mode": None,
        "apply_run_token": 0,
        "train_run_token": 0,
        "validation_run_token": 0,
    }
    update_state = {
        "check_running": False,
        "install_running": False,
        "manifest": None,
        "status_before_check": "Ready for a new run.",
    }
    apply_preflight_state = {
        "token": 0,
        "running": False,
        "signature": None,
        "scene_batch_info": None,
        "pending_scene_inputs": [],
        "skipped_existing_scene_inputs": [],
        "skipped_existing_outputs": [],
    }
    history_log_path = default_apply_output_dir / "run_history.jsonl"
    history_max_records = 2000

    def _load_history_records_from_disk(log_path: Path) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        if not log_path.exists():
            return records

        try:
            with log_path.open("r", encoding="utf-8") as log_stream:
                for raw_line in log_stream:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        parsed = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(parsed, dict):
                        records.append(parsed)
        except OSError:
            return []

        if len(records) > history_max_records:
            records = records[-history_max_records:]
        return records

    def _rewrite_history_records_to_disk() -> None:
        try:
            history_log_path.parent.mkdir(parents=True, exist_ok=True)
            temp_log_path = history_log_path.with_name(f"{history_log_path.name}.tmp")
            with temp_log_path.open("w", encoding="utf-8") as log_stream:
                for history_record in history_entries[-history_max_records:]:
                    log_stream.write(json.dumps(history_record, ensure_ascii=True))
                    log_stream.write("\n")
            temp_log_path.replace(history_log_path)
        except OSError as exc:
            history_status.value = f"Could not rewrite history file: {exc}"

    history_entries: list[dict[str, object]] = _load_history_records_from_disk(history_log_path)
    ui_refresh_state = {"scheduled": False, "last": 0.0}
    selected_training_csv_files: list[str] = []
    selected_training_csv_folders: list[str] = []
    selected_training_csvs: list[str] = []
    overlay_targets = {"input": "", "output": ""}

    app_status_title = ft.Text(
        "Ready",
        size=18,
        weight=ft.FontWeight.W_700,
        color=LIQUID_TEXT,
    )
    app_status = ft.Text(
        "Ready for a new run.",
        size=13,
        color=LIQUID_SUBTEXT,
    )
    app_status_dot = ft.Container(
        width=14,
        height=14,
        border_radius=999,
        bgcolor="#41B883",
    )
    app_status_ring = ft.ProgressRing(
        width=16,
        height=16,
        stroke_width=2.2,
        color="#FFD166",
        visible=False,
    )
    app_status_card = ft.Container()

    overlay_title = ft.Text(
        "Working",
        size=22,
        weight=ft.FontWeight.W_700,
        color=LIQUID_TEXT,
    )
    overlay_detail = ft.Text(
        "Preparing the workflow.",
        size=13,
        color=LIQUID_SUBTEXT,
        max_lines=4,
    )
    overlay_counter = ft.Text(
        "",
        size=12,
        color=LIQUID_MUTED,
    )
    overlay_percent = ft.Text(
        "0%",
        size=13,
        weight=ft.FontWeight.W_600,
        color=LIQUID_TEXT,
    )
    overlay_step_label = ft.Text(
        "Current step",
        size=11,
        color=LIQUID_MUTED,
    )
    overlay_progress = ft.ProgressBar(
        value=0,
        bar_height=10,
        border_radius=999,
        color=LIQUID_ACCENT,
        bgcolor=ft.Colors.with_opacity(0.34, "#B5CAE2"),
    )
    overlay_spinner = ft.ProgressRing(
        width=24,
        height=24,
        stroke_width=2.6,
        color=LIQUID_ACCENT,
    )
    overlay_job_label = ft.Text(
        "Current task",
        size=11,
        color=LIQUID_MUTED,
    )
    overlay_job_value = ft.Text(
        "-",
        size=13,
        color=LIQUID_TEXT,
        max_lines=2,
    )
    overlay_input_label = ft.Text(
        "Input folder",
        size=11,
        color=LIQUID_MUTED,
    )
    overlay_input_path = ft.Text(
        "-",
        size=13,
        color=LIQUID_TEXT,
        max_lines=3,
    )
    overlay_input_hint = ft.Text(
        "Open in Windows Explorer",
        size=11,
        color=LIQUID_ACCENT,
    )
    overlay_input_value = ft.Container(
        border_radius=14,
        padding=ft.padding.symmetric(horizontal=12, vertical=10),
        bgcolor=ft.Colors.with_opacity(0.54, LIQUID_SURFACE_ALT),
        content=ft.Column(
            spacing=4,
            tight=True,
            controls=[
                overlay_input_path,
                overlay_input_hint,
            ],
        ),
    )
    overlay_output_label = ft.Text(
        "Output folder",
        size=11,
        color=LIQUID_MUTED,
    )
    overlay_output_path = ft.Text(
        "-",
        size=13,
        color=LIQUID_TEXT,
        max_lines=3,
    )
    overlay_output_hint = ft.Text(
        "Open in Windows Explorer",
        size=11,
        color=LIQUID_ACCENT,
    )
    overlay_output_value = ft.Container(
        border_radius=14,
        padding=ft.padding.symmetric(horizontal=12, vertical=10),
        bgcolor=ft.Colors.with_opacity(0.54, LIQUID_SURFACE_ALT),
        content=ft.Column(
            spacing=4,
            tight=True,
            controls=[
                overlay_output_path,
                overlay_output_hint,
            ],
        ),
    )
    overlay_blocker = ft.Container()

    apply_safe_field = ft.TextField(
        value="",
        hint_text="Select a single .zip/.tif file, a single .SAFE folder, or a batch folder",
        read_only=True,
        border_radius=18,
        filled=True,
        fill_color=ft.Colors.with_opacity(0.72, LIQUID_SURFACE_ALT),
        border_color=ft.Colors.with_opacity(0.32, "#A6BFD9"),
        focused_border_color=LIQUID_ACCENT,
        color=LIQUID_TEXT,
        text_size=14,
        height=56,
        content_padding=ft.padding.symmetric(horizontal=16, vertical=14),
    )
    apply_mask_field = ft.TextField(
        value="",
        hint_text="Select a shapefile mask (.shp)",
        read_only=True,
        border_radius=18,
        filled=True,
        fill_color=ft.Colors.with_opacity(0.72, LIQUID_SURFACE_ALT),
        border_color=ft.Colors.with_opacity(0.32, "#A6BFD9"),
        focused_border_color=LIQUID_ACCENT,
        color=LIQUID_TEXT,
        text_size=14,
        height=56,
        content_padding=ft.padding.symmetric(horizontal=16, vertical=14),
    )
    apply_output_path_field = ft.TextField(
        value=str(default_apply_output_dir),
        hint_text="Select an output folder",
        read_only=True,
        border_radius=18,
        filled=True,
        fill_color=ft.Colors.with_opacity(0.72, LIQUID_SURFACE_ALT),
        border_color=ft.Colors.with_opacity(0.32, "#A6BFD9"),
        focused_border_color=LIQUID_ACCENT,
        color=LIQUID_TEXT,
        text_size=14,
        height=56,
        content_padding=ft.padding.symmetric(horizontal=16, vertical=14),
    )
    apply_model_dropdown = ft.Dropdown(
        value=str(default_apply_model) if default_apply_model else None,
        options=[
            ft.dropdown.Option(
                key=str(model_path),
                text=model_path.name,
            )
            for model_path in existing_models
        ],
        hint_text="Select a FastAI model from the models folder",
        enable_search=True,
        border_radius=18,
        filled=True,
        fill_color=ft.Colors.with_opacity(0.72, LIQUID_SURFACE_ALT),
        border_color=ft.Colors.with_opacity(0.32, "#A6BFD9"),
        focused_border_color=LIQUID_ACCENT,
        color=LIQUID_TEXT,
        text_size=14,
        height=56,
        content_padding=ft.padding.symmetric(horizontal=16, vertical=14),
    )

    apply_output_preview = ft.Text(
        "Choose an output folder. GeoTIFF names include the selected model.",
        size=12,
        color=LIQUID_SUBTEXT,
    )
    apply_progress = ft.ProgressBar(
        value=0,
        bar_height=8,
        border_radius=999,
        color=LIQUID_ACCENT,
        bgcolor=ft.Colors.with_opacity(0.34, "#B5CAE2"),
    )
    apply_spinner = ft.ProgressRing(
        width=18,
        height=18,
        stroke_width=2.2,
        color=LIQUID_ACCENT,
        visible=False,
    )
    apply_status = ft.Text(
        "Waiting for the required inputs.",
        size=13,
        color=LIQUID_SUBTEXT,
    )
    apply_log = ft.ListView(
        expand=True,
        spacing=10,
        auto_scroll=True,
        scroll=ft.ScrollMode.AUTO,
    )

    batch_popup_title = ft.Text(
        "Batch Discovery",
        size=22,
        weight=ft.FontWeight.W_700,
        color=LIQUID_TEXT,
    )
    batch_popup_summary = ft.Text(
        "Batch summary",
        size=13,
        color=LIQUID_SUBTEXT,
    )
    batch_popup_formats = ft.Text(
        "Formats: -",
        size=12,
        color=LIQUID_SUBTEXT,
    )
    batch_popup_dates = ft.Text(
        "Acquisition dates: -",
        size=12,
        color=LIQUID_SUBTEXT,
    )
    batch_popup_warning = ft.Text(
        "",
        size=12,
        color="#FFD8A8",
        visible=False,
    )
    batch_popup_table = ft.DataTable(
        columns=[
            ft.DataColumn(ft.Text("Scene ID", color=LIQUID_TEXT, size=12, weight=ft.FontWeight.W_600)),
            ft.DataColumn(ft.Text("Format", color=LIQUID_TEXT, size=12, weight=ft.FontWeight.W_600)),
            ft.DataColumn(ft.Text("Acquired", color=LIQUID_TEXT, size=12, weight=ft.FontWeight.W_600)),
            ft.DataColumn(ft.Text("Status", color=LIQUID_TEXT, size=12, weight=ft.FontWeight.W_600)),
            ft.DataColumn(ft.Text("Path", color=LIQUID_TEXT, size=12, weight=ft.FontWeight.W_600)),
        ],
        rows=[],
        horizontal_lines=ft.BorderSide(1, ft.Colors.with_opacity(0.10, ft.Colors.WHITE)),
        vertical_lines=ft.BorderSide(1, ft.Colors.with_opacity(0.06, ft.Colors.WHITE)),
        heading_row_height=40,
        data_row_min_height=40,
        data_row_max_height=44,
        column_spacing=16,
    )
    batch_popup_close_button = ft.IconButton(
        icon=ft.Icons.CLOSE,
        icon_color=LIQUID_TEXT,
        tooltip="Close batch discovery",
        style=ft.ButtonStyle(
            bgcolor=ft.Colors.with_opacity(0.52, LIQUID_SURFACE_ALT),
            side=ft.BorderSide(1, ft.Colors.with_opacity(0.30, "#A6BFD9")),
            shape=ft.RoundedRectangleBorder(radius=12),
            overlay_color=ft.Colors.TRANSPARENT,
            animation_duration=0,
            enable_feedback=False,
        ),
    )
    batch_popup_blocker = ft.Container()
    validation_popup_title = ft.Text(
        "Validation Results",
        size=22,
        weight=ft.FontWeight.W_700,
        color=LIQUID_TEXT,
    )
    validation_popup_accuracy = ft.Text(
        "Overall accuracy: -",
        size=14,
        weight=ft.FontWeight.W_600,
        color=LIQUID_TEXT,
    )
    validation_popup_note = ft.Text(
        "",
        size=12,
        color=LIQUID_SUBTEXT,
        visible=False,
    )
    validation_popup_table = ft.DataTable(
        columns=[
            ft.DataColumn(
                ft.Text("Confusion Matrix", color=LIQUID_TEXT, size=12, weight=ft.FontWeight.W_700)
            )
        ],
        rows=[
            ft.DataRow(
                cells=[
                    ft.DataCell(
                        ft.Text(
                            "Run a validation to display confusion matrix and accuracy.",
                            size=12,
                            color=LIQUID_SUBTEXT,
                        )
                    )
                ]
            )
        ],
        horizontal_lines=ft.BorderSide(1, ft.Colors.with_opacity(0.10, ft.Colors.WHITE)),
        vertical_lines=ft.BorderSide(1, ft.Colors.with_opacity(0.06, ft.Colors.WHITE)),
        heading_row_height=40,
        data_row_min_height=40,
        data_row_max_height=44,
        column_spacing=14,
    )
    validation_popup_close_button = ft.IconButton(
        icon=ft.Icons.CLOSE,
        icon_color=LIQUID_TEXT,
        tooltip="Close validation results",
        style=ft.ButtonStyle(
            bgcolor=ft.Colors.with_opacity(0.52, LIQUID_SURFACE_ALT),
            side=ft.BorderSide(1, ft.Colors.with_opacity(0.30, "#A6BFD9")),
            shape=ft.RoundedRectangleBorder(radius=12),
            overlay_color=ft.Colors.TRANSPARENT,
            animation_duration=0,
            enable_feedback=False,
        ),
    )
    validation_popup_blocker = ft.Container()
    history_map_popup_title = ft.Text(
        "Mask Extent Map",
        size=22,
        weight=ft.FontWeight.W_700,
        color=LIQUID_TEXT,
    )
    history_map_popup_summary = ft.Text(
        "Stored shapefile extent polygon for this run.",
        size=13,
        color=LIQUID_SUBTEXT,
    )
    history_map_popup_bounds = ft.Text(
        "-",
        size=12,
        color=LIQUID_SUBTEXT,
        selectable=True,
    )
    history_map_popup_close_button = ft.IconButton(
        icon=ft.Icons.CLOSE,
        icon_color=LIQUID_TEXT,
        tooltip="Close mask extent map",
        style=ft.ButtonStyle(
            bgcolor=ft.Colors.with_opacity(0.52, LIQUID_SURFACE_ALT),
            side=ft.BorderSide(1, ft.Colors.with_opacity(0.30, "#A6BFD9")),
            shape=ft.RoundedRectangleBorder(radius=12),
            overlay_color=ft.Colors.TRANSPARENT,
            animation_duration=0,
            enable_feedback=False,
        ),
    )
    history_map_popup_external_button = ft.ElevatedButton(
        "Open in browser",
        icon=ft.Icons.OPEN_IN_BROWSER,
        style=_frosted_button_style("#E6F4FF", "#14324C"),
        visible=False,
    )
    history_map_popup_content_host = ft.Container(
        height=520,
        border_radius=18,
        bgcolor=ft.Colors.with_opacity(0.72, LIQUID_SURFACE_ALT),
        alignment=ft.Alignment(0, 0),
        content=ft.Text(
            "Select a run with a stored mask extent to display the map.",
            size=12,
            color=LIQUID_SUBTEXT,
            text_align=ft.TextAlign.CENTER,
        ),
    )
    history_map_popup_blocker = ft.Container()
    about_popup_close_button = ft.IconButton(
        icon=ft.Icons.CLOSE,
        icon_color=LIQUID_TEXT,
        tooltip="Close about panel",
        style=ft.ButtonStyle(
            bgcolor=ft.Colors.with_opacity(0.52, LIQUID_SURFACE_ALT),
            side=ft.BorderSide(1, ft.Colors.with_opacity(0.30, "#A6BFD9")),
            shape=ft.RoundedRectangleBorder(radius=12),
            overlay_color=ft.Colors.TRANSPARENT,
            animation_duration=0,
            enable_feedback=False,
        ),
    )
    menu_about_button = ft.Container(
        right=22,
        top=22,
        visible=False,
        border_radius=999,
        padding=ft.padding.symmetric(horizontal=14, vertical=10),
        bgcolor=ft.Colors.with_opacity(0.86, "#F5A623"),
        border=ft.border.all(1, ft.Colors.with_opacity(0.68, ft.Colors.WHITE)),
        shadow=[
            ft.BoxShadow(
                blur_radius=14,
                spread_radius=0,
                color=ft.Colors.with_opacity(0.24, "#8A5A06"),
                offset=(0, 4),
            )
        ],
        content=ft.Text("About", size=14, weight=ft.FontWeight.W_800, color="#10253B"),
    )
    menu_update_button = ft.Container(
        right=22,
        top=74,
        visible=False,
        border_radius=999,
        padding=ft.padding.symmetric(horizontal=14, vertical=10),
        bgcolor=ft.Colors.with_opacity(0.88, "#D4F7E3"),
        border=ft.border.all(1, ft.Colors.with_opacity(0.68, ft.Colors.WHITE)),
        shadow=[
            ft.BoxShadow(
                blur_radius=14,
                spread_radius=0,
                color=ft.Colors.with_opacity(0.20, "#1F6E52"),
                offset=(0, 4),
            )
        ],
        content=ft.Row(
            tight=True,
            spacing=6,
            controls=[
                ft.Icon(ft.Icons.SYSTEM_UPDATE_ALT, size=16, color="#103B2F"),
                ft.Text("Check updates", size=13, weight=ft.FontWeight.W_800, color="#103B2F"),
            ],
        ),
    )
    about_popup_blocker = ft.Container()
    update_popup_title = ft.Text(
        "Update Available",
        size=22,
        weight=ft.FontWeight.W_700,
        color=LIQUID_TEXT,
    )
    update_popup_summary = ft.Text(
        "A newer version of ICE CREAMS Studio is ready to install.",
        size=13,
        color=LIQUID_SUBTEXT,
    )
    update_popup_details = ft.Text(
        "-",
        size=12,
        color=LIQUID_SUBTEXT,
        selectable=True,
    )
    update_popup_status = ft.Text(
        "",
        size=12,
        color="#D6455D",
        visible=False,
    )
    update_popup_close_button = ft.IconButton(
        icon=ft.Icons.CLOSE,
        icon_color=LIQUID_TEXT,
        tooltip="Dismiss update prompt",
        style=ft.ButtonStyle(
            bgcolor=ft.Colors.with_opacity(0.52, LIQUID_SURFACE_ALT),
            side=ft.BorderSide(1, ft.Colors.with_opacity(0.30, "#A6BFD9")),
            shape=ft.RoundedRectangleBorder(radius=12),
            overlay_color=ft.Colors.TRANSPARENT,
            animation_duration=0,
            enable_feedback=False,
        ),
    )
    update_popup_later_button = ft.ElevatedButton(
        "Later",
        icon=ft.Icons.SCHEDULE,
        style=_frosted_button_style("#E6F4FF", "#14324C"),
    )
    update_popup_install_button = ft.ElevatedButton(
        "Update now",
        icon=ft.Icons.SYSTEM_UPDATE_ALT,
        style=_frosted_button_style("#D4F7E3", "#103B2F"),
    )
    update_popup_release_notes_button = ft.ElevatedButton(
        "Release notes",
        icon=ft.Icons.OPEN_IN_BROWSER,
        style=_frosted_button_style("#FFF1D6", "#5F4500"),
        visible=False,
    )
    update_popup_blocker = ft.Container()

    def _open_external_link(url_value: str) -> None:
        target_url = _normalise_web_url(url_value)
        if not target_url:
            return
        try:
            if webbrowser.open_new_tab(target_url):
                return
        except Exception:
            pass
        try:
            page.launch_url(target_url)
        except Exception:
            return

    def _about_image_card(path_value: Path, title: str, height: int = 220) -> ft.Container:
        if path_value.exists():
            image_content: ft.Control = ft.Image(
                src=str(path_value),
                fit="contain",
                height=height,
            )
        else:
            image_content = ft.Container(
                height=height,
                alignment=ft.Alignment(0, 0),
                content=ft.Text(f"Missing asset: {title}", size=11, color=LIQUID_MUTED),
            )
        return ft.Container(
            padding=ft.padding.symmetric(horizontal=8, vertical=8),
            border_radius=18,
            bgcolor=ft.Colors.with_opacity(0.62, LIQUID_SURFACE_ALT),
            border=ft.border.all(1, ft.Colors.with_opacity(0.32, ft.Colors.WHITE)),
            content=image_content,
        )

    def _about_link_chip(icon_name: str, label: str, url_value: str) -> ft.Container:
        target_url = _normalise_web_url(url_value)
        has_url = bool(target_url)
        return ft.Container(
            ink=has_url,
            border_radius=14,
            padding=ft.padding.symmetric(horizontal=10, vertical=8),
            bgcolor=ft.Colors.with_opacity(0.62, LIQUID_SURFACE_ALT),
            border=ft.border.all(1, ft.Colors.with_opacity(0.30, ft.Colors.WHITE)),
            opacity=1.0 if has_url else 0.48,
            on_click=(
                (lambda _, target=target_url: _open_external_link(target))
                if has_url
                else None
            ),
            content=ft.Row(
                spacing=6,
                tight=True,
                controls=[
                    ft.Icon(icon_name, size=14, color=LIQUID_ACCENT),
                    ft.Text(label, size=11, color=LIQUID_TEXT),
                ],
            ),
        )

    simon_links_row = ft.Row(
        wrap=True,
        spacing=8,
        run_spacing=8,
        controls=[
            _about_link_chip(ft.Icons.LANGUAGE, "Website", about_links.get("simon_website", "")),
            _about_link_chip(ft.Icons.CODE, "GitHub", about_links.get("simon_github", "")),
            _about_link_chip(ft.Icons.SCIENCE, "ResearchGate", about_links.get("simon_researchgate", "")),
        ],
    )
    bede_links_row = ft.Row(
        wrap=True,
        spacing=8,
        run_spacing=8,
        controls=[
            _about_link_chip(ft.Icons.LANGUAGE, "Website", about_links.get("bede_website", "")),
            _about_link_chip(ft.Icons.CODE, "GitHub", about_links.get("bede_github", "")),
            _about_link_chip(ft.Icons.SCIENCE, "ResearchGate", about_links.get("bede_researchgate", "")),
        ],
    )

    about_popup_dialog_container = ft.Container(
        width=1240,
        padding=ft.padding.symmetric(horizontal=12, vertical=8),
        content=_glass_panel(
            padding=24,
            variant="modal",
            accent=LIQUID_ACCENT,
            content=ft.Column(
                spacing=14,
                scroll=ft.ScrollMode.AUTO,
                controls=[
                    ft.Row(
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        controls=[
                            ft.Column(
                                spacing=2,
                                controls=[
                                    ft.Text(
                                        "About ICE CREAMS",
                                        size=24,
                                        weight=ft.FontWeight.W_700,
                                        color=LIQUID_TEXT,
                                    ),
                                     ft.Text(
                                         "People, project, and partners",
                                         size=12,
                                         color=LIQUID_SUBTEXT,
                                     ),
                                     ft.Text(
                                         f"App version {APP_VERSION}",
                                         size=12,
                                         weight=ft.FontWeight.W_600,
                                         color=LIQUID_TEXT,
                                     ),
                                 ],
                             ),
                             about_popup_close_button,
                         ],
                     ),
                    ft.ResponsiveRow(
                        columns=12,
                        run_spacing=10,
                        controls=[
                            ft.Container(
                                col={"xs": 12, "lg": 5},
                                content=_about_image_card(project_logo_path, "ICE CREAMS logo", height=180),
                            ),
                            ft.Container(
                                col={"xs": 12, "lg": 7},
                                content=ft.Column(
                                    spacing=8,
                                    controls=[
                                        _about_image_card(bicome_logo_path, "BiCOME logo", height=94),
                                        ft.Row(
                                            spacing=8,
                                            controls=[
                                                ft.Container(
                                                    expand=True,
                                                    content=_about_image_card(
                                                        esa_logo_path, "ESA logo", height=76
                                                    ),
                                                ),
                                                ft.Container(
                                                    expand=True,
                                                    content=_about_image_card(
                                                        nantes_logo_path, "Nantes Universite logo", height=76
                                                    ),
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ),
                        ],
                    ),
                    ft.Container(
                        padding=ft.padding.symmetric(horizontal=14, vertical=12),
                        border_radius=16,
                        bgcolor=ft.Colors.with_opacity(0.62, LIQUID_SURFACE_ALT),
                        border=ft.border.all(1, ft.Colors.with_opacity(0.30, ft.Colors.WHITE)),
                        content=ft.Column(
                            spacing=6,
                            controls=[
                                ft.Text(about_project_statement, size=12, color=LIQUID_SUBTEXT),
                                ft.Text(about_ui_statement, size=12, color=LIQUID_SUBTEXT),
                                _about_link_chip(
                                    ft.Icons.HUB,
                                    "ICE CREAMS repository",
                                    about_links.get("repo", ""),
                                ),
                            ],
                        ),
                    ),
                    ft.ResponsiveRow(
                        columns=12,
                        run_spacing=12,
                        controls=[
                            ft.Container(
                                col={"xs": 12, "lg": 6},
                                content=ft.Container(
                                    padding=ft.padding.symmetric(horizontal=14, vertical=14),
                                    border_radius=18,
                                    bgcolor=ft.Colors.with_opacity(0.62, LIQUID_SURFACE_ALT),
                                    border=ft.border.all(1, ft.Colors.with_opacity(0.30, ft.Colors.WHITE)),
                                    content=ft.Column(
                                        spacing=8,
                                        controls=[
                                            _about_image_card(simon_photo_path, "Simon Oiry", height=260),
                                            ft.Text(
                                                "Simon Oiry",
                                                size=16,
                                                weight=ft.FontWeight.W_700,
                                                color=LIQUID_TEXT,
                                            ),
                                            ft.Text(
                                                "PhD researcher and designer/developer of this ICE CREAMS Studio UI.",
                                                size=12,
                                                color=LIQUID_SUBTEXT,
                                            ),
                                            simon_links_row,
                                        ],
                                    ),
                                ),
                            ),
                            ft.Container(
                                col={"xs": 12, "lg": 6},
                                content=ft.Container(
                                    padding=ft.padding.symmetric(horizontal=14, vertical=14),
                                    border_radius=18,
                                    bgcolor=ft.Colors.with_opacity(0.62, LIQUID_SURFACE_ALT),
                                    border=ft.border.all(1, ft.Colors.with_opacity(0.30, ft.Colors.WHITE)),
                                    content=ft.Column(
                                        spacing=8,
                                        controls=[
                                            _about_image_card(bede_photo_path, "Bede Davies", height=260),
                                            ft.Text(
                                                "Bede Davies",
                                                size=16,
                                                weight=ft.FontWeight.W_700,
                                                color=LIQUID_TEXT,
                                            ),
                                            ft.Text(
                                                "Post-doctoral researcher and creator of the ICE CREAMS algorithm.",
                                                size=12,
                                                color=LIQUID_SUBTEXT,
                                            ),
                                            bede_links_row,
                                        ],
                                    ),
                                ),
                            ),
                        ],
                    ),
                ],
            ),
        ),
    )

    training_source_field = ft.TextField(
        value="",
        hint_text="Select one or more training CSV files",
        read_only=True,
        border_radius=18,
        filled=True,
        fill_color=ft.Colors.with_opacity(0.72, LIQUID_SURFACE_ALT),
        border_color=ft.Colors.with_opacity(0.32, "#A6BFD9"),
        focused_border_color=LIQUID_ACCENT,
        color=LIQUID_TEXT,
        text_size=14,
        height=56,
        content_padding=ft.padding.symmetric(horizontal=16, vertical=14),
    )
    training_output_dir_field = ft.TextField(
        value=str(default_models_dir),
        hint_text="New models are saved automatically to the default models folder",
        read_only=True,
        border_radius=18,
        filled=True,
        fill_color=ft.Colors.with_opacity(0.72, LIQUID_SURFACE_ALT),
        border_color=ft.Colors.with_opacity(0.32, "#A6BFD9"),
        focused_border_color=LIQUID_ACCENT,
        color=LIQUID_TEXT,
        text_size=14,
        height=56,
        content_padding=ft.padding.symmetric(horizontal=16, vertical=14),
    )
    training_model_name_field = ft.TextField(
        value=f"ICECREAMS_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl",
        hint_text="Model filename",
        border_radius=18,
        filled=True,
        fill_color=ft.Colors.with_opacity(0.72, LIQUID_SURFACE_ALT),
        border_color=ft.Colors.with_opacity(0.32, "#A6BFD9"),
        focused_border_color=LIQUID_ACCENT,
        color=LIQUID_TEXT,
        text_size=14,
        content_padding=ft.padding.symmetric(horizontal=16, vertical=14),
    )
    training_epochs_field = ft.TextField(
        value="20",
        hint_text="1 to 1000",
        border_radius=18,
        filled=True,
        fill_color=ft.Colors.with_opacity(0.72, LIQUID_SURFACE_ALT),
        border_color=ft.Colors.with_opacity(0.32, "#A6BFD9"),
        focused_border_color=LIQUID_ACCENT,
        color=LIQUID_TEXT,
        text_size=14,
        height=56,
        content_padding=ft.padding.symmetric(horizontal=16, vertical=14),
    )
    training_split_field = ft.TextField(
        value="30",
        hint_text="1 to 99 (%)",
        border_radius=18,
        filled=True,
        fill_color=ft.Colors.with_opacity(0.72, LIQUID_SURFACE_ALT),
        border_color=ft.Colors.with_opacity(0.32, "#A6BFD9"),
        focused_border_color=LIQUID_ACCENT,
        color=LIQUID_TEXT,
        text_size=14,
        height=56,
        content_padding=ft.padding.symmetric(horizontal=16, vertical=14),
    )
    training_mode_dropdown = ft.Dropdown(
        value=DEFAULT_FEATURE_MODE,
        options=[
            ft.dropdown.Option(
                key=FEATURE_MODE_HIGH_SPATIAL_ACCURACY,
                text=feature_mode_label(FEATURE_MODE_HIGH_SPATIAL_ACCURACY),
            ),
            ft.dropdown.Option(
                key=FEATURE_MODE_HIGH_SPECTRAL_COMPLEXITY,
                text=feature_mode_label(FEATURE_MODE_HIGH_SPECTRAL_COMPLEXITY),
            ),
        ],
        border_radius=18,
        filled=True,
        fill_color=ft.Colors.with_opacity(0.72, LIQUID_SURFACE_ALT),
        border_color=ft.Colors.with_opacity(0.32, "#A6BFD9"),
        focused_border_color=LIQUID_ACCENT,
        color=LIQUID_TEXT,
        text_size=14,
        height=56,
        content_padding=ft.padding.symmetric(horizontal=16, vertical=14),
    )
    training_spectral_cnn_checkbox = ft.Checkbox(
        value=False,
        label="Train as Spectral 1D CNN",
        active_color=LIQUID_ACCENT,
        check_color=ft.Colors.WHITE,
        label_style=ft.TextStyle(size=13, color=LIQUID_TEXT, weight=ft.FontWeight.W_500),
    )
    training_sequence_standardization_checkbox = ft.Checkbox(
        value=DEFAULT_SPECTRAL_CNN_USE_STANDARDIZED_REFLECTANCE,
        label="Include Standardized Reflectance Channel",
        active_color=LIQUID_ACCENT,
        check_color=ft.Colors.WHITE,
        label_style=ft.TextStyle(size=13, color=LIQUID_TEXT, weight=ft.FontWeight.W_500),
        disabled=True,
    )

    train_progress = ft.ProgressBar(
        value=0,
        bar_height=8,
        border_radius=999,
        color=LIQUID_ACCENT,
        bgcolor=ft.Colors.with_opacity(0.34, "#B5CAE2"),
    )
    train_spinner = ft.ProgressRing(
        width=18,
        height=18,
        stroke_width=2.2,
        color=LIQUID_ACCENT,
        visible=False,
    )
    train_status = ft.Text(
        "Select training CSV files and/or folders containing CSV files to begin.",
        size=13,
        color=LIQUID_SUBTEXT,
    )
    train_log = ft.ListView(
        expand=True,
        spacing=10,
        auto_scroll=True,
        scroll=ft.ScrollMode.AUTO,
    )

    validation_dataset_field = ft.TextField(
        value="",
        hint_text="Select a validation dataset (.csv or .xlsx)",
        read_only=True,
        border_radius=18,
        filled=True,
        fill_color=ft.Colors.with_opacity(0.72, LIQUID_SURFACE_ALT),
        border_color=ft.Colors.with_opacity(0.32, "#A6BFD9"),
        focused_border_color=LIQUID_ACCENT,
        color=LIQUID_TEXT,
        text_size=14,
        height=56,
        content_padding=ft.padding.symmetric(horizontal=16, vertical=14),
    )
    validation_model_dropdown = ft.Dropdown(
        value=str(default_apply_model) if default_apply_model else None,
        options=[
            ft.dropdown.Option(
                key=str(model_path),
                text=model_path.name,
            )
            for model_path in existing_models
        ],
        hint_text="Select a FastAI model from the models folder",
        enable_search=True,
        border_radius=18,
        filled=True,
        fill_color=ft.Colors.with_opacity(0.72, LIQUID_SURFACE_ALT),
        border_color=ft.Colors.with_opacity(0.32, "#A6BFD9"),
        focused_border_color=LIQUID_ACCENT,
        color=LIQUID_TEXT,
        text_size=14,
        height=56,
        content_padding=ft.padding.symmetric(horizontal=16, vertical=14),
    )
    validation_external_model_field = ft.TextField(
        value="",
        hint_text="Optional: choose an external .pkl model file",
        read_only=True,
        border_radius=18,
        filled=True,
        fill_color=ft.Colors.with_opacity(0.72, LIQUID_SURFACE_ALT),
        border_color=ft.Colors.with_opacity(0.32, "#A6BFD9"),
        focused_border_color=LIQUID_ACCENT,
        color=LIQUID_TEXT,
        text_size=14,
        height=56,
        content_padding=ft.padding.symmetric(horizontal=16, vertical=14),
    )

    def _select_default_model(model_paths: list[Path]) -> Path | None:
        return next(
            (
                model_path
                for model_path in model_paths
                if _normalise_model_name(model_path.stem) == preferred_model_key
            ),
            model_paths[0] if model_paths else None,
        )

    def refresh_model_dropdowns(
        preferred_model_path: str | None = None,
        refresh: bool = True,
    ) -> None:
        nonlocal existing_models, default_apply_model

        existing_models = (
            sorted(default_models_dir.rglob("*.pkl"))
            if default_models_dir.exists()
            else []
        )
        default_apply_model = _select_default_model(existing_models)

        option_payloads = [
            {
                "key": str(model_path),
                "text": model_path.name,
            }
            for model_path in existing_models
        ]
        apply_model_dropdown.options = [
            ft.dropdown.Option(key=item["key"], text=item["text"])
            for item in option_payloads
        ]
        validation_model_dropdown.options = [
            ft.dropdown.Option(key=item["key"], text=item["text"])
            for item in option_payloads
        ]

        preferred_value = ""
        if preferred_model_path:
            preferred_candidate = Path(preferred_model_path)
            if preferred_candidate.exists():
                preferred_value = str(preferred_candidate.resolve())

        available_values = {str(model_path) for model_path in existing_models}

        def _resolve_dropdown_value(current_value: str | None, *, allow_preferred: bool) -> str | None:
            if allow_preferred and preferred_value and preferred_value in available_values:
                return preferred_value
            normalized_current = (current_value or "").strip()
            if normalized_current and normalized_current in available_values:
                return normalized_current
            return str(default_apply_model) if default_apply_model else None

        apply_model_dropdown.value = _resolve_dropdown_value(apply_model_dropdown.value, allow_preferred=True)
        if not (validation_external_model_field.value or "").strip():
            validation_model_dropdown.value = _resolve_dropdown_value(
                validation_model_dropdown.value,
                allow_preferred=True,
            )

        if refresh:
            refresh_apply_preview()
            refresh_apply_run_button_state()
            refresh_validation_preview()
            request_ui_refresh(force=True)
    validation_label_column_field = ft.TextField(
        value="Label_Char",
        hint_text="Ground-truth label column name",
        border_radius=18,
        filled=True,
        fill_color=ft.Colors.with_opacity(0.72, LIQUID_SURFACE_ALT),
        border_color=ft.Colors.with_opacity(0.32, "#A6BFD9"),
        focused_border_color=LIQUID_ACCENT,
        color=LIQUID_TEXT,
        text_size=14,
        height=56,
        content_padding=ft.padding.symmetric(horizontal=16, vertical=14),
    )
    validation_mode_dropdown = ft.Dropdown(
        value=VALIDATION_MODE_MULTICLASS,
        options=[
            ft.dropdown.Option(
                key=VALIDATION_MODE_MULTICLASS,
                text="Use model classes (as-is)",
            ),
            ft.dropdown.Option(
                key=VALIDATION_MODE_PRESENCE_ABSENCE,
                text="Presence/absence of one class",
            ),
        ],
        border_radius=18,
        filled=True,
        fill_color=ft.Colors.with_opacity(0.72, LIQUID_SURFACE_ALT),
        border_color=ft.Colors.with_opacity(0.32, "#A6BFD9"),
        focused_border_color=LIQUID_ACCENT,
        color=LIQUID_TEXT,
        text_size=14,
        height=56,
        content_padding=ft.padding.symmetric(horizontal=16, vertical=14),
    )
    validation_target_class_field = ft.TextField(
        value=DEFAULT_TARGET_CLASS,
        hint_text="Target class for presence/absence mode",
        border_radius=18,
        filled=True,
        fill_color=ft.Colors.with_opacity(0.72, LIQUID_SURFACE_ALT),
        border_color=ft.Colors.with_opacity(0.32, "#A6BFD9"),
        focused_border_color=LIQUID_ACCENT,
        color=LIQUID_TEXT,
        text_size=14,
        height=56,
        content_padding=ft.padding.symmetric(horizontal=16, vertical=14),
        disabled=True,
    )
    validation_target_class_container = ft.Container(
        visible=False,
        content=validation_target_class_field,
    )
    validation_output_dir_field = ft.TextField(
        value=str(default_validation_output_dir),
        hint_text="Select an output folder for predictions and metrics CSVs",
        read_only=True,
        border_radius=18,
        filled=True,
        fill_color=ft.Colors.with_opacity(0.72, LIQUID_SURFACE_ALT),
        border_color=ft.Colors.with_opacity(0.32, "#A6BFD9"),
        focused_border_color=LIQUID_ACCENT,
        color=LIQUID_TEXT,
        text_size=14,
        height=56,
        content_padding=ft.padding.symmetric(horizontal=16, vertical=14),
    )
    validation_output_preview = ft.Text(
        "Select dataset, model, and output folder to preview validation output files.",
        size=12,
        color=LIQUID_SUBTEXT,
    )
    validation_progress = ft.ProgressBar(
        value=0,
        bar_height=8,
        border_radius=999,
        color=LIQUID_ACCENT,
        bgcolor=ft.Colors.with_opacity(0.34, "#B5CAE2"),
    )
    validation_spinner = ft.ProgressRing(
        width=18,
        height=18,
        stroke_width=2.2,
        color=LIQUID_ACCENT,
        visible=False,
    )
    validation_status = ft.Text(
        "Select dataset, model, and label column to begin validation.",
        size=13,
        color=LIQUID_SUBTEXT,
    )
    validation_log = ft.ListView(
        expand=True,
        spacing=10,
        auto_scroll=True,
        scroll=ft.ScrollMode.AUTO,
    )
    history_summary = ft.Text(
        "Run history",
        size=13,
        color=LIQUID_SUBTEXT,
    )
    history_status = ft.Text(
        "",
        size=12,
        color=LIQUID_MUTED,
    )
    history_table = ft.DataTable(
        columns=[
            ft.DataColumn(ft.Text("Date", color=LIQUID_TEXT, size=12, weight=ft.FontWeight.W_600)),
            ft.DataColumn(ft.Text("Model", color=LIQUID_TEXT, size=12, weight=ft.FontWeight.W_600)),
            ft.DataColumn(ft.Text("Location", color=LIQUID_TEXT, size=12, weight=ft.FontWeight.W_600)),
            ft.DataColumn(ft.Text("Status", color=LIQUID_TEXT, size=12, weight=ft.FontWeight.W_600)),
        ],
        rows=[],
        horizontal_lines=ft.BorderSide(1, ft.Colors.with_opacity(0.10, ft.Colors.WHITE)),
        vertical_lines=ft.BorderSide(1, ft.Colors.with_opacity(0.06, ft.Colors.WHITE)),
        heading_row_height=40,
        data_row_min_height=42,
        data_row_max_height=46,
        column_spacing=14,
    )
    history_table_container = ft.Container(
        height=360,
        content=ft.Column(
            scroll=ft.ScrollMode.AUTO,
            controls=[history_table],
        ),
    )
    history_visible_entries: list[dict[str, object]] = []
    history_row_lookup: dict[str, ft.DataRow] = {}
    history_location_text_lookup: dict[str, ft.Text] = {}
    history_selected_key = {"value": ""}
    history_selected_paths = {
        "input": "",
        "model": "",
        "output": "",
        "mask": "",
    }
    history_detail_heading = ft.Text(
        "Select a run",
        size=18,
        weight=ft.FontWeight.W_600,
        color=LIQUID_TEXT,
    )
    history_detail_subheading = ft.Text(
        "Choose a row on the left to display full run details.",
        size=12,
        color=LIQUID_SUBTEXT,
    )
    history_detail_status_value = ft.Text(
        "Status: -",
        size=12,
        weight=ft.FontWeight.W_600,
        color=LIQUID_TEXT,
    )
    history_detail_status_badge = ft.Container(
        border_radius=999,
        padding=ft.padding.symmetric(horizontal=12, vertical=6),
        bgcolor=ft.Colors.with_opacity(0.18, LIQUID_ACCENT),
        border=ft.border.all(1, ft.Colors.with_opacity(0.25, LIQUID_ACCENT)),
        content=history_detail_status_value,
    )
    history_detail_date_value = ft.Text("-", size=12, color=LIQUID_TEXT, selectable=True)
    history_detail_workflow_value = ft.Text("-", size=12, color=LIQUID_TEXT, selectable=True)
    history_detail_duration_value = ft.Text("-", size=12, color=LIQUID_TEXT, selectable=True)
    history_detail_extent_note_value = ft.Text("-", size=11, color=LIQUID_SUBTEXT, selectable=True)
    history_detail_error_value = ft.Text("-", size=12, color="#B23B4D", selectable=True)
    history_detail_notes_value = ft.Text("-", size=12, color=LIQUID_SUBTEXT, selectable=True)

    def _history_detail_card(
        title: str,
        value_control: ft.Control,
        col_value: dict[str, int] | None = None,
    ) -> ft.Container:
        return ft.Container(
            col=col_value or {"xs": 12, "md": 6},
            padding=12,
            border_radius=16,
            bgcolor=ft.Colors.with_opacity(0.62, LIQUID_SURFACE_ALT),
            border=ft.border.all(1, ft.Colors.with_opacity(0.28, ft.Colors.WHITE)),
            content=ft.Column(
                spacing=6,
                tight=True,
                controls=[
                    ft.Text(title, size=11, color=LIQUID_MUTED),
                    value_control,
                ],
            ),
        )

    def _history_path_card(
        title: str,
        action_button: ft.ElevatedButton,
    ) -> ft.Container:
        return ft.Container(
            col={"xs": 12, "md": 6},
            padding=12,
            border_radius=16,
            bgcolor=ft.Colors.with_opacity(0.62, LIQUID_SURFACE_ALT),
            border=ft.border.all(1, ft.Colors.with_opacity(0.28, ft.Colors.WHITE)),
            content=ft.Column(
                spacing=8,
                tight=True,
                controls=[
                    ft.Row(
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        controls=[
                            ft.Text(title, size=11, color=LIQUID_MUTED),
                            action_button,
                        ],
                    ),
                ],
            ),
        )

    history_open_button_style = ft.ButtonStyle(
        padding=ft.padding.symmetric(horizontal=10, vertical=6),
        color=LIQUID_ACCENT,
        bgcolor=ft.Colors.with_opacity(0.14, LIQUID_ACCENT),
        side=ft.BorderSide(1, ft.Colors.with_opacity(0.24, LIQUID_ACCENT)),
        shape=ft.RoundedRectangleBorder(radius=10),
        overlay_color=ft.Colors.TRANSPARENT,
        animation_duration=0,
        enable_feedback=False,
    )
    history_open_input_button = ft.ElevatedButton(
        "Open",
        icon=ft.Icons.FOLDER_OPEN,
        style=history_open_button_style,
        visible=False,
    )
    history_open_model_button = ft.ElevatedButton(
        "Open",
        icon=ft.Icons.FOLDER_OPEN,
        style=history_open_button_style,
        visible=False,
    )
    history_open_output_button = ft.ElevatedButton(
        "Open",
        icon=ft.Icons.FOLDER_OPEN,
        style=history_open_button_style,
        visible=False,
    )
    history_open_mask_button = ft.ElevatedButton(
        "Open",
        icon=ft.Icons.FOLDER_OPEN,
        style=history_open_button_style,
        visible=False,
    )
    history_show_map_button = ft.ElevatedButton(
        "Show extent map",
        icon=ft.Icons.MAP,
        style=history_open_button_style,
        disabled=True,
    )
    history_extent_card_content = ft.Column(
        spacing=8,
        tight=True,
        controls=[
            history_show_map_button,
            history_detail_extent_note_value,
        ],
    )
    history_details_container = ft.Container(
        height=360,
        content=ft.Column(
            spacing=12,
            scroll=ft.ScrollMode.AUTO,
            controls=[
                history_detail_heading,
                history_detail_subheading,
                history_detail_status_badge,
                ft.ResponsiveRow(
                    columns=12,
                    run_spacing=10,
                    spacing=10,
                    controls=[
                        _history_detail_card("Date", history_detail_date_value),
                        _history_detail_card("Workflow", history_detail_workflow_value),
                        _history_detail_card("Duration", history_detail_duration_value),
                        _history_detail_card("Mask extent", history_extent_card_content),
                    ],
                ),
                ft.ResponsiveRow(
                    columns=12,
                    run_spacing=10,
                    spacing=10,
                    controls=[
                        _history_path_card("Input dataset", history_open_input_button),
                        _history_path_card("Model", history_open_model_button),
                        _history_path_card("Output", history_open_output_button),
                        _history_path_card("Mask", history_open_mask_button),
                    ],
                ),
                ft.ResponsiveRow(
                    columns=12,
                    run_spacing=10,
                    spacing=10,
                    controls=[
                        _history_detail_card("Error", history_detail_error_value, {"xs": 12, "md": 12}),
                        _history_detail_card("Details", history_detail_notes_value, {"xs": 12, "md": 12}),
                    ],
                ),
            ],
        ),
    )

    def _is_batch_scene_source(path_value: str | None = None) -> bool:
        if not path_value:
            return False
        input_path = Path(path_value)
        return input_path.is_dir() and not input_path.name.upper().endswith(".SAFE")

    def _derive_scene_stem(scene_value: str) -> str:
        scene_name = Path(scene_value).name
        if scene_name.lower().endswith(".zip"):
            scene_name = scene_name[:-4]
        if scene_name.lower().endswith(".tiff"):
            scene_name = scene_name[:-5]
        elif scene_name.lower().endswith(".tif"):
            scene_name = scene_name[:-4]
        if scene_name.upper().endswith(".SAFE"):
            scene_name = scene_name[:-5]
        return scene_name

    def _derive_model_stem(model_value: str | None = None) -> str:
        selected_model = (model_value or apply_model_dropdown.value or "").strip()
        if not selected_model:
            return ""
        model_stem = Path(selected_model).stem
        sanitized = "".join(
            char if char.isalnum() or char in ("-", "_", ".") else "-"
            for char in model_stem
        ).strip("-_.")
        without_ice_creams_prefix = re.sub(
            r"(?i)^ice[_-]*creams?[_-]*",
            "",
            sanitized,
        ).strip("-_.")
        return without_ice_creams_prefix or sanitized

    def _apply_naming_pattern(model_value: str | None = None) -> str:
        model_stem = _derive_model_stem(model_value)
        if model_stem:
            return f"<SCENE_ID>_ICECREAMS_{model_stem}.tif"
        return "<SCENE_ID>_ICECREAMS_<MODEL>.tif"

    def build_apply_output_path(
        scene_value: str | None = None,
        output_value: str | None = None,
        model_value: str | None = None,
    ) -> str:
        scene_input = (scene_value or apply_safe_field.value).strip()
        output_folder = (output_value or apply_output_path_field.value).strip()
        if not output_folder:
            return ""
        scene_stem = _derive_scene_stem(scene_input) if scene_input else "ICECREAMS_output"
        model_stem = _derive_model_stem(model_value) or "MODEL"
        return str(Path(output_folder) / f"{scene_stem}_ICECREAMS_{model_stem}.tif")

    def suggest_apply_output_path() -> str:
        return str(default_apply_output_dir)

    def build_training_output_path() -> str:
        model_name = training_model_name_field.value.strip()
        if not model_name:
            return ""
        if not model_name.lower().endswith(".pkl"):
            model_name = f"{model_name}.pkl"
        return str(default_models_dir / model_name)

    def _derive_validation_dataset_stem(dataset_value: str | None = None) -> str:
        dataset_input = (dataset_value or validation_dataset_field.value).strip()
        if not dataset_input:
            return ""
        return Path(dataset_input).stem

    def resolve_validation_model_path() -> str:
        external_model = (validation_external_model_field.value or "").strip()
        if external_model:
            return external_model
        return (validation_model_dropdown.value or "").strip()

    def build_validation_predictions_output_path(
        dataset_value: str | None = None,
        model_value: str | None = None,
        output_value: str | None = None,
    ) -> str:
        output_folder = (output_value or validation_output_dir_field.value).strip()
        if not output_folder:
            return ""

        dataset_stem = _derive_validation_dataset_stem(dataset_value) or "validation"
        model_path = (model_value or resolve_validation_model_path()).strip()
        model_stem = _derive_model_stem(model_path) or "MODEL"
        return str(Path(output_folder) / f"{dataset_stem}__{model_stem}__predictions.csv")

    def build_validation_metrics_output_path(
        dataset_value: str | None = None,
        model_value: str | None = None,
        output_value: str | None = None,
    ) -> str:
        output_folder = (output_value or validation_output_dir_field.value).strip()
        if not output_folder:
            return ""

        dataset_stem = _derive_validation_dataset_stem(dataset_value) or "validation"
        model_path = (model_value or resolve_validation_model_path()).strip()
        model_stem = _derive_model_stem(model_path) or "MODEL"
        return str(Path(output_folder) / f"{dataset_stem}__{model_stem}__metrics.csv")

    def refresh_validation_preview() -> None:
        dataset_path = validation_dataset_field.value.strip()
        model_path = resolve_validation_model_path()
        output_folder = validation_output_dir_field.value.strip()

        if not output_folder:
            validation_output_preview.value = "Choose an output folder."
            return
        if not dataset_path and not model_path:
            validation_output_preview.value = "Select a dataset and model to preview output files."
            return
        if not dataset_path:
            validation_output_preview.value = "Select a dataset to preview output files."
            return
        if not model_path:
            validation_output_preview.value = "Select a model to preview output files."
            return

        predictions_path = build_validation_predictions_output_path(
            dataset_value=dataset_path,
            model_value=model_path,
            output_value=output_folder,
        )
        metrics_path = build_validation_metrics_output_path(
            dataset_value=dataset_path,
            model_value=model_path,
            output_value=output_folder,
        )
        validation_output_preview.value = (
            f"Predictions: {predictions_path}\nMetrics: {metrics_path}"
        )

    def _discover_training_folder_csvs(folder_path: str) -> list[str]:
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            return []
        return sorted(
            {
                str(csv_file.resolve())
                for csv_file in folder.rglob("*.csv")
                if csv_file.is_file()
            }
        )

    def _rebuild_training_csv_selection() -> None:
        valid_file_selections = sorted(
            {
                str(Path(csv_path).resolve())
                for csv_path in selected_training_csv_files
                if csv_path
                and Path(csv_path).is_file()
                and str(csv_path).lower().endswith(".csv")
            }
        )
        selected_training_csv_files.clear()
        selected_training_csv_files.extend(valid_file_selections)

        valid_folder_selections = sorted(
            {
                str(Path(folder_path).resolve())
                for folder_path in selected_training_csv_folders
                if folder_path and Path(folder_path).is_dir()
            }
        )
        selected_training_csv_folders.clear()
        selected_training_csv_folders.extend(valid_folder_selections)

        discovered_from_folders: list[str] = []
        for folder_path in selected_training_csv_folders:
            discovered_from_folders.extend(_discover_training_folder_csvs(folder_path))

        merged_csvs = sorted(set(selected_training_csv_files + discovered_from_folders))
        selected_training_csvs.clear()
        selected_training_csvs.extend(merged_csvs)

    def _training_picker_initial_directory() -> str:
        if selected_training_csv_files:
            return _resolve_initial_directory(selected_training_csv_files[0])
        if selected_training_csv_folders:
            return _resolve_initial_directory(selected_training_csv_folders[0])
        if selected_training_csvs:
            return _resolve_initial_directory(selected_training_csvs[0])
        if default_training_source.exists():
            return str(default_training_source)
        return _resolve_initial_directory(training_source_field.value)

    def _training_source_overlay_path() -> str:
        if not selected_training_csvs:
            return ""
        if len(selected_training_csvs) == 1:
            return selected_training_csvs[0]
        try:
            return str(Path(os.path.commonpath(selected_training_csvs)))
        except ValueError:
            return str(Path(selected_training_csvs[0]).parent)

    def _update_training_source_field() -> None:
        folder_count = len(selected_training_csv_folders)
        file_count = len(selected_training_csv_files)
        csv_count = len(selected_training_csvs)

        if not selected_training_csvs and not selected_training_csv_folders and not selected_training_csv_files:
            training_source_field.value = ""
            return

        if csv_count == 1 and file_count <= 1 and folder_count == 0:
            training_source_field.value = selected_training_csvs[0]
            return

        preview_names = [Path(path).name for path in selected_training_csvs[:3]]
        preview_suffix = f", +{csv_count - len(preview_names)} more" if csv_count > len(preview_names) else ""
        source_parts: list[str] = []
        if file_count:
            source_parts.append(f"{file_count} file(s)")
        if folder_count:
            source_parts.append(f"{folder_count} folder(s)")
        source_summary = " + ".join(source_parts) if source_parts else "selection"

        if csv_count == 0:
            training_source_field.value = f"{source_summary} selected. No CSV files found yet."
            return

        training_source_field.value = (
            f"{csv_count} CSV files from {source_summary}: "
            f"{', '.join(preview_names)}{preview_suffix}"
        )

    def _apply_preflight_signature(
        scene_value: str | None = None,
        output_value: str | None = None,
        model_value: str | None = None,
    ) -> tuple[str, str, str]:
        return (
            (scene_value or apply_safe_field.value).strip(),
            (output_value or apply_output_path_field.value).strip(),
            (model_value or apply_model_dropdown.value or "").strip(),
        )

    def _apply_has_preflight_inputs() -> bool:
        return all(_apply_preflight_signature())

    def _reset_apply_preflight_state() -> None:
        apply_preflight_state["signature"] = None
        apply_preflight_state["scene_batch_info"] = None
        apply_preflight_state["pending_scene_inputs"] = []
        apply_preflight_state["skipped_existing_scene_inputs"] = []
        apply_preflight_state["skipped_existing_outputs"] = []

    def _apply_preflight_matches_current_selection() -> bool:
        cached_signature = apply_preflight_state.get("signature")
        return bool(cached_signature and cached_signature == _apply_preflight_signature())

    def _apply_can_run() -> bool:
        return (
            not state["busy"]
            and not bool(apply_preflight_state["running"])
            and bool(apply_safe_field.value.strip())
            and bool(apply_mask_field.value.strip())
            and bool(apply_output_path_field.value.strip())
            and bool((apply_model_dropdown.value or "").strip())
            and _apply_preflight_matches_current_selection()
        )

    def refresh_apply_run_button_state() -> None:
        apply_run_button.disabled = not _apply_can_run()
        if state["busy"]:
            apply_run_button.tooltip = "Apply workflow is running."
            return
        if apply_preflight_state["running"]:
            apply_run_button.tooltip = "Checking already processed outputs for the selected scenes."
            return
        if not apply_safe_field.value.strip():
            apply_run_button.tooltip = "Choose an input scene or batch folder."
            return
        if not apply_output_path_field.value.strip():
            apply_run_button.tooltip = "Choose an output folder."
            return
        if not (apply_model_dropdown.value or "").strip():
            apply_run_button.tooltip = "Choose a model."
            return
        if not apply_mask_field.value.strip():
            apply_run_button.tooltip = "Choose a mask shapefile."
            return
        if not _apply_preflight_matches_current_selection():
            apply_run_button.tooltip = "Waiting for the pre-run output check to finish."
            return
        apply_run_button.tooltip = "Run Apply workflow (Ctrl+Enter on Apply tab)"

    def _set_apply_preflight_running(is_running: bool, status_text: str | None = None) -> None:
        apply_preflight_state["running"] = is_running
        apply_spinner.visible = is_running
        if is_running and status_text and not state["busy"]:
            set_app_status("busy", status_text)
        refresh_apply_run_button_state()
        refresh_apply_preview()

    def _collect_apply_preflight_plan(
        input_source: str,
        output_folder: str,
        selected_model: str,
    ) -> dict[str, object]:
        scene_batch_info = discover_scene_batch_info(input_source)
        pending_scene_inputs: list[str] = []
        skipped_existing_scene_inputs: list[str] = []
        skipped_existing_outputs: list[str] = []
        for scene_record in scene_batch_info["selected"]:
            scene_input = str(scene_record["path"])
            scene_output = build_apply_output_path(scene_input, output_folder, selected_model)
            if scene_output and Path(scene_output).is_file():
                skipped_existing_scene_inputs.append(scene_input)
                skipped_existing_outputs.append(scene_output)
            else:
                pending_scene_inputs.append(scene_input)
        return {
            "scene_batch_info": scene_batch_info,
            "pending_scene_inputs": pending_scene_inputs,
            "skipped_existing_scene_inputs": skipped_existing_scene_inputs,
            "skipped_existing_outputs": skipped_existing_outputs,
        }

    async def _run_apply_preflight_scan(status_text: str) -> None:
        input_source, output_folder, selected_model = _apply_preflight_signature()
        if not (input_source and output_folder and selected_model):
            return

        current_token = int(apply_preflight_state.get("token", 0))
        current_signature = (input_source, output_folder, selected_model)
        _set_apply_preflight_running(True, status_text)
        push_apply_status(status_text)
        try:
            plan = await asyncio.to_thread(
                _collect_apply_preflight_plan,
                input_source,
                output_folder,
                selected_model,
            )
        except Exception as exc:  # noqa: BLE001 - surface preflight errors to the user.
            if current_token != int(apply_preflight_state.get("token", -1)):
                return
            _reset_apply_preflight_state()
            _set_apply_preflight_running(False)
            push_apply_status(f"Pre-run output check failed: {exc}", level="error")
            set_app_status("error", "Apply pre-run check failed.")
            request_ui_refresh(force=True)
            return

        if current_token != int(apply_preflight_state.get("token", -1)):
            return

        apply_preflight_state["signature"] = current_signature
        apply_preflight_state["scene_batch_info"] = plan["scene_batch_info"]
        apply_preflight_state["pending_scene_inputs"] = list(plan["pending_scene_inputs"])
        apply_preflight_state["skipped_existing_scene_inputs"] = list(plan["skipped_existing_scene_inputs"])
        apply_preflight_state["skipped_existing_outputs"] = list(plan["skipped_existing_outputs"])
        _set_apply_preflight_running(False)

        scene_batch_info = plan["scene_batch_info"]
        selected_count = len(scene_batch_info["selected"])
        pending_count = len(plan["pending_scene_inputs"])
        skipped_existing_count = len(plan["skipped_existing_outputs"])
        duplicate_count = int(scene_batch_info.get("duplicate_count", 0))
        ignored_count = int(scene_batch_info.get("ignored_count", 0))

        if pending_count > 0:
            push_apply_status(
                f"Checked {selected_count} scene(s). "
                f"{pending_count} scene(s) ready to process and {skipped_existing_count} already processed."
            )
        elif skipped_existing_count > 0:
            push_apply_status(
                f"Checked {selected_count} scene(s). All matching outputs already exist.",
                level="warning",
            )
        else:
            push_apply_status(
                f"Checked {selected_count} scene(s). Ready to run.",
            )

        if duplicate_count > 0:
            push_apply_status(
                f"Warning: {duplicate_count} duplicate scene(s) detected. "
                f"{scene_batch_info['skipped_count']} duplicate file(s) will be skipped (SAFE preferred over TIFF preferred over ZIP).",
                level="warning",
            )
        if ignored_count > 0:
            push_apply_status(
                f"Warning: {ignored_count} unsupported .SAFE/.zip/.tif/.tiff item(s) were ignored.",
                level="warning",
            )

        ready_message = (
            f"Apply pre-run check complete. {pending_count} scene(s) ready."
            if pending_count > 0
            else "Apply pre-run check complete."
        )
        set_app_status("ready" if apply_mask_field.value.strip() else "input", ready_message)
        request_ui_refresh(force=True)

    def schedule_apply_preflight_scan(status_text: str = "Checking already processed outputs.") -> None:
        apply_preflight_state["token"] = int(apply_preflight_state.get("token", 0)) + 1
        _reset_apply_preflight_state()
        refresh_apply_run_button_state()
        refresh_apply_preview()
        if not _apply_has_preflight_inputs():
            if not state["busy"]:
                set_app_status("input", "Ready for a new run.")
            request_ui_refresh(force=True)
            return
        asyncio.get_running_loop().create_task(_run_apply_preflight_scan(status_text))

    def refresh_apply_preview() -> None:
        output_folder = apply_output_path_field.value.strip()
        scene_input = apply_safe_field.value.strip()
        naming_pattern = _apply_naming_pattern()
        if not output_folder:
            apply_output_preview.value = "Choose an output folder. GeoTIFF names include the selected model."
            return
        if not scene_input:
            apply_output_preview.value = (
                f"Output folder: {output_folder} | Naming: {naming_pattern}"
            )
            return
        if apply_preflight_state["running"] and _apply_has_preflight_inputs():
            apply_output_preview.value = (
                f"Checking already processed outputs in {output_folder} for the selected scene set..."
            )
            return
        if _apply_preflight_matches_current_selection():
            skipped_existing_outputs = list(apply_preflight_state["skipped_existing_outputs"])
            pending_scene_inputs = list(apply_preflight_state["pending_scene_inputs"])
            if _is_batch_scene_source(scene_input):
                apply_output_preview.value = (
                    f"Batch output folder: {output_folder} | "
                    f"Ready: {len(pending_scene_inputs)} | "
                    f"Already processed: {len(skipped_existing_outputs)} | "
                    f"Naming: {naming_pattern}"
                )
                return
            if skipped_existing_outputs and not pending_scene_inputs:
                apply_output_preview.value = (
                    f"Output already exists and will be skipped: {skipped_existing_outputs[0]}"
                )
                return
        if _is_batch_scene_source(scene_input):
            apply_output_preview.value = (
                f"Batch output folder: {output_folder} | Naming: {naming_pattern}"
            )
            return
        apply_output_preview.value = f"Next output file: {build_apply_output_path(scene_input, output_folder)}"

    def _format_compact_list(values: list[str], max_items: int = 6) -> str:
        if not values:
            return "-"
        if len(values) <= max_items:
            return ", ".join(values)
        return f"{', '.join(values[:max_items])}, +{len(values) - max_items} more"

    def _flatten_batch_records(batch_info: dict[str, object]) -> list[dict[str, str]]:
        records: list[dict[str, str]] = []
        for selected_record in batch_info["selected"]:
            records.append(
                {
                    "scene_id": str(selected_record["scene_id"]),
                    "format": str(selected_record["format"]),
                    "acquired": str(selected_record["acquisition_date"] or "Unknown"),
                    "status": "Used",
                    "path": str(selected_record["path"]),
                }
            )

        for duplicate_group in batch_info["duplicate_groups"]:
            for skipped_record in duplicate_group["skipped"]:
                records.append(
                    {
                        "scene_id": str(skipped_record["scene_id"]),
                        "format": str(skipped_record["format"]),
                        "acquired": str(skipped_record["acquisition_date"] or "Unknown"),
                        "status": "Skipped duplicate",
                        "path": str(skipped_record["path"]),
                    }
                )

        return sorted(
            records,
            key=lambda row: (
                row["acquired"] if row["acquired"] != "Unknown" else "9999-99-99",
                row["scene_id"],
                0 if row["status"] == "Used" else 1,
                row["format"],
            ),
        )

    def hide_about_popup() -> None:
        about_popup_blocker.visible = False
        _sync_modal_shell_emphasis()

    def show_about_popup(_: ft.ControlEvent | None = None) -> None:
        about_popup_blocker.visible = True
        _sync_modal_shell_emphasis()
        _refresh_ui_surface(shell, about_popup_blocker)

    def close_about_popup(_: ft.ControlEvent | None = None) -> None:
        hide_about_popup()
        _refresh_ui_surface(shell, about_popup_blocker)

    def hide_update_popup() -> None:
        if update_state["install_running"]:
            return
        update_popup_blocker.visible = False
        update_popup_status.value = ""
        update_popup_status.visible = False
        _sync_modal_shell_emphasis()

    def show_update_popup(
        manifest: dict[str, object],
        error_message: str = "",
    ) -> None:
        update_state["manifest"] = dict(manifest)
        latest_version = _manifest_string(manifest, "version") or "Unknown"
        installer_name = _manifest_string(manifest, "installer_name") or "Installer.exe"
        published_at = _manifest_string(manifest, "published_at") or "-"
        update_popup_summary.value = (
            f"Version {latest_version} is available. You are currently running {APP_VERSION}."
        )
        update_popup_details.value = (
            f"Current version: {APP_VERSION}\n"
            f"Latest version: {latest_version}\n"
            f"Installer: {installer_name}\n"
            f"Published: {published_at}\n"
            "Selecting 'Update now' downloads the installer from the hosted "
            "ICE_CREAMS_STUDIO repository, closes the app, installs the update, "
            "and relaunches the application."
        )
        release_notes_url = _normalise_web_url(_manifest_string(manifest, "release_notes_url"))
        update_popup_release_notes_button.visible = bool(release_notes_url)
        update_popup_release_notes_button.on_click = (
            (lambda _, target=release_notes_url: _open_external_link(target))
            if release_notes_url
            else None
        )
        update_popup_status.value = error_message
        update_popup_status.visible = bool(error_message)
        update_popup_blocker.visible = True
        _sync_modal_shell_emphasis()
        _refresh_ui_surface(shell, update_popup_blocker)

    def close_update_popup(_: ft.ControlEvent | None = None) -> None:
        if update_state["install_running"]:
            return
        hide_update_popup()
        if not state["busy"]:
            report_idle("Ready for a new run.")
        _refresh_ui_surface(shell, update_popup_blocker)

    def _close_application_window() -> None:
        try:
            if hasattr(page, "window") and hasattr(page.window, "close"):
                page.window.close()
                return
        except Exception:
            pass
        for closer_name in ("window_destroy", "window_close"):
            try:
                getattr(page, closer_name)()
                return
            except Exception:
                continue

    async def _run_update_check(*, interactive: bool = False) -> None:
        if update_state["check_running"] or update_state["install_running"]:
            return

        update_state["check_running"] = True
        previous_status = str(app_status.value or "").strip() or "Ready for a new run."
        update_state["status_before_check"] = previous_status
        if not state["busy"]:
            set_app_status("busy", "Checking for application updates...")
            request_ui_refresh(force=True)

        try:
            manifest = await asyncio.to_thread(_fetch_update_manifest)
        except Exception as exc:
            if interactive and not state["busy"]:
                set_app_status("error", f"Update check failed: {exc}")
                request_ui_refresh(force=True)
            elif not state["busy"] and not update_popup_blocker.visible:
                report_idle(previous_status)
                request_ui_refresh(force=True)
            return
        finally:
            update_state["check_running"] = False

        latest_version = _manifest_string(manifest, "version")
        if not latest_version or not _is_newer_version(latest_version, APP_VERSION):
            update_state["manifest"] = None
            if interactive and not state["busy"]:
                set_app_status("ready", f"ICE CREAMS Studio {APP_VERSION} is up to date.")
                request_ui_refresh(force=True)
            elif not state["busy"] and not update_popup_blocker.visible:
                report_idle(previous_status)
                request_ui_refresh(force=True)
            return

        if not state["busy"]:
            set_app_status("ready", f"Version {latest_version} is available.")
        show_update_popup(manifest)
        request_ui_refresh(force=True)

    async def _run_startup_update_check() -> None:
        await _run_update_check(interactive=False)

    async def _run_manual_update_check() -> None:
        await _run_update_check(interactive=True)

    def trigger_manual_update_check(_: ft.ControlEvent | None = None) -> None:
        page.run_task(_run_manual_update_check)

    async def run_update_install(_: ft.ControlEvent | None = None) -> None:
        if update_state["install_running"]:
            return

        manifest = update_state.get("manifest")
        if not isinstance(manifest, dict):
            return
        if state["busy"]:
            show_update_popup(
                manifest,
                error_message="Wait for the current workflow to finish before starting the update.",
            )
            return

        latest_version = _manifest_string(manifest, "version") or "Unknown"
        installer_name = _manifest_string(manifest, "installer_name") or "ICE_CREAMS_Installer.exe"
        installer_url = _resolve_manifest_installer_url(manifest)
        if not installer_url:
            show_update_popup(manifest, error_message="The update manifest does not contain a valid installer URL.")
            return

        temp_update_dir = Path(tempfile.gettempdir()) / "ICE_CREAMS_Studio" / "updates"
        installer_path = temp_update_dir / installer_name
        expected_hash = _manifest_string(manifest, "sha256").upper()
        update_state["install_running"] = True
        hide_update_popup()
        set_global_busy(True, f"Downloading version {latest_version}.")
        state["operation_mode"] = "update"
        update_overlay(
            title="Installing Update",
            detail=f"Downloading ICE CREAMS Studio {latest_version}.",
            progress=0,
            counter="Preparing download",
            job="App update",
            source="",
            destination=str(installer_path),
        )
        request_ui_refresh(force=True)

        loop = asyncio.get_running_loop()

        def _dispatch_update_progress(bytes_downloaded: int, total_bytes: int) -> None:
            progress_value = 0.0
            counter_text = "Downloading installer"
            if total_bytes > 0:
                progress_value = max(0.0, min(1.0, bytes_downloaded / total_bytes))
                counter_text = (
                    f"{bytes_downloaded / (1024 * 1024):.1f} MB of "
                    f"{total_bytes / (1024 * 1024):.1f} MB"
                )
            update_overlay(
                title="Installing Update",
                detail=f"Downloading ICE CREAMS Studio {latest_version}.",
                progress=progress_value,
                counter=counter_text,
                job="App update",
                destination=str(installer_path),
            )
            request_ui_refresh()

        def schedule_update_progress(bytes_downloaded: int, total_bytes: int) -> None:
            loop.call_soon_threadsafe(_dispatch_update_progress, bytes_downloaded, total_bytes)

        try:
            download_result = await asyncio.to_thread(
                _download_update_installer,
                installer_url,
                installer_path,
                schedule_update_progress,
            )
            actual_hash = str(download_result.get("sha256", "")).upper()
            if expected_hash and actual_hash != expected_hash:
                raise ValueError("Downloaded installer hash does not match the manifest.")

            update_overlay(
                title="Installing Update",
                detail="Launching the installer and closing the app.",
                progress=1.0,
                counter=f"Version {latest_version}",
                job="App update",
                destination=str(installer_path),
            )
            request_ui_refresh(force=True)
            await asyncio.to_thread(_launch_windows_update_installer, installer_path)
            await asyncio.sleep(0.35)
            _close_application_window()
        except Exception as exc:
            update_state["install_running"] = False
            set_global_busy(False)
            overlay_blocker.visible = False
            set_app_status("error", f"Update failed: {exc}")
            show_update_popup(manifest, error_message=f"Update failed: {exc}")
            request_ui_refresh(force=True)

    def hide_batch_popup() -> None:
        batch_popup_blocker.visible = False
        _sync_modal_shell_emphasis()

    def show_batch_popup(batch_info: dict[str, object]) -> None:
        format_counts = batch_info["format_counts"]
        ignored_count = int(batch_info.get("ignored_count", 0))
        batch_popup_summary.value = (
            f"Detected {batch_info['raw_count']} input(s). {batch_info['unique_count']} unique scene(s) selected."
        )
        format_parts = [
            f"{int(format_counts.get('SAFE', 0))} SAFE",
            f"{int(format_counts.get('TIFF', 0))} TIFF",
            f"{int(format_counts.get('ZIP', 0))} ZIP",
        ]
        batch_popup_formats.value = (
            f"Formats used: {', '.join(format_parts)}."
        )
        batch_popup_dates.value = (
            f"Acquisition dates: {_format_compact_list(batch_info['acquisition_dates'], max_items=10)}"
        )

        duplicate_count = int(batch_info["duplicate_count"])
        warning_messages: list[str] = []
        if duplicate_count > 0:
            warning_messages.append(
                f"{duplicate_count} duplicate scene(s) detected. "
                f"{batch_info['skipped_count']} duplicate file(s) skipped (SAFE preferred over TIFF preferred over ZIP)."
            )
        if ignored_count > 0:
            warning_messages.append(
                f"{ignored_count} unsupported .SAFE/.zip/.tif/.tiff item(s) ignored."
            )

        if warning_messages:
            batch_popup_warning.value = (
                "Warning: " + " ".join(warning_messages)
            )
            batch_popup_warning.visible = True
        else:
            batch_popup_warning.value = ""
            batch_popup_warning.visible = False

        table_rows = []
        for row in _flatten_batch_records(batch_info):
            status_color = LIQUID_ACCENT if row["status"] == "Used" else "#FFD8A8"
            table_rows.append(
                ft.DataRow(
                    cells=[
                        ft.DataCell(ft.Text(row["scene_id"], size=12, color=LIQUID_TEXT)),
                        ft.DataCell(ft.Text(row["format"], size=12, color=LIQUID_SUBTEXT)),
                        ft.DataCell(ft.Text(row["acquired"], size=12, color=LIQUID_SUBTEXT)),
                        ft.DataCell(ft.Text(row["status"], size=12, color=status_color)),
                        ft.DataCell(
                            ft.Text(
                                _compact_path(row["path"], max_len=84),
                                size=12,
                                color=LIQUID_SUBTEXT,
                                tooltip=row["path"],
                            )
                        ),
                    ]
                )
            )
        batch_popup_table.rows = table_rows
        batch_popup_blocker.visible = True
        _sync_modal_shell_emphasis()
        _refresh_ui_surface(shell, batch_popup_blocker)

    def close_batch_popup(_: ft.ControlEvent) -> None:
        hide_batch_popup()
        _refresh_ui_surface(shell, batch_popup_blocker)

    def hide_validation_popup() -> None:
        validation_popup_blocker.visible = False
        _sync_modal_shell_emphasis()

    def show_validation_popup(result: dict[str, object]) -> None:
        accuracy_raw = result.get("overall_accuracy")
        if isinstance(accuracy_raw, (int, float)):
            validation_popup_accuracy.value = (
                f"Overall accuracy: {float(accuracy_raw):.4f} ({float(accuracy_raw) * 100:.2f}%)"
            )
        else:
            validation_popup_accuracy.value = "Overall accuracy: -"

        labels = [
            str(label_value).strip()
            for label_value in (result.get("confusion_labels") or [])
            if str(label_value).strip()
        ]
        matrix_raw = result.get("confusion_matrix") or []

        table_columns = [
            ft.DataColumn(
                ft.Text("True \\ Pred", color=LIQUID_TEXT, size=12, weight=ft.FontWeight.W_700)
            )
        ]
        table_columns.extend(
            ft.DataColumn(
                ft.Text(label_value, color=LIQUID_TEXT, size=12, weight=ft.FontWeight.W_600)
            )
            for label_value in labels
        )

        table_rows: list[ft.DataRow] = []
        for row_index, row_label in enumerate(labels):
            row_values_raw = (
                matrix_raw[row_index]
                if isinstance(matrix_raw, list)
                and row_index < len(matrix_raw)
                and isinstance(matrix_raw[row_index], list)
                else []
            )
            row_values: list[int] = []
            for col_index in range(len(labels)):
                if col_index < len(row_values_raw):
                    try:
                        row_values.append(int(row_values_raw[col_index]))
                    except Exception:
                        row_values.append(0)
                else:
                    row_values.append(0)

            row_cells = [
                ft.DataCell(
                    ft.Text(row_label, size=12, color=LIQUID_TEXT, weight=ft.FontWeight.W_600)
                )
            ]
            row_cells.extend(
                ft.DataCell(ft.Text(str(cell_value), size=12, color=LIQUID_SUBTEXT))
                for cell_value in row_values
            )
            table_rows.append(ft.DataRow(cells=row_cells))

        if not table_rows:
            table_columns = [
                ft.DataColumn(ft.Text("Confusion Matrix", color=LIQUID_TEXT, size=12))
            ]
            table_rows = [
                ft.DataRow(
                    cells=[
                        ft.DataCell(
                            ft.Text(
                                "No confusion-matrix values available.",
                                size=12,
                                color=LIQUID_SUBTEXT,
                            )
                        )
                    ]
                )
            ]

        validation_popup_table.columns = table_columns
        validation_popup_table.rows = table_rows

        excluded_predictions = result.get("predictions_outside_validation_classes")
        if isinstance(excluded_predictions, int) and excluded_predictions > 0:
            validation_popup_note.value = (
                f"Note: {excluded_predictions} prediction(s) were outside validation classes "
                "and are not shown in this confusion matrix."
            )
            validation_popup_note.visible = True
        else:
            validation_popup_note.value = ""
            validation_popup_note.visible = False

        validation_popup_blocker.visible = True
        _sync_modal_shell_emphasis()
        _refresh_ui_surface(shell, validation_popup_blocker)

    def close_validation_popup(_: ft.ControlEvent) -> None:
        hide_validation_popup()
        _refresh_ui_surface(shell, validation_popup_blocker)

    def _history_map_placeholder(message: str) -> ft.Control:
        return ft.Container(
            expand=True,
            alignment=ft.Alignment(0, 0),
            content=ft.Text(
                message,
                size=12,
                color=LIQUID_SUBTEXT,
                text_align=ft.TextAlign.CENTER,
            ),
        )

    def hide_history_map_popup() -> None:
        history_map_popup_blocker.visible = False
        history_map_popup_external_button.visible = False
        history_map_popup_external_button.on_click = None
        history_map_popup_content_host.content = _history_map_placeholder(
            "Select a run with a stored mask extent to display the map."
        )
        _sync_modal_shell_emphasis()

    def show_history_map_popup(entry: dict[str, object]) -> None:
        extent_coords = _history_mask_extent_coords(entry, resolve_missing=True)
        extent_pending = _history_entry_needs_extent_resolution(entry)
        mask_name = _history_path_name(entry.get("mask_path"))
        history_map_popup_title.value = "Mask Extent Map"
        history_map_popup_summary.value = (
            f"Stored shapefile extent polygon for {mask_name}."
            if mask_name != "-"
            else "Stored shapefile extent polygon for this run."
        )
        history_map_popup_bounds.value = _format_mask_extent_summary(extent_coords)

        if extent_coords:
            native_map_view = _build_history_extent_native_map_view(extent_coords)
            if native_map_view is not None:
                history_map_popup_content_host.content = native_map_view
                history_map_popup_external_button.visible = False
                history_map_popup_external_button.on_click = None
            else:
                map_url = _write_history_extent_map_file(entry, extent_coords)
                if map_url:
                    history_map_popup_content_host.content = _build_history_extent_map_view(map_url)
                    history_map_popup_external_button.visible = True
                    history_map_popup_external_button.on_click = (
                        lambda _, target_url=map_url: _open_external_link(target_url)
                    )
                else:
                    history_map_popup_content_host.content = _history_map_placeholder(
                        "The map file could not be created for this run."
                    )
                    history_map_popup_external_button.visible = False
                    history_map_popup_external_button.on_click = None
        else:
            history_map_popup_summary.value = (
                "Stored shapefile extent coordinates are still being prepared for this run."
                if extent_pending
                else "No stored shapefile extent polygon is available for this run."
            )
            history_map_popup_content_host.content = _history_map_placeholder(
                "Try again in a moment while the stored mask extent is prepared."
                if extent_pending
                else "This run does not contain stored mask extent coordinates."
            )
            history_map_popup_external_button.visible = False
            history_map_popup_external_button.on_click = None

        history_map_popup_blocker.visible = True
        _sync_modal_shell_emphasis()
        _refresh_ui_surface(shell, history_map_popup_blocker)

    def close_history_map_popup(_: ft.ControlEvent | None = None) -> None:
        hide_history_map_popup()
        _refresh_ui_surface(shell, history_map_popup_blocker)

    def _resolve_log_level(message: str, level: str = "info") -> str:
        if level != "info":
            return level
        lowered_message = message.lower()
        if "failed" in lowered_message or "error" in lowered_message:
            return "error"
        if "warning" in lowered_message:
            return "warning"
        return "info"

    def append_log(log_view: ft.ListView, message: str, level: str = "info") -> None:
        accent_palette = {
            "info": LIQUID_ACCENT,
            "warning": "#FFB347",
            "error": "#FF6B6B",
        }
        resolved_level = _resolve_log_level(message, level)
        log_view.controls.append(_log_entry(message, accent_palette.get(resolved_level, LIQUID_ACCENT)))
        if len(log_view.controls) > 120:
            overflow = len(log_view.controls) - 120
            del log_view.controls[:overflow]

    def request_ui_refresh(force: bool = False) -> None:
        """Coalesce frequent UI updates to keep batch runs responsive."""
        min_interval = UI_REFRESH_MIN_INTERVAL_SECONDS

        def _flush() -> None:
            ui_refresh_state["scheduled"] = False
            ui_refresh_state["last"] = time.monotonic()
            _refresh_ui_surface()

        now = time.monotonic()
        elapsed = now - float(ui_refresh_state["last"])
        if force or elapsed >= min_interval:
            _flush()
            return

        if ui_refresh_state["scheduled"]:
            return

        delay = max(0.0, min_interval - elapsed)
        ui_refresh_state["scheduled"] = True
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            ui_refresh_state["scheduled"] = False
            _refresh_ui_surface()
            return

        loop.call_later(delay, _flush)

    def _refresh_ui_surface(*controls: ft.Control) -> None:
        refresh_targets: list[ft.Control] = []
        seen_target_ids: set[int] = set()
        for control in controls:
            if control is None:
                continue
            control_id = id(control)
            if control_id in seen_target_ids:
                continue
            seen_target_ids.add(control_id)
            refresh_targets.append(control)

        if not refresh_targets:
            page.update()
            return

        try:
            for control in refresh_targets:
                control.update()
        except Exception:
            page.update()

    def _sync_modal_shell_emphasis() -> None:
        modal_visible = any(
            bool(getattr(blocker, "visible", False))
            for blocker in (
                about_popup_blocker,
                update_popup_blocker,
                batch_popup_blocker,
                validation_popup_blocker,
                history_map_popup_blocker,
                overlay_blocker,
            )
        )
        modal_visible = modal_visible or bool(menu_state.get("open"))
        shell.opacity = MODAL_BACKGROUND_OPACITY if modal_visible else (0.90 if state["busy"] else 1.0)

    def set_app_status(level: str, message: str) -> None:
        styles = {
            "ready": ("Ready", "#41B883", ft.Colors.with_opacity(0.78, "#E8F8F0")),
            "busy": ("Working", "#E6A700", ft.Colors.with_opacity(0.84, "#FFF4DA")),
            "error": ("Attention", "#D6455D", ft.Colors.with_opacity(0.84, "#FFE6EA")),
            "input": ("Waiting", LIQUID_ACCENT, ft.Colors.with_opacity(0.78, "#EAF2FF")),
        }
        title, accent, fill = styles.get(level, styles["ready"])
        app_status_title.value = title
        app_status.value = message
        app_status_dot.bgcolor = accent
        app_status_card.bgcolor = fill
        app_status_card.border = ft.border.all(1, ft.Colors.with_opacity(0.24, accent))
        app_status_ring.visible = level == "busy"
        app_status_dot.visible = level != "busy"

    def update_overlay(
        title: str | None = None,
        detail: str | None = None,
        progress: float | None = None,
        counter: str | None = None,
        job: str | None = None,
        source: str | None = None,
        destination: str | None = None,
    ) -> None:
        if title is not None:
            overlay_title.value = title
        if detail is not None:
            overlay_detail.value = detail
        if progress is not None:
            bounded_progress = max(0.0, min(1.0, progress))
            overlay_progress.value = bounded_progress
            overlay_percent.value = f"{int(round(bounded_progress * 100))}%"
        if counter is not None:
            overlay_counter.value = counter
        if job is not None:
            overlay_job_value.value = _compact_path(job)
        if source is not None:
            input_target = _resolve_folder_target(source)
            overlay_targets["input"] = str(input_target) if input_target else ""
            overlay_input_path.value = _compact_path(overlay_targets["input"], max_len=124)
            overlay_input_value.opacity = 1.0 if overlay_targets["input"] else 0.55
        if destination is not None:
            output_target = _resolve_folder_target(destination)
            overlay_targets["output"] = str(output_target) if output_target else ""
            overlay_output_path.value = _compact_path(overlay_targets["output"], max_len=124)
            overlay_output_value.opacity = 1.0 if overlay_targets["output"] else 0.55

    def _open_path_target(path_value: str, label: str = "path") -> None:
        cleaned = (path_value or "").strip()
        if not cleaned:
            return

        target_path = Path(cleaned)
        if target_path.exists():
            launch_path = target_path if target_path.is_dir() else target_path.parent
        else:
            launch_path = target_path.parent
        if not launch_path.exists():
            set_app_status("error", f"Could not open {label}: {cleaned}")
            request_ui_refresh(force=True)
            return

        try:
            if os.name != "nt":
                raise OSError("Folder opening is only configured for Windows.")
            os.startfile(str(launch_path))
        except OSError as exc:
            set_app_status("error", f"Could not open {label}: {exc}")
            request_ui_refresh(force=True)

    def open_overlay_target(target_key: str) -> None:
        target_value = overlay_targets.get(target_key, "").strip()
        if not target_value:
            return
        _open_path_target(target_value, label="folder")

    overlay_input_value.on_click = lambda _: open_overlay_target("input")
    overlay_output_value.on_click = lambda _: open_overlay_target("output")
    batch_popup_close_button.on_click = close_batch_popup
    validation_popup_close_button.on_click = close_validation_popup
    history_map_popup_close_button.on_click = close_history_map_popup
    about_popup_close_button.on_click = close_about_popup
    update_popup_close_button.on_click = close_update_popup
    update_popup_later_button.on_click = close_update_popup
    update_popup_install_button.on_click = run_update_install
    menu_about_button.on_click = show_about_popup
    menu_update_button.on_click = trigger_manual_update_check

    def _history_string(value: object) -> str:
        return str(value).strip() if value is not None else ""

    def _history_float(value: object) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value))
        except (TypeError, ValueError):
            return 0.0

    def _format_history_started(started_at_value: object) -> str:
        started_text = _history_string(started_at_value)
        if not started_text:
            return "-"
        try:
            return datetime.fromisoformat(started_text).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return started_text

    def _format_history_duration(duration_value: object) -> str:
        duration_seconds = max(0.0, _history_float(duration_value))
        if duration_seconds < 60:
            return f"{duration_seconds:.1f}s"
        minutes, seconds = divmod(duration_seconds, 60)
        if minutes < 60:
            return f"{int(minutes)}m {seconds:.1f}s"
        hours, rem_minutes = divmod(int(minutes), 60)
        return f"{hours}h {rem_minutes}m"

    location_lookup_cache: dict[str, str] = {}
    mask_extent_lookup_cache: dict[str, list[list[float]]] = {}
    reverse_geocode_state = {"last_request_ts": 0.0}
    history_enrichment_queue: deque[str] = deque()
    history_enrichment_pending: set[str] = set()
    history_enrichment_task = {"value": None}
    history_enrichment_dirty = {"value": False}

    def _history_entry_key(entry: dict[str, object]) -> str:
        return "|".join(
            [
                _history_string(entry.get("started_at")),
                _history_string(entry.get("workflow")),
                _history_string(entry.get("model_path")),
                _history_string(entry.get("output_path")),
            ]
        )

    def _history_path_name(path_value: object) -> str:
        cleaned_path = _history_string(path_value)
        if not cleaned_path:
            return "-"
        path_obj = Path(cleaned_path)
        return path_obj.name.strip() or cleaned_path

    def _history_entry_needs_location_resolution(entry: dict[str, object]) -> bool:
        return (
            _history_string(entry.get("workflow")).lower() == "apply"
            and bool(_history_string(entry.get("mask_path")))
            and _location_name_needs_refinement(_history_string(entry.get("location_name")))
        )

    def _history_entry_needs_extent_resolution(entry: dict[str, object]) -> bool:
        return (
            _history_string(entry.get("workflow")).lower() == "apply"
            and bool(_history_string(entry.get("mask_path")))
            and not _normalise_mask_extent_coords(entry.get("mask_extent_coords"))
        )

    def _find_history_entry(entry_key: str) -> dict[str, object] | None:
        for entry in history_entries:
            if _history_entry_key(entry) == entry_key:
                return entry
        return None

    def _update_history_location_cell(entry_key: str, entry: dict[str, object]) -> bool:
        location_text = history_location_text_lookup.get(entry_key)
        if location_text is None:
            return False
        display_value = _history_location_name(entry, resolve_missing=False)
        location_text.value = display_value
        location_text.tooltip = display_value if display_value != "-" else ""
        return True

    def _normalise_mask_extent_coords(coords_value: object) -> list[list[float]]:
        if not isinstance(coords_value, (list, tuple)):
            return []
        normalized_coords: list[list[float]] = []
        for point_value in coords_value:
            if len(normalized_coords) == 4:
                break
            if not isinstance(point_value, (list, tuple)) or len(point_value) < 2:
                return []
            lat_value = _history_float(point_value[0])
            lon_value = _history_float(point_value[1])
            if not (-90.0 <= lat_value <= 90.0 and -180.0 <= lon_value <= 180.0):
                return []
            normalized_coords.append([round(lat_value, 6), round(lon_value, 6)])
        return normalized_coords if len(normalized_coords) == 4 else []

    def _resolve_mask_extent_coordinates(mask_path_value: object) -> list[list[float]]:
        mask_path = _history_string(mask_path_value)
        if not mask_path:
            return []

        try:
            resolved_path = str(Path(mask_path).expanduser().resolve())
        except Exception:
            resolved_path = mask_path

        cache_key = f"extent::{resolved_path}"
        cached_extent = mask_extent_lookup_cache.get(cache_key)
        if cached_extent is not None:
            return [coord[:] for coord in cached_extent]

        extent_coords: list[list[float]] = []
        try:
            import geopandas as gpd

            gdf = gpd.read_file(mask_path)
            if not gdf.empty and "geometry" in gdf and gdf.crs is not None:
                geometry_series = gdf.geometry.dropna()
                if not geometry_series.empty:
                    gdf = gdf.to_crs(epsg=4326)
                    minx, miny, maxx, maxy = (float(value) for value in gdf.total_bounds)
                    extent_coords = _normalise_mask_extent_coords(
                        [
                            [miny, minx],
                            [maxy, minx],
                            [maxy, maxx],
                            [miny, maxx],
                        ]
                    )
        except Exception:
            extent_coords = []

        mask_extent_lookup_cache[cache_key] = [coord[:] for coord in extent_coords]
        return [coord[:] for coord in extent_coords]

    def _format_mask_extent_summary(extent_coords_value: object) -> str:
        extent_coords = _normalise_mask_extent_coords(extent_coords_value)
        if not extent_coords:
            return "No stored WGS84 shapefile extent is available for this run."
        south_west, north_west, north_east, south_east = extent_coords
        return (
            "Stored WGS84 extent corners. "
            f"SW {south_west[0]:.4f}, {south_west[1]:.4f}; "
            f"NW {north_west[0]:.4f}, {north_west[1]:.4f}; "
            f"NE {north_east[0]:.4f}, {north_east[1]:.4f}; "
            f"SE {south_east[0]:.4f}, {south_east[1]:.4f}."
        )

    def _build_history_extent_map_html(
        entry: dict[str, object],
        extent_coords_value: object,
    ) -> str:
        extent_coords = _normalise_mask_extent_coords(extent_coords_value)
        polygon_coords = extent_coords + [extent_coords[0]]
        run_title = (
            f"{_history_path_name(entry.get('model_path'))} | "
            f"{_format_history_started(entry.get('started_at'))}"
        )
        extent_summary = _format_mask_extent_summary(extent_coords)
        mask_path = _history_string(entry.get("mask_path")) or "Mask path unavailable"
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ICE CREAMS Mask Extent Map</title>
  <link
    rel="stylesheet"
    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
    integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
    crossorigin=""
  >
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      height: 100%;
      background: #eef5ff;
      font-family: "Segoe UI", sans-serif;
    }}
    #map {{
      width: 100%;
      height: 100%;
    }}
    .map-panel {{
      position: absolute;
      top: 14px;
      left: 14px;
      z-index: 999;
      max-width: min(420px, calc(100% - 28px));
      padding: 12px 14px;
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.92);
      box-shadow: 0 10px 28px rgba(15, 37, 61, 0.18);
      color: #0f253d;
      line-height: 1.35;
    }}
    .map-panel h1 {{
      margin: 0 0 6px 0;
      font-size: 15px;
    }}
    .map-panel p {{
      margin: 0;
      font-size: 12px;
    }}
    .leaflet-container {{
      background: #dbe9ff;
    }}
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="map-panel">
    <h1 id="mapRunTitle"></h1>
    <p id="mapExtentSummary"></p>
    <p id="mapMaskPath" style="margin-top: 6px;"></p>
  </div>
  <script
    src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
    integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
    crossorigin=""
  ></script>
  <script>
    const polygonCoords = {json.dumps(polygon_coords)};
    const cornerLabels = ["SW", "NW", "NE", "SE"];
    const runTitle = {json.dumps(run_title)};
    const extentSummary = {json.dumps(extent_summary)};
    const maskPath = {json.dumps(mask_path)};
    document.getElementById("mapRunTitle").textContent = runTitle;
    document.getElementById("mapExtentSummary").textContent = extentSummary;
    document.getElementById("mapMaskPath").textContent = "Mask: " + maskPath;
    const map = L.map("map", {{
      zoomControl: true,
      attributionControl: true
    }});
    L.tileLayer("https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
      maxZoom: 19,
      attribution: "&copy; OpenStreetMap contributors"
    }}).addTo(map);
    const polygon = L.polygon(polygonCoords, {{
      color: "#1f6feb",
      weight: 3,
      fillColor: "#4F8CFF",
      fillOpacity: 0.16
    }}).addTo(map);
    polygon.bindTooltip("Mask extent", {{ sticky: true }});
    polygonCoords.slice(0, 4).forEach((coord, index) => {{
      L.circleMarker(coord, {{
        radius: 5,
        color: "#0f253d",
        weight: 2,
        fillColor: "#ffffff",
        fillOpacity: 1
      }})
        .addTo(map)
        .bindTooltip(cornerLabels[index], {{ permanent: true, direction: "top", offset: [0, -6] }});
    }});
    map.fitBounds(polygon.getBounds(), {{ padding: [24, 24] }});
  </script>
</body>
</html>
"""

    def _write_history_extent_map_file(
        entry: dict[str, object],
        extent_coords_value: object,
    ) -> str:
        extent_coords = _normalise_mask_extent_coords(extent_coords_value)
        if not extent_coords:
            return ""
        history_map_dir = Path(tempfile.gettempdir()) / "ICE_CREAMS_Studio" / "history_maps"
        history_map_dir.mkdir(parents=True, exist_ok=True)
        file_stem_seed = _history_entry_key(entry) or f"history_extent_{int(time.time())}"
        file_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", file_stem_seed).strip("_")[:80]
        if not file_stem:
            file_stem = f"history_extent_{int(time.time())}"
        map_file_path = history_map_dir / f"{file_stem}.html"
        map_file_path.write_text(
            _build_history_extent_map_html(entry, extent_coords),
            encoding="utf-8",
        )
        return map_file_path.as_uri()

    def _build_history_extent_native_map_view(extent_coords_value: object) -> ft.Control | None:
        if ftm is None:
            return None

        extent_coords = _normalise_mask_extent_coords(extent_coords_value)
        if not extent_coords:
            return None

        map_points = [
            ftm.MapLatitudeLongitude(latitude=lat_value, longitude=lon_value)
            for lat_value, lon_value in extent_coords
        ]
        corner_markers = []
        for label, (lat_value, lon_value) in zip(
            ["SW", "NW", "NE", "SE"],
            extent_coords,
        ):
            corner_markers.append(
                ftm.Marker(
                    coordinates=ftm.MapLatitudeLongitude(
                        latitude=lat_value,
                        longitude=lon_value,
                    ),
                    width=44,
                    height=24,
                    content=ft.Container(
                        padding=ft.padding.symmetric(horizontal=6, vertical=3),
                        border_radius=999,
                        bgcolor=LIQUID_TEXT,
                        border=ft.border.all(1, ft.Colors.WHITE),
                        content=ft.Text(
                            label,
                            size=10,
                            weight=ft.FontWeight.W_700,
                            color=ft.Colors.WHITE,
                            text_align=ft.TextAlign.CENTER,
                        ),
                    ),
                )
            )

        return ftm.Map(
            expand=True,
            initial_camera_fit=ftm.CameraFit(
                bounds=ftm.MapLatitudeLongitudeBounds(
                    corner_1=map_points[0],
                    corner_2=map_points[2],
                ),
                padding=36,
                max_zoom=16,
            ),
            layers=[
                ftm.TileLayer(
                    url_template="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
                    user_agent_package_name="ICE_CREAMS_Studio",
                ),
                ftm.PolygonLayer(
                    polygons=[
                        ftm.PolygonMarker(
                            coordinates=map_points,
                            color=ft.Colors.with_opacity(0.18, LIQUID_ACCENT),
                            border_color=LIQUID_ACCENT,
                            border_stroke_width=3,
                        )
                    ]
                ),
                ftm.MarkerLayer(markers=corner_markers),
                ftm.RichAttribution(
                    attributions=[
                        ftm.TextSourceAttribution(
                            text="OpenStreetMap contributors",
                        )
                    ],
                ),
            ],
        )

    def _build_history_extent_map_view(map_url: str) -> ft.Control:
        webview_cls = getattr(ft, "WebView", None)
        if callable(webview_cls):
            for kwargs in (
                {"url": map_url, "expand": True},
                {"src": map_url, "expand": True},
            ):
                try:
                    return webview_cls(**kwargs)
                except TypeError:
                    continue
                except Exception:
                    break
        return ft.Container(
            expand=True,
            alignment=ft.Alignment(0, 0),
            content=ft.Column(
                spacing=10,
                tight=True,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    ft.Icon(ft.Icons.MAP, size=34, color=LIQUID_ACCENT),
                    ft.Text(
                        "Embedded map preview is not available in this environment.",
                        size=13,
                        color=LIQUID_TEXT,
                        text_align=ft.TextAlign.CENTER,
                    ),
                    ft.Text(
                        "Use 'Open in browser' to view the Leaflet map for this run.",
                        size=12,
                        color=LIQUID_SUBTEXT,
                        text_align=ft.TextAlign.CENTER,
                    ),
                ],
            ),
        )

    def _format_location_coordinates(lat: float, lon: float) -> str:
        return f"{lat:.4f}, {lon:.4f}"

    def _dedupe_location_parts(parts: list[object]) -> list[str]:
        unique_parts: list[str] = []
        seen_parts: set[str] = set()
        for part in parts:
            cleaned_part = str(part).strip()
            if not cleaned_part:
                continue
            normalized_part = cleaned_part.casefold()
            if normalized_part in seen_parts:
                continue
            seen_parts.add(normalized_part)
            unique_parts.append(cleaned_part)
        return unique_parts

    def _build_precise_location_name(address: dict[str, object], lat: float, lon: float) -> str:
        micro_place = (
            address.get("neighbourhood")
            or address.get("suburb")
            or address.get("borough")
            or address.get("quarter")
            or address.get("island")
            or address.get("archipelago")
            or address.get("hamlet")
        )
        locality = (
            address.get("village")
            or address.get("town")
            or address.get("city")
            or address.get("municipality")
            or address.get("locality")
            or address.get("county")
        )
        marine_area = (
            address.get("sea")
            or address.get("ocean")
            or address.get("waterway")
            or address.get("bay")
        )
        region = (
            address.get("county")
            or address.get("state_district")
            or address.get("state")
            or address.get("region")
        )
        country = address.get("country")
        location_parts = _dedupe_location_parts(
            [micro_place, locality, marine_area, region, country]
        )
        country_text = str(country).strip()
        if country_text:
            non_country_parts = [
                part for part in location_parts if part.casefold() != country_text.casefold()
            ]
            location_parts = non_country_parts[:3] + [country_text]
        else:
            location_parts = location_parts[:4]

        if location_parts:
            if len(location_parts) == 1 and country_text:
                return f"{country_text} ({_format_location_coordinates(lat, lon)})"
            return ", ".join(location_parts)
        return _format_location_coordinates(lat, lon)

    def _location_name_needs_refinement(location_name: str) -> bool:
        cleaned_location = location_name.strip()
        if not cleaned_location or cleaned_location == "-":
            return True
        return "," not in cleaned_location and "(" not in cleaned_location

    def _reverse_geocode_location(lat: float, lon: float) -> str:
        coordinate_key = f"{round(lat, 4)},{round(lon, 4)}"
        cached_location = location_lookup_cache.get(coordinate_key)
        if cached_location is not None:
            return cached_location

        now = time.monotonic()
        elapsed = now - float(reverse_geocode_state["last_request_ts"])
        if elapsed < 1.0:
            time.sleep(max(0.0, 1.0 - elapsed))

        query = urlencode(
            {
                "lat": f"{lat:.7f}",
                "lon": f"{lon:.7f}",
                "format": "jsonv2",
                "zoom": "13",
                "addressdetails": "1",
            }
        )
        request = Request(
            f"https://nominatim.openstreetmap.org/reverse?{query}",
            headers={"User-Agent": "ICE_CREAMS_Studio/1.0"},
        )
        resolved_location = ""
        try:
            with urlopen(request, timeout=8) as response:
                payload = json.loads(response.read().decode("utf-8"))
            address = payload.get("address", {}) if isinstance(payload, dict) else {}
            if isinstance(address, dict):
                resolved_location = _build_precise_location_name(address, lat, lon)
            if not resolved_location and isinstance(payload, dict):
                display_name = str(payload.get("display_name", "")).strip()
                if display_name:
                    resolved_location = display_name
        except Exception:
            resolved_location = ""
        reverse_geocode_state["last_request_ts"] = time.monotonic()
        location_lookup_cache[coordinate_key] = resolved_location
        return resolved_location

    def _resolve_location_from_mask(mask_path_value: object) -> str:
        mask_path = _history_string(mask_path_value)
        if not mask_path:
            return ""

        try:
            resolved_path = str(Path(mask_path).expanduser().resolve())
        except Exception:
            resolved_path = mask_path

        cache_key = f"mask::{resolved_path}"
        cached_location = location_lookup_cache.get(cache_key)
        if cached_location is not None:
            return cached_location

        location_name = ""
        try:
            import geopandas as gpd

            gdf = gpd.read_file(mask_path)
            if not gdf.empty and "geometry" in gdf:
                geometry_series = gdf.geometry.dropna()
                if not geometry_series.empty:
                    if gdf.crs is not None:
                        gdf = gdf.to_crs(epsg=4326)
                        geometry_series = gdf.geometry.dropna()
                    union_geometry = geometry_series.unary_union
                    if union_geometry is not None and not union_geometry.is_empty:
                        representative_point = union_geometry.representative_point()
                        lon_value = float(representative_point.x)
                        lat_value = float(representative_point.y)
                        if -180 <= lon_value <= 180 and -90 <= lat_value <= 90:
                            location_name = _reverse_geocode_location(lat_value, lon_value)
        except Exception:
            location_name = ""

        location_lookup_cache[cache_key] = location_name
        return location_name

    def _resolve_history_entry_metadata(
        mask_path_value: object,
        resolve_location: bool,
        resolve_extent: bool,
    ) -> tuple[str, list[list[float]]]:
        location_name = _resolve_location_from_mask(mask_path_value) if resolve_location else ""
        extent_coords = _resolve_mask_extent_coordinates(mask_path_value) if resolve_extent else []
        return location_name, extent_coords

    async def _run_history_enrichment_worker() -> None:
        try:
            while history_enrichment_queue:
                entry_key = history_enrichment_queue.popleft()
                history_enrichment_pending.discard(entry_key)
                entry = _find_history_entry(entry_key)
                if entry is None:
                    continue

                needs_location = _history_entry_needs_location_resolution(entry)
                needs_extent = _history_entry_needs_extent_resolution(entry)
                mask_path = _history_string(entry.get("mask_path"))
                if not mask_path or not (needs_location or needs_extent):
                    continue

                try:
                    location_name, extent_coords = await asyncio.to_thread(
                        _resolve_history_entry_metadata,
                        mask_path,
                        needs_location,
                        needs_extent,
                    )
                except Exception:
                    continue

                entry_updated = False
                ui_updated = False
                if needs_location and location_name and location_name != _history_string(entry.get("location_name")):
                    entry["location_name"] = location_name
                    entry_updated = True
                    ui_updated = _update_history_location_cell(entry_key, entry) or ui_updated
                if needs_extent and extent_coords:
                    normalised_extent = _normalise_mask_extent_coords(entry.get("mask_extent_coords"))
                    if extent_coords != normalised_extent:
                        entry["mask_extent_coords"] = extent_coords
                        entry_updated = True
                if not entry_updated:
                    continue

                history_enrichment_dirty["value"] = True
                if history_selected_key["value"] == entry_key:
                    _populate_history_details(entry)
                    ui_updated = True
                if ui_updated:
                    request_ui_refresh()
        finally:
            dirty = bool(history_enrichment_dirty["value"])
            history_enrichment_dirty["value"] = False
            history_enrichment_task["value"] = None
            if dirty:
                _rewrite_history_records_to_disk()
            if history_enrichment_queue:
                history_enrichment_task["value"] = page.run_task(_run_history_enrichment_worker)

    def _queue_history_enrichment(
        entries: list[dict[str, object]] | tuple[dict[str, object], ...],
        *,
        prioritize: bool = False,
    ) -> None:
        queued = False
        for entry in entries:
            entry_key = _history_entry_key(entry)
            if not entry_key or entry_key in history_enrichment_pending:
                continue
            if not (
                _history_entry_needs_location_resolution(entry)
                or _history_entry_needs_extent_resolution(entry)
            ):
                continue
            if prioritize:
                history_enrichment_queue.appendleft(entry_key)
            else:
                history_enrichment_queue.append(entry_key)
            history_enrichment_pending.add(entry_key)
            queued = True

        if not queued:
            return

        current_task = history_enrichment_task["value"]
        if current_task is None or bool(getattr(current_task, "done", lambda: True)()):
            history_enrichment_task["value"] = page.run_task(_run_history_enrichment_worker)

    def _history_location_name(entry: dict[str, object], resolve_missing: bool = False) -> str:
        location_name = _history_string(entry.get("location_name"))
        workflow_name = _history_string(entry.get("workflow")).lower()
        if workflow_name != "apply":
            return location_name or "-"
        if resolve_missing and _history_entry_needs_location_resolution(entry):
            _queue_history_enrichment((entry,), prioritize=True)
        if location_name:
            return location_name
        return "-"

    def _history_mask_extent_coords(
        entry: dict[str, object],
        resolve_missing: bool = False,
    ) -> list[list[float]]:
        extent_coords = _normalise_mask_extent_coords(entry.get("mask_extent_coords"))
        workflow_name = _history_string(entry.get("workflow")).lower()
        if workflow_name != "apply":
            return extent_coords
        if not extent_coords and resolve_missing and _history_entry_needs_extent_resolution(entry):
            _queue_history_enrichment((entry,), prioritize=True)
        return extent_coords

    def _set_history_detail_button(
        button: ft.ElevatedButton,
        path_key: str,
        path_value: object,
        label: str,
    ) -> None:
        cleaned_path = _history_string(path_value)
        history_selected_paths[path_key] = cleaned_path
        button.visible = bool(cleaned_path)
        button.text = "Open"
        button.tooltip = cleaned_path if cleaned_path else ""
        if cleaned_path:
            button.on_click = (
                lambda _, selected_path=cleaned_path, selected_label=label: _open_path_target(
                    selected_path,
                    selected_label,
                )
            )

    def _populate_history_details(entry: dict[str, object] | None) -> None:
        if entry is None:
            history_detail_heading.value = "Select a run"
            history_detail_subheading.value = "Choose a row on the left to display full run details."
            history_detail_status_value.value = "Status: -"
            history_detail_status_value.color = LIQUID_TEXT
            history_detail_status_badge.bgcolor = ft.Colors.with_opacity(0.18, LIQUID_ACCENT)
            history_detail_status_badge.border = ft.border.all(1, ft.Colors.with_opacity(0.25, LIQUID_ACCENT))
            history_detail_date_value.value = "-"
            history_detail_workflow_value.value = "-"
            history_detail_duration_value.value = "-"
            history_detail_extent_note_value.value = "-"
            history_detail_error_value.value = "-"
            history_detail_notes_value.value = "-"
            history_show_map_button.disabled = True
            history_show_map_button.on_click = None
            history_show_map_button.tooltip = "Select an apply run with a stored mask extent."
            for key_name in history_selected_paths:
                history_selected_paths[key_name] = ""
            for button in (
                history_open_input_button,
                history_open_model_button,
                history_open_output_button,
                history_open_mask_button,
            ):
                button.visible = False
            return

        workflow_name = _history_string(entry.get("workflow")).title() or "Unknown"
        model_name = _history_path_name(entry.get("model_path"))
        location_name = _history_location_name(entry, resolve_missing=True)
        extent_coords = _history_mask_extent_coords(entry, resolve_missing=True)
        location_pending = _history_entry_needs_location_resolution(entry)
        extent_pending = _history_entry_needs_extent_resolution(entry)
        status_text = _history_string(entry.get("status")).capitalize() or "-"
        status_color_map = {
            "success": "#41B883",
            "failed": "#D6455D",
            "partial": "#E6A700",
            "skipped": "#7D8EA3",
        }
        status_key = _history_string(entry.get("status")).lower()
        status_color = status_color_map.get(status_key, LIQUID_ACCENT)
        detail_location = location_name
        if detail_location == "-" and location_pending:
            detail_location = "Resolving location..."
        history_detail_heading.value = f"{model_name}"
        history_detail_subheading.value = (
            f"{workflow_name} run in {detail_location}"
            if detail_location and detail_location != "-"
            else f"{workflow_name} run"
        )
        history_detail_status_value.value = f"Status: {status_text}"
        history_detail_status_value.color = status_color
        history_detail_status_badge.bgcolor = ft.Colors.with_opacity(0.16, status_color)
        history_detail_status_badge.border = ft.border.all(1, ft.Colors.with_opacity(0.30, status_color))
        history_detail_date_value.value = _format_history_started(entry.get("started_at"))
        history_detail_workflow_value.value = workflow_name
        history_detail_duration_value.value = _format_history_duration(entry.get("duration_seconds"))
        history_detail_error_value.value = _history_string(entry.get("error")) or "-"
        history_detail_notes_value.value = _history_string(entry.get("details")) or "-"
        if extent_coords:
            history_detail_extent_note_value.value = _format_mask_extent_summary(extent_coords)
            history_show_map_button.disabled = False
            history_show_map_button.tooltip = "Open a Leaflet map of the stored mask extent."
            history_show_map_button.on_click = (
                lambda _, selected_entry=entry: show_history_map_popup(selected_entry)
            )
        else:
            history_detail_extent_note_value.value = (
                "Preparing stored shapefile extent coordinates for this run."
                if extent_pending
                else "No stored shapefile extent coordinates are available for this run."
                if _history_string(entry.get("workflow")).lower() == "apply"
                else "Extent maps are only available for apply runs that used a mask shapefile."
            )
            history_show_map_button.disabled = True
            history_show_map_button.tooltip = (
                "The stored mask extent is still being prepared."
                if extent_pending
                else "No mask extent map is available for this run."
            )
            history_show_map_button.on_click = None
        _set_history_detail_button(history_open_input_button, "input", entry.get("input_path"), "Input")
        _set_history_detail_button(history_open_model_button, "model", entry.get("model_path"), "Model")
        _set_history_detail_button(history_open_output_button, "output", entry.get("output_path"), "Output")
        _set_history_detail_button(history_open_mask_button, "mask", entry.get("mask_path"), "Mask")

    def _on_history_row_select(entry_key: str, is_selected: bool) -> None:
        previous_key = history_selected_key["value"]
        history_selected_key["value"] = entry_key if is_selected else ""
        if previous_key and previous_key != history_selected_key["value"]:
            previous_row = history_row_lookup.get(previous_key)
            if previous_row is not None:
                previous_row.selected = False
        current_row = history_row_lookup.get(entry_key)
        if current_row is not None:
            current_row.selected = is_selected
        selected_entry = None
        if history_selected_key["value"]:
            for history_entry in history_visible_entries:
                if _history_entry_key(history_entry) == history_selected_key["value"]:
                    selected_entry = history_entry
                    break
        _populate_history_details(selected_entry)
        request_ui_refresh(force=True)

    def refresh_history_view(force_reload: bool = False, refresh_ui: bool = True) -> None:
        if force_reload:
            history_entries.clear()
            history_entries.extend(_load_history_records_from_disk(history_log_path))

        sorted_entries = sorted(
            history_entries,
            key=lambda item: _history_string(item.get("started_at")),
            reverse=True,
        )
        history_visible_entries.clear()
        history_visible_entries.extend(sorted_entries[:300])
        history_row_lookup.clear()
        history_location_text_lookup.clear()
        rows: list[ft.DataRow] = []
        current_selection_found = False
        for entry in history_visible_entries:
            entry_key = _history_entry_key(entry)
            selected = entry_key == history_selected_key["value"]
            if selected:
                current_selection_found = True
            model_name = _history_path_name(entry.get("model_path"))
            location_name = _history_location_name(entry, resolve_missing=False)
            status_key = _history_string(entry.get("status")).lower()
            status_label_map = {
                "success": "Succeeded",
                "failed": "Failed",
                "partial": "Partial",
                "skipped": "Skipped",
            }
            status_color_map = {
                "success": "#41B883",
                "failed": "#D6455D",
                "partial": "#E6A700",
                "skipped": "#7D8EA3",
            }
            status_label = status_label_map.get(status_key, status_key.capitalize() or "-")
            status_color = status_color_map.get(status_key, LIQUID_TEXT)
            location_text = ft.Text(
                location_name,
                size=11,
                color=LIQUID_TEXT,
                max_lines=1,
                overflow=ft.TextOverflow.ELLIPSIS,
                tooltip=location_name if location_name != "-" else "",
            )
            history_location_text_lookup[entry_key] = location_text
            row = ft.DataRow(
                selected=selected,
                on_select_change=lambda event, selected_key=entry_key: _on_history_row_select(
                    selected_key,
                    str(event.data).lower() == "true" or bool(getattr(event.control, "selected", False)),
                ),
                cells=[
                    ft.DataCell(ft.Text(_format_history_started(entry.get("started_at")), size=11, color=LIQUID_TEXT)),
                    ft.DataCell(ft.Text(model_name, size=11, color=LIQUID_TEXT)),
                    ft.DataCell(location_text),
                    ft.DataCell(ft.Text(status_label, size=11, color=status_color, weight=ft.FontWeight.W_600)),
                ]
            )
            history_row_lookup[entry_key] = row
            rows.append(row)

        if rows:
            history_table.rows = rows
            history_status.value = f"Showing {len(rows)} most recent run(s)."
        else:
            history_table.rows = [
                ft.DataRow(
                    cells=[
                        ft.DataCell(
                            ft.Text(
                                "No run history available yet.",
                                size=12,
                                color=LIQUID_SUBTEXT,
                            )
                        )
                    ]
                    + [ft.DataCell(ft.Text("", size=11, color=LIQUID_SUBTEXT)) for _ in range(3)]
                )
            ]
            history_status.value = "No run history available yet."

        if not current_selection_found:
            history_selected_key["value"] = ""
            _populate_history_details(None)

        history_summary.value = f"{len(history_entries)} total run(s) recorded."
        _queue_history_enrichment(tuple(history_visible_entries[:24]))
        if refresh_ui:
            request_ui_refresh(force=True)

    def record_run_history(
        workflow: str,
        status: str,
        started_at: datetime,
        duration_seconds: float,
        input_path: str = "",
        model_path: str = "",
        output_path: str = "",
        mask_path: str = "",
        error_message: str = "",
        details: str = "",
        location_name: str = "",
        mask_extent_coords: list[list[float]] | None = None,
    ) -> None:
        history_record: dict[str, object] = {
            "started_at": started_at.isoformat(timespec="seconds"),
            "workflow": workflow.strip().lower(),
            "status": status.strip().lower(),
            "duration_seconds": round(max(0.0, float(duration_seconds)), 3),
            "input_path": input_path.strip(),
            "model_path": model_path.strip(),
            "output_path": output_path.strip(),
            "mask_path": mask_path.strip(),
            "error": error_message.strip(),
            "details": details.strip(),
            "location_name": location_name.strip(),
            "mask_extent_coords": _normalise_mask_extent_coords(mask_extent_coords),
        }
        history_entries.append(history_record)
        overflow = len(history_entries) - history_max_records
        if overflow > 0:
            del history_entries[:overflow]

        try:
            history_log_path.parent.mkdir(parents=True, exist_ok=True)
            with history_log_path.open("a", encoding="utf-8") as log_stream:
                log_stream.write(json.dumps(history_record, ensure_ascii=True))
                log_stream.write("\n")
        except OSError as exc:
            history_status.value = f"Could not write history file: {exc}"

        refresh_history_view(refresh_ui=False)

    def on_history_refresh(_: ft.ControlEvent) -> None:
        refresh_history_view(force_reload=True)

    def set_global_busy(is_busy: bool, status_text: str | None = None) -> None:
        state["busy"] = is_busy
        if status_text:
            set_app_status("busy" if is_busy else "ready", status_text)
        overlay_blocker.visible = is_busy
        _sync_modal_shell_emphasis()
        if not is_busy:
            state["operation_mode"] = None
            update_overlay(
                title="Working",
                detail="Preparing the workflow.",
                progress=0,
                counter="",
                job="-",
                source="",
                destination="",
            )
        for control in interactive_controls:
            try:
                if hasattr(control, "disabled"):
                    control.disabled = is_busy
            except Exception:
                continue
        if not is_busy:
            try:
                sync_training_model_controls(refresh=False)
            except NameError:
                pass
            try:
                sync_validation_mode_controls(refresh=False)
            except NameError:
                pass
        try:
            refresh_apply_run_button_state()
        except NameError:
            pass
        _refresh_ui_surface(
            shell,
            side_menu_overlay,
            side_menu_backdrop,
            menu_toggle_badge,
            menu_about_button,
            menu_update_button,
            overlay_blocker,
            about_popup_blocker,
            update_popup_blocker,
            batch_popup_blocker,
            validation_popup_blocker,
            history_map_popup_blocker,
        )

    def push_apply_status(message: str, level: str = "info", refresh: bool = True) -> None:
        apply_status.value = message
        append_log(apply_log, message, level)
        if state["operation_mode"] == "apply":
            update_overlay(
                title="Applying ICE CREAMS",
                detail=message,
            )
        if refresh:
            request_ui_refresh()

    def push_apply_progress(value: float, refresh: bool = True) -> None:
        apply_progress.value = max(0.0, min(1.0, value))
        if state["operation_mode"] == "apply":
            update_overlay(progress=apply_progress.value)
        if refresh:
            request_ui_refresh()

    def push_train_status(message: str, level: str = "info", refresh: bool = True) -> None:
        train_status.value = message
        append_log(train_log, message, level)
        if state["operation_mode"] == "train":
            update_overlay(
                title="Training Model",
                detail=message,
            )
        if refresh:
            request_ui_refresh()

    def push_train_progress(value: float, refresh: bool = True) -> None:
        train_progress.value = max(0.0, min(1.0, value))
        if state["operation_mode"] == "train":
            update_overlay(progress=train_progress.value)
        if refresh:
            request_ui_refresh()

    def push_validation_status(message: str, level: str = "info", refresh: bool = True) -> None:
        validation_status.value = message
        append_log(validation_log, message, level)
        if state["operation_mode"] == "validation":
            update_overlay(
                title="Validating Model",
                detail=message,
            )
        if refresh:
            request_ui_refresh()

    def push_validation_progress(value: float, refresh: bool = True) -> None:
        validation_progress.value = max(0.0, min(1.0, value))
        if state["operation_mode"] == "validation":
            update_overlay(progress=validation_progress.value)
        if refresh:
            request_ui_refresh()

    def report_idle(message: str) -> None:
        set_app_status("ready", message)

    def show_error(target: str, message: str) -> None:
        if target == "apply":
            push_apply_status(message, level="warning")
        elif target == "train":
            push_train_status(message, level="warning")
        else:
            push_validation_status(message, level="warning")
        set_app_status("input", "Ready for a new run.")
        request_ui_refresh(force=True)

    async def _discover_scene_batch_info_async(
        input_source: str,
        status_text: str,
    ) -> dict[str, object]:
        """Run scene discovery off the UI thread while keeping the user informed."""
        set_global_busy(True, status_text)
        try:
            return await asyncio.to_thread(discover_scene_batch_info, input_source)
        finally:
            set_global_busy(False)

    async def choose_single_scene(_: ft.ControlEvent) -> None:
        if state["busy"]:
            return
        files = await ft.FilePicker().pick_files(
            dialog_title="Select a single .zip scene or 12-band .tif/.tiff image",
            initial_directory=_resolve_initial_directory(apply_safe_field.value),
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["zip", "tif", "tiff"],
            allow_multiple=False,
        )
        if not files:
            return

        selected_path = Path(files[0].path)
        if selected_path.suffix.lower() not in {".zip", ".tif", ".tiff"}:
            show_error("apply", "Please select a .zip, .tif, or .tiff input file for this action.")
            return

        try:
            await _discover_scene_batch_info_async(
                str(selected_path),
                "Validating the selected input file.",
            )
        except Exception as exc:  # noqa: BLE001 - surface discovery errors.
            show_error("apply", f"Selected input file is not a valid apply input: {exc}")
            return

        apply_safe_field.value = str(selected_path)
        if not apply_output_path_field.value.strip():
            apply_output_path_field.value = suggest_apply_output_path()
        refresh_apply_preview()
        refresh_apply_run_button_state()
        hide_batch_popup()
        if selected_path.suffix.lower() == ".zip":
            push_apply_status("Single .zip scene selected.")
        else:
            push_apply_status("Single TIFF input selected.")
        schedule_apply_preflight_scan("Checking already processed outputs for the selected input file.")

    async def choose_scene_folder(_: ft.ControlEvent) -> None:
        if state["busy"]:
            return
        selected = await ft.FilePicker().get_directory_path(
            dialog_title="Select a single .SAFE folder or a batch folder containing .SAFE/.zip/.tif inputs",
            initial_directory=_resolve_initial_directory(apply_safe_field.value),
        )
        if selected:
            selected_path = Path(selected)
            if selected_path.name.upper().endswith(".SAFE"):
                try:
                    await _discover_scene_batch_info_async(
                        selected,
                        "Validating the selected .SAFE folder.",
                    )
                except Exception as exc:  # noqa: BLE001 - surface discovery errors.
                    show_error("apply", f"Selected folder is not a valid Sentinel-2 .SAFE scene: {exc}")
                    return
                apply_safe_field.value = selected
                if not apply_output_path_field.value.strip():
                    apply_output_path_field.value = suggest_apply_output_path()
                refresh_apply_preview()
                refresh_apply_run_button_state()
                hide_batch_popup()
                push_apply_status("Single .SAFE folder selected.")
                schedule_apply_preflight_scan("Checking already processed outputs for the selected SAFE scene.")
                return

            try:
                batch_info = await _discover_scene_batch_info_async(
                    selected,
                    "Scanning the selected batch folder for apply inputs.",
                )
            except Exception as exc:
                show_error("apply", f"Could not discover batch inputs: {exc}")
                return

            apply_safe_field.value = selected
            if not apply_output_path_field.value.strip():
                apply_output_path_field.value = suggest_apply_output_path()
            refresh_apply_preview()
            refresh_apply_run_button_state()
            push_apply_status("Scene batch folder selected.")
            ignored_count = int(batch_info.get("ignored_count", 0))
            if ignored_count > 0:
                push_apply_status(
                    f"Warning: {ignored_count} unsupported .SAFE/.zip/.tif/.tiff item(s) were ignored.",
                    level="warning",
                )
            show_batch_popup(batch_info)
            schedule_apply_preflight_scan("Checking already processed outputs for the selected batch folder.")

    async def choose_mask_file(_: ft.ControlEvent) -> None:
        if state["busy"]:
            return
        files = await ft.FilePicker().pick_files(
            dialog_title="Select a shapefile mask",
            initial_directory=_resolve_initial_directory(apply_mask_field.value),
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["shp"],
            allow_multiple=False,
        )
        if files:
            apply_mask_field.value = files[0].path
            refresh_apply_run_button_state()
            if _apply_can_run():
                set_app_status("ready", "Ready for a new run.")
            push_apply_status("Mask shapefile selected.")

    async def choose_output_folder(_: ft.ControlEvent) -> None:
        if state["busy"]:
            return
        selected = await ft.FilePicker().get_directory_path(
            dialog_title="Select an output folder",
            initial_directory=_resolve_initial_directory(
                apply_output_path_field.value or str(default_apply_output_dir)
            ),
        )
        if selected:
            apply_output_path_field.value = selected
            refresh_apply_preview()
            refresh_apply_run_button_state()
            push_apply_status("Output folder selected.")
            schedule_apply_preflight_scan("Checking already processed outputs in the selected output folder.")

    def on_apply_model_select(_: ft.ControlEvent) -> None:
        selected_model = (apply_model_dropdown.value or "").strip()
        if not selected_model:
            push_apply_status("No model selected.", level="warning")
            refresh_apply_preview()
            refresh_apply_run_button_state()
            return
        refresh_apply_preview()
        refresh_apply_run_button_state()
        push_apply_status(f"Model selected: {Path(selected_model).name}")
        schedule_apply_preflight_scan("Checking already processed outputs for the selected model.")

    async def choose_training_dataset(_: ft.ControlEvent) -> None:
        if state["busy"]:
            return
        files = await ft.FilePicker().pick_files(
            dialog_title="Select one or more training CSV files",
            initial_directory=_training_picker_initial_directory(),
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["csv"],
            allow_multiple=True,
        )
        if not files:
            return

        new_csvs = sorted(
            {
                str(Path(file_item.path).resolve())
                for file_item in files
                if file_item.path and file_item.path.lower().endswith(".csv")
            }
        )
        if not new_csvs:
            show_error("train", "No valid CSV file was selected.")
            return

        added_count = 0
        for csv_path in new_csvs:
            if csv_path not in selected_training_csv_files:
                selected_training_csv_files.append(csv_path)
                added_count += 1

        _rebuild_training_csv_selection()
        _update_training_source_field()
        push_train_status(
            f"Added {added_count} CSV file(s). "
            f"Dataset now contains {len(selected_training_csvs)} CSV file(s) from "
            f"{len(selected_training_csv_files)} file selection(s) and "
            f"{len(selected_training_csv_folders)} folder selection(s)."
        )

    async def choose_training_dataset_folder(_: ft.ControlEvent) -> None:
        if state["busy"]:
            return
        selected = await ft.FilePicker().get_directory_path(
            dialog_title="Select a folder containing training CSV files",
            initial_directory=_training_picker_initial_directory(),
        )
        if not selected:
            return

        selected_folder = str(Path(selected).resolve())
        if selected_folder in selected_training_csv_folders:
            push_train_status("Folder already selected for training dataset.", level="warning")
            return

        selected_training_csv_folders.append(selected_folder)
        folder_csvs = _discover_training_folder_csvs(selected_folder)
        _rebuild_training_csv_selection()
        _update_training_source_field()

        if folder_csvs:
            push_train_status(
                f"Added folder with {len(folder_csvs)} CSV file(s). "
                f"Dataset now contains {len(selected_training_csvs)} CSV file(s) from "
                f"{len(selected_training_csv_files)} file selection(s) and "
                f"{len(selected_training_csv_folders)} folder selection(s)."
            )
        else:
            push_train_status(
                "Added folder, but no CSV files were found inside it.",
                level="warning",
            )

    def clear_training_dataset_selection(_: ft.ControlEvent) -> None:
        if state["busy"]:
            return
        selected_training_csv_files.clear()
        selected_training_csv_folders.clear()
        selected_training_csvs.clear()
        _update_training_source_field()
        push_train_status("Cleared training dataset selection.")

    async def choose_training_output_dir(_: ft.ControlEvent) -> None:
        if state["busy"]:
            return
        selected = await ft.FilePicker().get_directory_path(
            dialog_title="Select a model output folder",
            initial_directory=_resolve_initial_directory(training_output_dir_field.value),
        )
        if selected:
            training_output_dir_field.value = selected
            push_train_status("Training output directory selected.")

    async def choose_validation_dataset(_: ft.ControlEvent) -> None:
        if state["busy"]:
            return
        initial_dir = _resolve_initial_directory(validation_dataset_field.value)
        if not validation_dataset_field.value.strip() and default_validation_source.exists():
            initial_dir = str(default_validation_source)
        files = await ft.FilePicker().pick_files(
            dialog_title="Select a validation dataset (.csv or .xlsx)",
            initial_directory=initial_dir,
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["csv", "xlsx"],
            allow_multiple=False,
        )
        if not files:
            return

        selected_path = Path(files[0].path)
        if selected_path.suffix.lower() not in {".csv", ".xlsx"}:
            show_error("validation", "Please select a .csv or .xlsx validation dataset.")
            return

        validation_dataset_field.value = str(selected_path)
        refresh_validation_preview()
        push_validation_status(f"Validation dataset selected: {selected_path.name}")

    async def choose_validation_model_file(_: ft.ControlEvent) -> None:
        if state["busy"]:
            return
        files = await ft.FilePicker().pick_files(
            dialog_title="Select a trained model (.pkl)",
            initial_directory=_resolve_initial_directory(
                validation_external_model_field.value
                or validation_model_dropdown.value
                or str(default_models_dir)
            ),
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=["pkl"],
            allow_multiple=False,
        )
        if not files:
            return

        selected_path = Path(files[0].path)
        if selected_path.suffix.lower() != ".pkl":
            show_error("validation", "Please select a .pkl model file.")
            return

        validation_external_model_field.value = str(selected_path)
        refresh_validation_preview()
        push_validation_status(f"External model selected: {selected_path.name}")

    async def choose_validation_output_dir(_: ft.ControlEvent) -> None:
        if state["busy"]:
            return
        selected = await ft.FilePicker().get_directory_path(
            dialog_title="Select validation output folder",
            initial_directory=_resolve_initial_directory(
                validation_output_dir_field.value or str(default_validation_output_dir)
            ),
        )
        if selected:
            validation_output_dir_field.value = selected
            refresh_validation_preview()
            push_validation_status("Validation output directory selected.")

    def clear_validation_external_model(_: ft.ControlEvent) -> None:
        if state["busy"]:
            return
        if not validation_external_model_field.value.strip():
            push_validation_status("External model path is already empty.", level="warning")
            return
        validation_external_model_field.value = ""
        refresh_validation_preview()
        push_validation_status("Cleared external model selection. Using dropdown model.")

    def on_validation_model_select(_: ft.ControlEvent) -> None:
        selected_model = (validation_model_dropdown.value or "").strip()
        if not selected_model:
            push_validation_status("No model selected.", level="warning")
            refresh_validation_preview()
            return
        refresh_validation_preview()
        push_validation_status(f"Dropdown model selected: {Path(selected_model).name}")

    def on_validation_label_column_change(_: ft.ControlEvent) -> None:
        label_name = (validation_label_column_field.value or "").strip()
        if label_name:
            push_validation_status(f"Validation label column set to '{label_name}'.")
        else:
            push_validation_status("Validation label column is empty.", level="warning")

    def sync_validation_mode_controls(refresh: bool = True) -> None:
        selected_mode = (validation_mode_dropdown.value or VALIDATION_MODE_MULTICLASS).strip()
        is_presence_absence = selected_mode == VALIDATION_MODE_PRESENCE_ABSENCE
        validation_target_class_container.visible = is_presence_absence
        validation_target_class_field.disabled = not is_presence_absence
        if not validation_target_class_field.value.strip():
            validation_target_class_field.value = DEFAULT_TARGET_CLASS
        if refresh:
            request_ui_refresh()

    def on_validation_mode_select(_: ft.ControlEvent) -> None:
        sync_validation_mode_controls(refresh=False)
        selected_mode = (validation_mode_dropdown.value or VALIDATION_MODE_MULTICLASS).strip()
        if selected_mode == VALIDATION_MODE_PRESENCE_ABSENCE:
            target_label = (validation_target_class_field.value or "").strip() or DEFAULT_TARGET_CLASS
            push_validation_status(
                f"Validation mode set to presence/absence using '{target_label}' as Presence."
            )
        else:
            push_validation_status("Validation mode set to model classes (as-is).")
        request_ui_refresh()

    def on_validation_target_class_change(_: ft.ControlEvent) -> None:
        target_label = (validation_target_class_field.value or "").strip()
        if not target_label:
            validation_target_class_field.value = DEFAULT_TARGET_CLASS
            target_label = DEFAULT_TARGET_CLASS
        push_validation_status(f"Presence/absence target class set to '{target_label}'.")

    def resolve_training_feature_mode() -> tuple[str, str] | None:
        try:
            resolved_mode = normalize_feature_mode(training_mode_dropdown.value or DEFAULT_FEATURE_MODE)
        except ValueError as exc:
            show_error("train", str(exc))
            return None
        return resolved_mode, feature_mode_label(resolved_mode)

    def on_training_mode_select(_: ft.ControlEvent) -> None:
        resolved = resolve_training_feature_mode()
        if resolved is None:
            return
        _, selected_mode_label = resolved
        push_train_status(f"Training feature mode set to {selected_mode_label}.")

    def resolve_training_model_family() -> tuple[str, str]:
        resolved_family = (
            MODEL_FAMILY_SPECTRAL_1D_CNN
            if bool(training_spectral_cnn_checkbox.value)
            else MODEL_FAMILY_TABULAR_DENSE
        )
        return resolved_family, model_family_label(resolved_family)

    def resolve_training_sequence_standardization() -> tuple[bool, str]:
        use_standardized_reflectance = bool(training_sequence_standardization_checkbox.value)
        return (
            use_standardized_reflectance,
            spectral_cnn_sequence_input_label(use_standardized_reflectance),
        )

    def sync_training_model_controls(refresh: bool = True) -> None:
        model_family_value, _ = resolve_training_model_family()
        is_spectral_cnn = model_family_value == MODEL_FAMILY_SPECTRAL_1D_CNN
        training_sequence_standardization_checkbox.disabled = not is_spectral_cnn
        if refresh:
            request_ui_refresh()

    def on_training_model_family_toggle(_: ft.ControlEvent) -> None:
        _, selected_family_label = resolve_training_model_family()
        sync_training_model_controls(refresh=False)
        if bool(training_spectral_cnn_checkbox.value):
            _, sequence_input_label = resolve_training_sequence_standardization()
            push_train_status(
                f"Training model method set to {selected_family_label}. Spectral inputs: {sequence_input_label}."
            )
        else:
            push_train_status(f"Training model method set to {selected_family_label}.")
        request_ui_refresh()

    def on_training_sequence_standardization_toggle(_: ft.ControlEvent) -> None:
        selected_family_value, _ = resolve_training_model_family()
        if selected_family_value == MODEL_FAMILY_SPECTRAL_1D_CNN:
            _, sequence_input_label = resolve_training_sequence_standardization()
            push_train_status(f"Spectral CNN inputs set to {sequence_input_label}.")
        else:
            push_train_status("Standardized reflectance is only used by the Spectral 1D CNN.", level="warning")

    def parse_training_inputs() -> tuple[int, float, str, str, str, str, bool, str] | None:
        epochs_raw = (training_epochs_field.value or "").strip()
        split_raw = (training_split_field.value or "").strip()
        resolved_mode = resolve_training_feature_mode()
        if resolved_mode is None:
            return None
        feature_mode_value, feature_mode_display = resolved_mode
        model_family_value, model_family_display = resolve_training_model_family()
        sequence_use_standardized_reflectance, sequence_input_display = resolve_training_sequence_standardization()

        try:
            epochs_value = int(epochs_raw)
        except ValueError:
            show_error("train", "Epochs must be an integer between 1 and 1000.")
            return None
        if not 1 <= epochs_value <= 1000:
            show_error("train", "Epochs must be between 1 and 1000.")
            return None

        try:
            split_percent = int(split_raw)
        except ValueError:
            show_error("train", "Validation split must be an integer percentage between 1 and 99.")
            return None
        if not 1 <= split_percent <= 99:
            show_error("train", "Validation split must be between 1 and 99.")
            return None

        return (
            epochs_value,
            split_percent / 100.0,
            feature_mode_value,
            feature_mode_display,
            model_family_value,
            model_family_display,
            sequence_use_standardized_reflectance,
            sequence_input_display,
        )

    async def run_apply(_: ft.ControlEvent) -> None:
        if state["busy"]:
            return
        if not apply_safe_field.value.strip():
            show_error(
                "apply",
                "Choose a single .zip/.tif file, a single .SAFE folder, or a batch folder before running.",
            )
            return
        if not apply_mask_field.value.strip():
            show_error("apply", "Choose a mask shapefile before running.")
            return
        if not apply_output_path_field.value.strip():
            show_error("apply", "Choose an output folder before running.")
            return
        selected_model = (apply_model_dropdown.value or "").strip()
        if not selected_model:
            show_error("apply", "Choose a model from the dropdown before running.")
            return

        input_source = apply_safe_field.value.strip()
        output_folder = apply_output_path_field.value.strip()
        preflight_signature = (input_source, output_folder, selected_model)
        if apply_preflight_state["running"]:
            show_error("apply", "Please wait until the output check finishes before starting the run.")
            return

        scene_batch_info = apply_preflight_state.get("scene_batch_info")
        if apply_preflight_state.get("signature") != preflight_signature or not scene_batch_info:
            show_error(
                "apply",
                "Apply pre-run check is not ready yet. Wait for the animated output check to finish.",
            )
            return

        scene_inputs = list(apply_preflight_state.get("pending_scene_inputs", []))
        skipped_existing_scene_inputs = list(
            apply_preflight_state.get("skipped_existing_scene_inputs", [])
        )
        skipped_existing_outputs = list(apply_preflight_state.get("skipped_existing_outputs", []))
        ignored_count = int(scene_batch_info.get("ignored_count", 0))
        selected_scene_count = len(scene_batch_info.get("selected", []))
        batch_mode = _is_batch_scene_source(input_source)

        hide_batch_popup()
        apply_progress.value = 0
        apply_spinner.visible = True
        state["operation_mode"] = "apply"
        state["apply_run_token"] = int(state.get("apply_run_token", 0)) + 1
        apply_run_token = int(state["apply_run_token"])
        push_apply_status("Queued classification run.")
        set_global_busy(True, "Processing the selected apply input.")
        update_overlay(
            title="Applying ICE CREAMS",
            detail="Preparing the selected input.",
            progress=0,
            counter="",
            job="Apply workflow",
            source=input_source,
            destination=apply_output_path_field.value.strip(),
        )
        loop = asyncio.get_running_loop()
        run_failed = False
        run_error_message = ""
        run_started_at = datetime.now()
        run_started_clock = time.monotonic()
        completed_outputs: list[str] = []
        archived_done_inputs: list[str] = []
        archived_failed_inputs: list[str] = []
        failed_scene_records: list[tuple[str, str]] = []
        archive_warnings: list[str] = []
        single_scene_replacement_input = {"value": None}
        total_scenes = len(scene_inputs)
        detected_model_family_label = {"value": ""}
        detected_feature_mode_label = {"value": ""}
        detected_sequence_input_label = {"value": ""}

        def _is_current_apply_run(run_token: int = apply_run_token) -> bool:
            return (
                state["busy"]
                and state["operation_mode"] == "apply"
                and int(state.get("apply_run_token", -1)) == run_token
            )

        def _dispatch_apply_status(
            status_message: str,
            counter_value: str,
            source_value: str,
            destination_value: str,
        ) -> None:
            if not _is_current_apply_run():
                return
            if not status_message.strip():
                return
            push_apply_status(status_message, refresh=False)
            update_overlay(
                detail=status_message,
                counter=counter_value,
                source=source_value,
                destination=destination_value,
            )
            request_ui_refresh()

        last_apply_progress_ui = {"value": -1.0, "timestamp": 0.0}
        def _dispatch_apply_progress(
            overall_progress: float,
            counter_value: str,
            source_value: str,
            destination_value: str,
        ) -> None:
            if not _is_current_apply_run():
                return
            current_time = time.monotonic()
            progress_delta = abs(overall_progress - float(last_apply_progress_ui["value"]))
            elapsed = current_time - float(last_apply_progress_ui["timestamp"])
            if (
                overall_progress < 0.999
                and progress_delta < 0.01
                and elapsed < 0.10
            ):
                return
            last_apply_progress_ui["value"] = overall_progress
            last_apply_progress_ui["timestamp"] = current_time
            push_apply_progress(overall_progress, refresh=False)
            update_overlay(
                progress=overall_progress,
                counter=counter_value,
                source=source_value,
                destination=destination_value,
            )
            request_ui_refresh()

        def _scene_status_prefix(scene_index: int, scene_total: int = total_scenes) -> str:
            if scene_total > 1 or batch_mode:
                return f"[{scene_index}/{scene_total}] "
            return ""

        def _archive_scene_input(
            scene_input: str,
            status_key: str,
        ) -> tuple[str | None, str | None]:
            try:
                archived_path = move_scene_input_to_status_folder(scene_input, status_key)
            except FileNotFoundError:
                return None, None
            except Exception as exc:  # noqa: BLE001 - surface archive warnings to the user.
                return None, str(exc)

            if not batch_mode and selected_scene_count == 1:
                single_scene_replacement_input["value"] = archived_path
            return archived_path, None

        def _cleanup_partial_scene_output(scene_output: str, scene_label: str) -> None:
            output_path = Path(scene_output)
            if not output_path.exists():
                return
            try:
                output_path.unlink()
            except Exception as exc:  # noqa: BLE001 - surface cleanup warnings to the user.
                warning_message = (
                    f"Could not remove partial output for {scene_label}: {exc}"
                )
                archive_warnings.append(warning_message)
                push_apply_status(warning_message, level="warning")

        try:
            for scene_input in skipped_existing_scene_inputs:
                scene_label = _derive_scene_stem(scene_input)
                archived_path, archive_error = _archive_scene_input(scene_input, "done")
                if archived_path:
                    archived_done_inputs.append(archived_path)
                elif archive_error:
                    warning_message = (
                        f"Could not move already processed input {scene_label} to Done: {archive_error}"
                    )
                    archive_warnings.append(warning_message)
                    push_apply_status(warning_message, level="warning")

            for index, scene_input in enumerate(scene_inputs, start=1):
                scene_label = _derive_scene_stem(scene_input)
                scene_output = build_apply_output_path(scene_input, output_folder, selected_model)
                counter_text = (
                    f"Scene {index} of {total_scenes}"
                    if total_scenes > 1 or batch_mode
                    else "Single scene run"
                )
                update_overlay(
                    title="Applying ICE CREAMS",
                    detail=f"Preparing {scene_label}",
                    counter=counter_text,
                    job="Batch apply" if total_scenes > 1 or batch_mode else "Single apply",
                    source=scene_input,
                    destination=scene_output,
                )

                def format_apply_status_message(
                    raw_message: str,
                    is_batch_run: bool,
                    current_scene_input: str,
                    current_scene_output: str,
                ) -> str:
                    message = (raw_message or "").strip()
                    if not is_batch_run:
                        return message
                    if message.startswith("Detected model method: "):
                        return message
                    if message.startswith("Detected model feature mode: "):
                        return message
                    if message.startswith("Detected spectral inputs: "):
                        return message
                    if message.startswith("Extracting zipped Sentinel-2 scene from "):
                        return "Extracting zipped Sentinel-2 scene"
                    if message.startswith("Reading Sentinel-2 scene from "):
                        return "Reading Sentinel-2 scene"
                    if message.startswith("Reading multi-band TIFF from "):
                        return "Reading multi-band TIFF"
                    if message.startswith("Writing Cloud-Optimised GeoTIFF to "):
                        return "Writing Cloud-Optimised GeoTIFF"
                    if message.startswith("Completed. Output saved to "):
                        return "Completed. Output saved"
                    if message.startswith("Completed (") and "). Output saved to " in message:
                        return message.split(". Output saved to ", 1)[0] + ". Output saved"
                    for path_value in (current_scene_input, current_scene_output):
                        if path_value:
                            message = message.replace(path_value, "").strip()
                    return message

                def schedule_apply_status(
                    message: str,
                    scene_index: int = index,
                    scene_total: int = total_scenes,
                    current_scene_input: str = scene_input,
                    current_scene_output: str = scene_output,
                ) -> None:
                    is_batch_run = scene_total > 1 or batch_mode
                    if message.startswith("Detected model method: "):
                        detected_model_family_label["value"] = message.split(": ", 1)[1].strip()
                    if message.startswith("Detected model feature mode: "):
                        detected_feature_mode_label["value"] = message.split(": ", 1)[1].strip()
                    if message.startswith("Detected spectral inputs: "):
                        detected_sequence_input_label["value"] = message.split(": ", 1)[1].strip()
                    prefix = (
                        f"[{scene_index}/{scene_total}] "
                        if is_batch_run
                        else ""
                    )
                    formatted_message = format_apply_status_message(
                        message,
                        is_batch_run,
                        current_scene_input,
                        current_scene_output,
                    )
                    status_message = prefix + formatted_message
                    loop.call_soon_threadsafe(
                        _dispatch_apply_status,
                        status_message,
                        counter_text,
                        current_scene_input,
                        current_scene_output,
                    )

                def schedule_apply_progress(
                    value: float,
                    scene_index: int = index,
                    scene_total: int = total_scenes,
                ) -> None:
                    overall = ((scene_index - 1) + max(0.0, min(1.0, value))) / max(scene_total, 1)
                    loop.call_soon_threadsafe(
                        _dispatch_apply_progress,
                        overall,
                        counter_text,
                        scene_input,
                        scene_output,
                    )

                try:
                    result = await asyncio.to_thread(
                        classify_s2_scene,
                        scene_input,
                        scene_output,
                        selected_model,
                        apply_mask_field.value.strip(),
                        False,
                        schedule_apply_status,
                        schedule_apply_progress,
                    )
                except Exception as exc:  # noqa: BLE001 - continue processing the next scene.
                    failed_scene_records.append((scene_input, str(exc)))
                    _cleanup_partial_scene_output(scene_output, scene_label)
                    archived_path, archive_error = _archive_scene_input(scene_input, "failed")
                    failure_message = f"{_scene_status_prefix(index)}Failed {scene_label}: {exc}"
                    if archived_path:
                        archived_failed_inputs.append(archived_path)
                        failure_message += " Input moved to Failed."
                    elif archive_error:
                        warning_message = (
                            f"Could not move failed input {scene_label} to Failed: {archive_error}"
                        )
                        archive_warnings.append(warning_message)
                        push_apply_status(warning_message, level="warning")
                        failure_message += " Input could not be moved to Failed."
                    push_apply_status(failure_message, level="error", refresh=False)
                    update_overlay(
                        detail=failure_message,
                        counter=counter_text,
                        source=scene_input,
                        destination=scene_output,
                    )
                    _dispatch_apply_progress(
                        index / max(total_scenes, 1),
                        counter_text,
                        scene_input,
                        scene_output,
                    )
                    request_ui_refresh()
                    continue

                completed_outputs.append(result)
                archived_path, archive_error = _archive_scene_input(scene_input, "done")
                if archived_path:
                    archived_done_inputs.append(archived_path)
                elif archive_error:
                    warning_message = (
                        f"Output written for {scene_label}, but the input could not be moved to Done: "
                        f"{archive_error}"
                    )
                    archive_warnings.append(warning_message)
                    push_apply_status(warning_message, level="warning")

            processed_count = len(completed_outputs)
            skipped_existing_count = len(skipped_existing_outputs)
            failed_count = len(failed_scene_records)
            done_moved_count = len(archived_done_inputs)
            failed_moved_count = len(archived_failed_inputs)
            method_suffix = (
                f" Detected method: {detected_model_family_label['value']}."
                if detected_model_family_label["value"]
                else ""
            )
            mode_suffix = (
                f" Detected mode: {detected_feature_mode_label['value']}."
                if detected_feature_mode_label["value"]
                else ""
            )
            spectral_input_suffix = (
                f" Spectral inputs: {detected_sequence_input_label['value']}."
                if detected_sequence_input_label["value"]
                else ""
            )
            apply_progress.value = 1.0

            if failed_count == 0:
                if processed_count == 1 and skipped_existing_count == 0:
                    done_suffix = (
                        " Input moved to Done."
                        if done_moved_count == 1
                        else " Input could not be moved to Done."
                    )
                    push_apply_status(
                        f"Workflow finished successfully. Output written to {completed_outputs[0]}."
                        f"{done_suffix}{method_suffix}{mode_suffix}{spectral_input_suffix}"
                    )
                elif processed_count > 0 and skipped_existing_count == 0:
                    push_apply_status(
                        f"Batch workflow finished successfully. {processed_count} output(s) were written to "
                        f"{output_folder}. {done_moved_count} input(s) moved to Done."
                        f"{method_suffix}{mode_suffix}{spectral_input_suffix}"
                    )
                elif processed_count > 0:
                    push_apply_status(
                        f"Apply workflow completed. {processed_count} output(s) written, "
                        f"{skipped_existing_count} already processed scene(s) detected, and "
                        f"{done_moved_count} input(s) moved to Done."
                        f"{method_suffix}{mode_suffix}{spectral_input_suffix}"
                    )
                elif skipped_existing_count > 0:
                    push_apply_status(
                        "No new outputs were written. "
                        f"{skipped_existing_count} scene(s) were already processed and "
                        f"{done_moved_count} input(s) were moved to Done."
                        f"{method_suffix}{mode_suffix}{spectral_input_suffix}",
                    )
                else:
                    push_apply_status("No scenes were queued for processing.", level="warning")
                update_overlay(
                    detail="Apply workflow completed.",
                    progress=1.0,
                    counter="",
                    source=input_source,
                    destination=output_folder,
                )
                report_idle("Apply workflow completed.")
            else:
                summarized_failures = [
                    f"{_derive_scene_stem(scene_path)}: {error_message}"
                    for scene_path, error_message in failed_scene_records[:3]
                ]
                run_error_message = "; ".join(summarized_failures)
                remaining_failures = failed_count - len(summarized_failures)
                if remaining_failures > 0:
                    run_error_message += f"; +{remaining_failures} more failure(s)"
                summary_message = (
                    f"Apply workflow completed with errors. {processed_count} output(s) written, "
                    f"{failed_count} scene(s) failed, {done_moved_count} input(s) moved to Done, and "
                    f"{failed_moved_count} input(s) moved to Failed."
                    f"{method_suffix}{mode_suffix}{spectral_input_suffix}"
                )
                push_apply_status(summary_message, level="error")
                update_overlay(
                    detail=summary_message,
                    progress=1.0,
                    counter="",
                    source=input_source,
                    destination=output_folder,
                )
                set_app_status("error", "Apply workflow completed with failures.")
        except Exception as exc:  # noqa: BLE001 - surface backend errors to the user.
            run_failed = True
            run_error_message = str(exc)
            push_apply_status(f"Run failed: {exc}", level="error")
            update_overlay(detail=f"Run failed: {exc}")
            set_app_status("error", "Apply workflow failed.")
        finally:
            state["apply_run_token"] = int(state.get("apply_run_token", 0)) + 1
            apply_spinner.visible = False
            set_global_busy(False)
            overlay_blocker.visible = False
            processed_count = len(completed_outputs)
            skipped_existing_count = len(skipped_existing_outputs)
            failed_count = len(failed_scene_records)
            done_moved_count = len(archived_done_inputs)
            failed_moved_count = len(archived_failed_inputs)
            history_status_value = "failed" if run_failed else "success"
            if not run_failed:
                if failed_count > 0 and (processed_count > 0 or skipped_existing_count > 0):
                    history_status_value = "partial"
                elif failed_count > 0:
                    history_status_value = "failed"
                elif processed_count > 0 and skipped_existing_count > 0:
                    history_status_value = "partial"
                elif processed_count == 0 and skipped_existing_count > 0:
                    history_status_value = "skipped"
            history_output_path = output_folder
            if processed_count == 1 and completed_outputs:
                history_output_path = completed_outputs[0]
            mask_path_value = apply_mask_field.value.strip()
            record_run_history(
                workflow="apply",
                status=history_status_value,
                started_at=run_started_at,
                duration_seconds=time.monotonic() - run_started_clock,
                input_path=input_source,
                model_path=selected_model,
                output_path=history_output_path,
                mask_path=mask_path_value,
                error_message=run_error_message,
                details=(
                    f"model_family={detected_model_family_label['value'] or '-'}; "
                    f"feature_mode={detected_feature_mode_label['value'] or '-'}; "
                    f"spectral_inputs={detected_sequence_input_label['value'] or '-'}; "
                    f"Scenes={selected_scene_count}; queued={total_scenes}; processed={processed_count}; "
                    f"skipped_existing={skipped_existing_count}; failed={failed_count}; "
                    f"moved_done={done_moved_count}; moved_failed={failed_moved_count}; "
                    f"archive_warnings={len(archive_warnings)}; "
                    f"duplicate_skipped={scene_batch_info.get('skipped_count', 0)}; ignored={ignored_count}"
                ),
                location_name=_resolve_location_from_mask(mask_path_value),
                mask_extent_coords=_resolve_mask_extent_coordinates(mask_path_value),
            )
            if single_scene_replacement_input["value"]:
                apply_safe_field.value = str(single_scene_replacement_input["value"])
            _reset_apply_preflight_state()
            refresh_apply_preview()
            refresh_apply_run_button_state()
            if run_failed:
                set_app_status("error", "Apply workflow failed.")
            request_ui_refresh(force=True)

    async def run_training(_: ft.ControlEvent) -> None:
        if state["busy"]:
            return
        if not selected_training_csvs:
            show_error(
                "train",
                "Choose one or more training CSV files (directly or via selected folders) before training.",
            )
            return
        if not training_model_name_field.value.strip():
            show_error("train", "Provide a filename for the exported model.")
            return

        output_model = build_training_output_path()
        if not output_model:
            show_error("train", "Unable to build an output model path from the selected inputs.")
            return
        parsed_training_inputs = parse_training_inputs()
        if parsed_training_inputs is None:
            return
        (
            epochs_to_run,
            valid_pct_to_use,
            feature_mode_value,
            feature_mode_display,
            model_family_value,
            model_family_display,
            sequence_use_standardized_reflectance,
            sequence_input_display,
        ) = parsed_training_inputs

        train_progress.value = 0
        train_spinner.visible = True
        state["operation_mode"] = "train"
        push_train_status(
            (
                f"Queued model training run ({model_family_display}, {feature_mode_display})"
                + (
                    f" with {sequence_input_display}."
                    if model_family_value == MODEL_FAMILY_SPECTRAL_1D_CNN
                    else "."
                )
            )
        )
        set_global_busy(
            True,
            (
                f"Training a new {model_family_display} ({feature_mode_display}) ICE CREAMS model."
                if model_family_value != MODEL_FAMILY_SPECTRAL_1D_CNN
                else (
                    f"Training a new {model_family_display} ({feature_mode_display}) ICE CREAMS model "
                    f"with {sequence_input_display}."
                )
            ),
        )
        update_overlay(
            title="Training Model",
            detail=(
                f"Preparing the {model_family_display} ({feature_mode_display}) training run."
                if model_family_value != MODEL_FAMILY_SPECTRAL_1D_CNN
                else (
                    f"Preparing the {model_family_display} ({feature_mode_display}) training run "
                    f"with {sequence_input_display}."
                )
            ),
            progress=0,
            counter="Training session",
            job=f"Model training ({model_family_display})",
            source=_training_source_overlay_path(),
            destination=output_model,
        )
        loop = asyncio.get_running_loop()
        run_failed = False
        run_error_message = ""
        run_started_at = datetime.now()
        run_started_clock = time.monotonic()
        training_result: dict[str, object] | None = None

        def schedule_train_status(message: str) -> None:
            loop.call_soon_threadsafe(push_train_status, message)

        def schedule_train_progress(value: float) -> None:
            loop.call_soon_threadsafe(push_train_progress, value)

        try:
            result = await asyncio.to_thread(
                train_model,
                selected_training_csvs.copy(),
                output_model,
                int(epochs_to_run),
                float(valid_pct_to_use),
                4096,
                42,
                feature_mode_value,
                model_family_value,
                sequence_use_standardized_reflectance,
                schedule_train_status,
                schedule_train_progress,
            )
            training_result = result
            accuracy = result.get("accuracy")
            trained_mode_label = str(result.get("feature_mode_label") or feature_mode_display)
            trained_model_family_label = str(
                result.get("model_family_label") or model_family_display
            )
            trained_sequence_input_label = str(
                result.get("sequence_input_label") or sequence_input_display
            )
            accuracy_text = (
                f" Validation accuracy: {accuracy:.4f}."
                if isinstance(accuracy, float)
                else ""
            )
            spectral_input_text = (
                f"Spectral inputs: {trained_sequence_input_label}. "
                if model_family_value == MODEL_FAMILY_SPECTRAL_1D_CNN
                else ""
            )
            push_train_status(
                f"Training completed ({trained_model_family_label}, {trained_mode_label}). "
                f"{spectral_input_text}"
                f"{result['rows']} rows across {result['csv_files']} CSV files were used."
                f"{accuracy_text}"
            )
            report_idle("Training completed.")
        except Exception as exc:  # noqa: BLE001 - surface backend errors to the user.
            run_failed = True
            run_error_message = str(exc)
            push_train_status(f"Training failed: {exc}", level="error")
            update_overlay(detail=f"Training failed: {exc}")
            set_app_status("error", "Training failed.")
        finally:
            state["train_run_token"] = int(state.get("train_run_token", 0)) + 1
            train_spinner.visible = False
            set_global_busy(False)
            overlay_blocker.visible = False
            trained_model_path = output_model
            if training_result and isinstance(training_result.get("model_path"), str):
                trained_model_path = str(training_result["model_path"])
            rows_used = int(_history_float(training_result.get("rows", 0))) if training_result else 0
            csv_files_used = int(_history_float(training_result.get("csv_files", 0))) if training_result else 0
            accuracy_value = training_result.get("accuracy") if training_result else None
            accuracy_display = (
                f"{float(accuracy_value):.4f}"
                if isinstance(accuracy_value, (int, float))
                else "-"
            )
            trained_mode_label = str(
                (training_result or {}).get("feature_mode_label") or feature_mode_display
            )
            trained_model_family_label = str(
                (training_result or {}).get("model_family_label") or model_family_display
            )
            trained_sequence_input_label = str(
                (training_result or {}).get("sequence_input_label") or sequence_input_display
            )
            trained_model_family_value = str(
                (training_result or {}).get("model_family") or model_family_value
            )
            primary_input_path = (
                selected_training_csvs[0]
                if selected_training_csvs
                else (selected_training_csv_folders[0] if selected_training_csv_folders else "")
            )
            record_run_history(
                workflow="train",
                status="failed" if run_failed else "success",
                started_at=run_started_at,
                duration_seconds=time.monotonic() - run_started_clock,
                input_path=primary_input_path,
                model_path=trained_model_path,
                output_path=trained_model_path,
                error_message=run_error_message,
                details=(
                    f"model_family={trained_model_family_label}; mode={trained_mode_label}; "
                    f"spectral_inputs={trained_sequence_input_label if trained_model_family_value == MODEL_FAMILY_SPECTRAL_1D_CNN else '-'}; "
                    f"input_csvs={len(selected_training_csvs)}; rows={rows_used}; "
                    f"csv_files={csv_files_used}; accuracy={accuracy_display}"
                ),
            )
            if not run_failed:
                refresh_model_dropdowns(preferred_model_path=trained_model_path, refresh=False)
            if run_failed:
                set_app_status("error", "Training failed.")
            request_ui_refresh(force=True)

    async def run_validation(_: ft.ControlEvent) -> None:
        if state["busy"]:
            return

        dataset_path = validation_dataset_field.value.strip()
        if not dataset_path:
            show_error("validation", "Choose a validation dataset before running.")
            return

        model_path = resolve_validation_model_path()
        if not model_path:
            show_error("validation", "Choose a model from the dropdown or external picker before running.")
            return

        label_column = (validation_label_column_field.value or "").strip()
        if not label_column:
            show_error("validation", "Provide the label column name before running.")
            return

        validation_mode = (validation_mode_dropdown.value or VALIDATION_MODE_MULTICLASS).strip()
        target_class = (validation_target_class_field.value or "").strip() or DEFAULT_TARGET_CLASS
        if validation_mode == VALIDATION_MODE_PRESENCE_ABSENCE and not target_class:
            show_error("validation", "Provide a target class for presence/absence validation.")
            return

        output_dir = validation_output_dir_field.value.strip()
        if not output_dir:
            show_error("validation", "Choose an output folder before running.")
            return

        hide_validation_popup()
        validation_progress.value = 0
        validation_spinner.visible = True
        state["operation_mode"] = "validation"
        state["validation_run_token"] = int(state.get("validation_run_token", 0)) + 1
        push_validation_status("Queued validation run.")
        mode_display = (
            "presence/absence"
            if validation_mode == VALIDATION_MODE_PRESENCE_ABSENCE
            else "model classes"
        )
        set_global_busy(
            True,
            (
                f"Validating with {mode_display}"
                + (f" (target: {target_class})." if validation_mode == VALIDATION_MODE_PRESENCE_ABSENCE else ".")
            ),
        )
        update_overlay(
            title="Validating Model",
            detail="Preparing validation run.",
            progress=0,
            counter="Validation run",
            job="Model validation",
            source=dataset_path,
            destination=output_dir,
        )
        loop = asyncio.get_running_loop()
        run_failed = False
        validation_result: dict[str, object] | None = None
        run_error_message = ""
        run_started_at = datetime.now()
        run_started_clock = time.monotonic()

        def schedule_validation_status(message: str) -> None:
            loop.call_soon_threadsafe(push_validation_status, message)

        def schedule_validation_progress(value: float) -> None:
            loop.call_soon_threadsafe(push_validation_progress, value)

        try:
            result = await asyncio.to_thread(
                validate_model,
                dataset_path=dataset_path,
                model_path=model_path,
                label_column=label_column,
                output_dir=output_dir,
                status_callback=schedule_validation_status,
                progress_callback=schedule_validation_progress,
                validation_mode=validation_mode,
                target_class=target_class,
            )
            validation_result = result
            detected_model_family_label = str(
                result.get("model_family_label") or "Detected method unavailable"
            )
            detected_feature_mode_label = str(result.get("feature_mode_label") or "Detected mode unavailable")
            detected_sequence_input_label = str(result.get("sequence_input_label") or "")
            spectral_input_text = (
                f"Spectral inputs: {detected_sequence_input_label}. "
                if str(result.get("model_family") or "") == MODEL_FAMILY_SPECTRAL_1D_CNN
                and detected_sequence_input_label
                else ""
            )
            push_validation_status(
                f"Validation completed ({detected_model_family_label}, {detected_feature_mode_label}). "
                f"{spectral_input_text}"
                f"{result['rows']} row(s) evaluated across {result['classes']} class(es). "
                f"Predictions: {result['predictions_csv']} | Metrics: {result['metrics_csv']}"
            )
            update_overlay(
                detail="Validation workflow completed.",
                progress=1.0,
                counter="",
                source=dataset_path,
                destination=output_dir,
            )
            report_idle("Validation completed.")
            refresh_validation_preview()
        except Exception as exc:  # noqa: BLE001 - surface backend errors to the user.
            run_failed = True
            run_error_message = str(exc)
            push_validation_status(f"Validation failed: {exc}", level="error")
            update_overlay(detail=f"Validation failed: {exc}")
            set_app_status("error", "Validation failed.")
        finally:
            state["validation_run_token"] = int(state.get("validation_run_token", 0)) + 1
            validation_spinner.visible = False
            set_global_busy(False)
            overlay_blocker.visible = False
            predictions_output = output_dir
            metrics_output = ""
            rows_evaluated = 0
            class_count = 0
            if validation_result:
                if isinstance(validation_result.get("predictions_csv"), str):
                    predictions_output = str(validation_result["predictions_csv"])
                if isinstance(validation_result.get("metrics_csv"), str):
                    metrics_output = str(validation_result["metrics_csv"])
                rows_evaluated = int(_history_float(validation_result.get("rows", 0)))
                class_count = int(_history_float(validation_result.get("classes", 0)))
            detected_model_family_label = str(
                (validation_result or {}).get("model_family_label") or "-"
            )
            detected_feature_mode_label = str(
                (validation_result or {}).get("feature_mode_label") or "-"
            )
            detected_sequence_input_label = str(
                (validation_result or {}).get("sequence_input_label") or "-"
            )
            detected_model_family_value = str(
                (validation_result or {}).get("model_family") or "-"
            )
            record_run_history(
                workflow="validation",
                status="failed" if run_failed else "success",
                started_at=run_started_at,
                duration_seconds=time.monotonic() - run_started_clock,
                input_path=dataset_path,
                model_path=model_path,
                output_path=predictions_output,
                error_message=run_error_message,
                details=(
                    f"mode={mode_display}; model_family={detected_model_family_label}; "
                    f"feature_mode={detected_feature_mode_label}; "
                    f"spectral_inputs={detected_sequence_input_label if detected_model_family_value == MODEL_FAMILY_SPECTRAL_1D_CNN else '-'}; "
                    f"rows={rows_evaluated}; classes={class_count}; "
                    f"metrics={metrics_output or '-'}"
                ),
            )
            if run_failed:
                set_app_status("error", "Validation failed.")
            request_ui_refresh(force=True)
            if not run_failed and validation_result is not None:
                show_validation_popup(validation_result)

    apply_single_scene_button = ft.ElevatedButton(
        "Single File",
        icon=ft.Icons.IMAGE_SEARCH,
        on_click=choose_single_scene,
        style=_frosted_button_style("#E6F4FF", "#14324C"),
    )
    apply_scene_folder_button = ft.ElevatedButton(
        "SAFE / Batch",
        icon=ft.Icons.FOLDER_OPEN,
        on_click=choose_scene_folder,
        style=_frosted_button_style("#E6F4FF", "#14324C"),
    )
    apply_mask_button = ft.ElevatedButton(
        "Select Mask",
        icon=ft.Icons.MAP,
        on_click=choose_mask_file,
        style=_frosted_button_style("#E6F4FF", "#14324C"),
    )
    apply_output_folder_button = ft.ElevatedButton(
        "Select Output Folder",
        icon=ft.Icons.FOLDER_OPEN,
        on_click=choose_output_folder,
        style=_frosted_button_style("#E6F4FF", "#14324C"),
    )
    apply_run_button = ft.ElevatedButton(
        "Run Workflow",
        icon=ft.Icons.PLAY_ARROW,
        on_click=run_apply,
        style=_frosted_button_style("#D4F7E3", "#103B2F"),
        tooltip="Run Apply workflow (Ctrl+Enter on Apply tab)",
    )

    train_source_button = ft.ElevatedButton(
        "Select CSV(s)",
        icon=ft.Icons.INSERT_DRIVE_FILE,
        on_click=choose_training_dataset,
        style=_frosted_button_style("#E6F4FF", "#14324C"),
    )
    train_source_folder_button = ft.ElevatedButton(
        "Add Folder",
        icon=ft.Icons.FOLDER_OPEN,
        on_click=choose_training_dataset_folder,
        style=_frosted_button_style("#E6F4FF", "#14324C"),
    )
    train_source_clear_button = ft.ElevatedButton(
        "Clear",
        icon=ft.Icons.CLEAR_ALL,
        on_click=clear_training_dataset_selection,
        style=_frosted_button_style("#FFE8E8", "#6A2A2A"),
    )
    train_run_button = ft.ElevatedButton(
        "Train Model",
        icon=ft.Icons.AUTO_GRAPH,
        on_click=run_training,
        style=_frosted_button_style("#DCE9FF", "#102A56"),
        tooltip="Run training (Ctrl+Enter on Train tab)",
    )
    validation_dataset_button = ft.ElevatedButton(
        "Select Dataset",
        icon=ft.Icons.TABLE_CHART,
        on_click=choose_validation_dataset,
        style=_frosted_button_style("#E6F4FF", "#14324C"),
    )
    validation_dropdown_model_button = ft.ElevatedButton(
        "Clear External Model",
        icon=ft.Icons.UNDO,
        on_click=clear_validation_external_model,
        style=_frosted_button_style("#FFE8E8", "#6A2A2A"),
    )
    validation_external_model_button = ft.ElevatedButton(
        "Select External .pkl",
        icon=ft.Icons.UPLOAD_FILE,
        on_click=choose_validation_model_file,
        style=_frosted_button_style("#E6F4FF", "#14324C"),
    )
    validation_output_button = ft.ElevatedButton(
        "Select Output Folder",
        icon=ft.Icons.FOLDER_OPEN,
        on_click=choose_validation_output_dir,
        style=_frosted_button_style("#E6F4FF", "#14324C"),
    )
    validation_run_button = ft.ElevatedButton(
        "Run Validation",
        icon=ft.Icons.FACT_CHECK,
        on_click=run_validation,
        style=_frosted_button_style("#DCE9FF", "#102A56"),
        tooltip="Run validation (Ctrl+Enter on Validation tab)",
    )
    history_refresh_button = ft.ElevatedButton(
        "Refresh History",
        icon=ft.Icons.REFRESH,
        on_click=on_history_refresh,
        style=_frosted_button_style("#E6F4FF", "#14324C"),
        tooltip="Refresh history (Ctrl+R on History tab)",
    )

    interactive_controls = [
        apply_single_scene_button,
        apply_scene_folder_button,
        apply_mask_button,
        apply_output_folder_button,
        apply_model_dropdown,
        apply_run_button,
        train_source_button,
        train_source_folder_button,
        train_source_clear_button,
        train_run_button,
        training_model_name_field,
        training_epochs_field,
        training_split_field,
        training_mode_dropdown,
        training_spectral_cnn_checkbox,
        training_sequence_standardization_checkbox,
        validation_dataset_button,
        validation_dropdown_model_button,
        validation_external_model_button,
        validation_output_button,
        validation_model_dropdown,
        validation_label_column_field,
        validation_mode_dropdown,
        validation_target_class_field,
        validation_run_button,
        history_refresh_button,
    ]
    apply_model_dropdown.on_select = on_apply_model_select
    validation_model_dropdown.on_select = on_validation_model_select
    validation_label_column_field.on_submit = on_validation_label_column_change
    validation_mode_dropdown.on_select = on_validation_mode_select
    validation_target_class_field.on_submit = on_validation_target_class_change
    training_mode_dropdown.on_select = on_training_mode_select
    training_spectral_cnn_checkbox.on_change = on_training_model_family_toggle
    training_sequence_standardization_checkbox.on_change = on_training_sequence_standardization_toggle
    sync_training_model_controls(refresh=False)
    sync_validation_mode_controls(refresh=False)

    apply_intro_panel = _workflow_intro_panel(
        "Apply ICE CREAMS",
        "Run classification with a clear, guided sequence to reduce cognitive load and avoid setup mistakes.",
        [
            "Select one SAFE/ZIP/TIFF input or a batch folder",
            "Choose the mask polygon",
            "Set output folder and model",
            "Run and monitor progress",
        ],
    )
    train_intro_panel = _workflow_intro_panel(
        "Train a Model",
        "Build a new model from CSV datasets with a consistent workflow from data selection to export.",
        [
            "Select CSV files and/or folders",
            "Set output model name",
            "Set method, mode, epochs, and validation split",
            "Train and review final accuracy",
        ],
    )
    validation_intro_panel = _workflow_intro_panel(
        "Validate a Model",
        "Evaluate a trained model against labelled data and export reproducible metrics and predictions.",
        [
            "Select validation dataset",
            "Select model and label field",
            "Choose output folder and mode",
            "Run and review confusion matrix",
        ],
    )
    train_paths_panel_ref: ft.Container | None = None
    train_settings_panel_ref: ft.Container | None = None
    train_main_content_ref: ft.Container | None = None
    validation_paths_panel_ref: ft.Container | None = None
    validation_preview_panel_ref: ft.Container | None = None
    validation_main_content_ref: ft.Container | None = None
    history_main_content_ref: ft.Container | None = None

    apply_paths_panel = _glass_panel(
        padding=18,
        content=ft.Column(
            spacing=12,
            controls=[
                ft.Text(
                    "Paths",
                    size=14,
                    weight=ft.FontWeight.W_600,
                    color=LIQUID_TEXT,
                ),
                ft.ResponsiveRow(
                    columns=12,
                    run_spacing=12,
                    spacing=12,
                    controls=[
                        ft.Container(
                            col={"xs": 12, "sm": 6, "lg": 4},
                            padding=14,
                            border_radius=20,
                            bgcolor=ft.Colors.with_opacity(0.62, LIQUID_SURFACE_ALT),
                            border=ft.border.all(1, ft.Colors.with_opacity(0.30, ft.Colors.WHITE)),
                            content=ft.Column(
                                spacing=10,
                                controls=[
                                    ft.Text(
                                        "Image Input",
                                        size=14,
                                        weight=ft.FontWeight.W_600,
                                        color=LIQUID_TEXT,
                                    ),
                                    apply_safe_field,
                                    ft.Row(
                                        wrap=True,
                                        spacing=8,
                                        controls=[
                                            apply_single_scene_button,
                                            apply_scene_folder_button,
                                        ],
                                    ),
                                    ft.Text(
                                        "Single .zip/.tif: use Single File. Single .SAFE or batch folder with .SAFE/.zip/.tif: use SAFE / Batch.",
                                        size=11,
                                        color=LIQUID_MUTED,
                                    ),
                                ],
                            ),
                        ),
                        ft.Container(
                            col={"xs": 12, "sm": 6, "lg": 4},
                            padding=14,
                            border_radius=20,
                            bgcolor=ft.Colors.with_opacity(0.62, LIQUID_SURFACE_ALT),
                            border=ft.border.all(1, ft.Colors.with_opacity(0.30, ft.Colors.WHITE)),
                            content=ft.Column(
                                spacing=10,
                                controls=[
                                    ft.Text(
                                        "Mask",
                                        size=14,
                                        weight=ft.FontWeight.W_600,
                                        color=LIQUID_TEXT,
                                    ),
                                    apply_mask_field,
                                    apply_mask_button,
                                    ft.Text(
                                        "Only pixels inside this polygon mask are classified.",
                                        size=11,
                                        color=LIQUID_MUTED,
                                    ),
                                ],
                            ),
                        ),
                        ft.Container(
                            col={"xs": 12, "md": 4},
                            padding=14,
                            border_radius=20,
                            bgcolor=ft.Colors.with_opacity(0.62, LIQUID_SURFACE_ALT),
                            border=ft.border.all(1, ft.Colors.with_opacity(0.30, ft.Colors.WHITE)),
                            content=ft.Column(
                                spacing=10,
                                controls=[
                                    ft.Text(
                                        "Output",
                                        size=14,
                                        weight=ft.FontWeight.W_600,
                                        color=LIQUID_TEXT,
                                    ),
                                    apply_output_path_field,
                                    apply_output_folder_button,
                                    ft.Text(
                                        "GeoTIFF files are auto-named and written in this folder.",
                                        size=11,
                                        color=LIQUID_MUTED,
                                    ),
                                ],
                            ),
                        ),
                    ],
                ),
            ],
        ),
    )

    apply_bottom_cards_height = 190
    apply_preview_card_body = ft.Container(
        height=apply_bottom_cards_height,
        content=ft.Column(
            spacing=10,
            controls=[
                ft.Text(
                    "Output Preview",
                    size=14,
                    weight=ft.FontWeight.W_600,
                    color=LIQUID_TEXT,
                ),
                apply_output_preview,
                apply_progress,
                apply_status,
            ],
        ),
    )
    apply_feed_card_body = ft.Container(
        height=apply_bottom_cards_height,
        content=ft.Column(
            expand=True,
            spacing=12,
            controls=[
                ft.Text(
                    "Activity Feed",
                    size=14,
                    weight=ft.FontWeight.W_600,
                    color=LIQUID_TEXT,
                ),
                ft.Container(
                    expand=True,
                    content=apply_log,
                ),
            ],
        ),
    )

    apply_model_panel = ft.Container(
        padding=ft.padding.symmetric(horizontal=18, vertical=16),
        border_radius=22,
        bgcolor=ft.Colors.with_opacity(0.62, LIQUID_SURFACE_ALT),
        border=ft.border.all(1, ft.Colors.with_opacity(0.30, ft.Colors.WHITE)),
        content=ft.Column(
            spacing=10,
            controls=[
                ft.Text(
                    "Model",
                    size=14,
                    weight=ft.FontWeight.W_600,
                    color=LIQUID_TEXT,
                ),
                apply_model_dropdown,
                ft.Text(
                    "Choose a model from the models folder.",
                    size=11,
                    color=LIQUID_MUTED,
                ),
            ],
        ),
    )

    apply_main_content = ft.Container(
        width=1460,
        padding=ft.padding.only(bottom=90),
        content=ft.Column(
            expand=True,
            scroll=ft.ScrollMode.AUTO,
            spacing=16,
            controls=[
                apply_intro_panel,
                apply_paths_panel,
                apply_model_panel,
                ft.ResponsiveRow(
                    columns=12,
                    run_spacing=16,
                    controls=[
                        ft.Container(
                            col={"xs": 12, "lg": 4},
                            content=_glass_panel(
                                padding=18,
                                content=apply_preview_card_body,
                            ),
                        ),
                        ft.Container(
                            col={"xs": 12, "lg": 8},
                            content=_glass_panel(
                                padding=18,
                                content=apply_feed_card_body,
                            ),
                        ),
                    ],
                ),
            ],
        ),
    )

    apply_run_badge = ft.Container(
        right=24,
        bottom=20,
        content=_glass_panel(
            padding=10,
            content=ft.Row(
                spacing=10,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                controls=[
                    apply_spinner,
                    apply_run_button,
                ],
            ),
        ),
    )

    apply_view = _glass_panel(
        expand=True,
        content=ft.Stack(
            expand=True,
            controls=[
                ft.Container(
                    expand=True,
                    alignment=ft.Alignment(0, -1),
                    content=apply_main_content,
                ),
                apply_run_badge,
            ],
        ),
    )

    train_log_container = ft.Container(
        expand=True,
        height=260,
        content=train_log,
    )

    train_bottom_cards_height = 190
    train_paths_card_height = 248
    train_progress_card_body = ft.Container(
        height=train_bottom_cards_height,
        content=ft.Column(
            spacing=10,
            controls=[
                ft.Text(
                    "Training Progress",
                    size=14,
                    weight=ft.FontWeight.W_600,
                    color=LIQUID_TEXT,
                ),
                train_progress,
                train_status,
            ],
        ),
    )
    train_feed_card_body = ft.Container(
        height=train_bottom_cards_height,
        content=ft.Column(
            expand=True,
            spacing=12,
            controls=[
                ft.Text(
                    "Training Feed",
                    size=14,
                    weight=ft.FontWeight.W_600,
                    color=LIQUID_TEXT,
                ),
                train_log_container,
            ],
        ),
    )

    lazy_tab_view_cache: dict[str, ft.Control] = {}
    lazy_tab_layout_refs: dict[str, dict[str, ft.Control | None]] = {
        "train": {"main_content": None, "badge": None},
        "validation": {"main_content": None, "badge": None},
        "history": {"main_content": None, "badge": None},
    }
    history_view_initialized = {"value": False}

    def _build_train_view() -> ft.Control:
        nonlocal train_main_content_ref, train_paths_panel_ref, train_settings_panel_ref
        cached_view = lazy_tab_view_cache.get("train")
        if cached_view is not None:
            return cached_view

        train_paths_panel = _glass_panel(
            padding=18,
            content=ft.Column(
                spacing=12,
                controls=[
                    ft.Text(
                        "Training Paths",
                        size=14,
                        weight=ft.FontWeight.W_600,
                        color=LIQUID_TEXT,
                    ),
                    ft.ResponsiveRow(
                        columns=12,
                        run_spacing=12,
                        spacing=12,
                        controls=[
                            ft.Container(
                                col={"xs": 12, "md": 6},
                                height=train_paths_card_height,
                                padding=14,
                                border_radius=20,
                                bgcolor=ft.Colors.with_opacity(0.62, LIQUID_SURFACE_ALT),
                                border=ft.border.all(1, ft.Colors.with_opacity(0.30, ft.Colors.WHITE)),
                                content=ft.Column(
                                    expand=True,
                                    spacing=10,
                                    controls=[
                                        ft.Text(
                                            "Training Dataset (CSV)",
                                            size=14,
                                            weight=ft.FontWeight.W_600,
                                            color=LIQUID_TEXT,
                                        ),
                                        training_source_field,
                                        ft.Row(
                                            wrap=True,
                                            spacing=8,
                                            controls=[
                                                train_source_button,
                                                train_source_folder_button,
                                                train_source_clear_button,
                                            ],
                                        ),
                                        ft.Container(expand=True),
                                        ft.Text(
                                            "Add one or several CSV files and/or one or several folders containing CSV files. All discovered CSV files are merged into one training table.",
                                            size=11,
                                            color=LIQUID_MUTED,
                                        ),
                                    ],
                                ),
                            ),
                            ft.Container(
                                col={"xs": 12, "md": 6},
                                height=train_paths_card_height,
                                padding=14,
                                border_radius=20,
                                bgcolor=ft.Colors.with_opacity(0.62, LIQUID_SURFACE_ALT),
                                border=ft.border.all(1, ft.Colors.with_opacity(0.30, ft.Colors.WHITE)),
                                content=ft.Column(
                                    expand=True,
                                    spacing=10,
                                    controls=[
                                        ft.Text(
                                            "Model Output",
                                            size=14,
                                            weight=ft.FontWeight.W_600,
                                            color=LIQUID_TEXT,
                                        ),
                                        training_output_dir_field,
                                        training_model_name_field,
                                        ft.Container(expand=True),
                                        ft.Text(
                                            "New models are always saved to the default models folder with the filename shown above.",
                                            size=11,
                                            color=LIQUID_MUTED,
                                        ),
                                    ],
                                ),
                            ),
                        ],
                    ),
                ],
            ),
        )
        train_paths_panel_ref = train_paths_panel

        train_main_content = ft.Container(
            width=1460,
            content=ft.Column(
                expand=True,
                scroll=ft.ScrollMode.AUTO,
                spacing=16,
                controls=[
                    train_intro_panel,
                    train_paths_panel,
                    ft.Container(
                        padding=ft.padding.symmetric(horizontal=18, vertical=16),
                        border_radius=22,
                        bgcolor=ft.Colors.with_opacity(0.62, LIQUID_SURFACE_ALT),
                        border=ft.border.all(1, ft.Colors.with_opacity(0.30, ft.Colors.WHITE)),
                        content=ft.Column(
                            spacing=14,
                            controls=[
                                ft.Text(
                                    "Training Settings",
                                    size=14,
                                    weight=ft.FontWeight.W_600,
                                    color=LIQUID_TEXT,
                                ),
                                ft.ResponsiveRow(
                                    columns=12,
                                    run_spacing=16,
                                    controls=[
                                        ft.Container(
                                            col={"xs": 12, "md": 3},
                                            content=ft.Column(
                                                spacing=8,
                                                controls=[
                                                    ft.Text(
                                                        "Feature Mode",
                                                        size=13,
                                                        weight=ft.FontWeight.W_500,
                                                        color=LIQUID_SUBTEXT,
                                                    ),
                                                    training_mode_dropdown,
                                                    ft.Text(
                                                        "High Spectral Complexity stays the default training workflow.",
                                                        size=11,
                                                        color=LIQUID_MUTED,
                                                    ),
                                                ],
                                            ),
                                        ),
                                        ft.Container(
                                            col={"xs": 12, "md": 3},
                                            content=ft.Column(
                                                spacing=8,
                                                controls=[
                                                    ft.Text(
                                                        "Model Method",
                                                        size=13,
                                                        weight=ft.FontWeight.W_500,
                                                        color=LIQUID_SUBTEXT,
                                                    ),
                                                    training_spectral_cnn_checkbox,
                                                    training_sequence_standardization_checkbox,
                                                    ft.Text(
                                                        "Checked: use the spectral 1D CNN. Unchecked: use the legacy tabular dense network. The standardized-reflectance option is stored with CNN models and auto-detected during apply and validation.",
                                                        size=11,
                                                        color=LIQUID_MUTED,
                                                    ),
                                                ],
                                            ),
                                        ),
                                        ft.Container(
                                            col={"xs": 12, "md": 3},
                                            content=ft.Column(
                                                spacing=8,
                                                controls=[
                                                    ft.Text(
                                                        "Epochs",
                                                        size=13,
                                                        weight=ft.FontWeight.W_500,
                                                        color=LIQUID_SUBTEXT,
                                                    ),
                                                    training_epochs_field,
                                                    ft.Text(
                                                        "Enter an integer between 1 and 1000.",
                                                        size=11,
                                                        color=LIQUID_MUTED,
                                                    ),
                                                ],
                                            ),
                                        ),
                                        ft.Container(
                                            col={"xs": 12, "md": 3},
                                            content=ft.Column(
                                                spacing=8,
                                                controls=[
                                                    ft.Text(
                                                        "Validation Split",
                                                        size=13,
                                                        weight=ft.FontWeight.W_500,
                                                        color=LIQUID_SUBTEXT,
                                                    ),
                                                    training_split_field,
                                                    ft.Text(
                                                        "Validation split in percent (1 to 99).",
                                                        size=11,
                                                        color=LIQUID_MUTED,
                                                    ),
                                                ],
                                            ),
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ),
                    ft.ResponsiveRow(
                        columns=12,
                        run_spacing=16,
                        controls=[
                            ft.Container(
                                col={"xs": 12, "lg": 4},
                                content=_glass_panel(
                                    padding=18,
                                    content=train_progress_card_body,
                                ),
                            ),
                            ft.Container(
                                col={"xs": 12, "lg": 8},
                                content=_glass_panel(
                                    padding=18,
                                    content=train_feed_card_body,
                                ),
                            ),
                        ],
                    ),
                ],
            ),
        )
        train_main_content_ref = train_main_content
        train_column = train_main_content.content
        if isinstance(train_column, ft.Column) and len(train_column.controls) >= 3:
            train_settings_panel_ref = train_column.controls[2]

        train_run_badge = ft.Container(
            right=24,
            bottom=20,
            content=_glass_panel(
                padding=10,
                content=ft.Row(
                    spacing=10,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=[
                        train_spinner,
                        train_run_button,
                    ],
                ),
            ),
        )

        train_view = _glass_panel(
            expand=True,
            content=ft.Stack(
                expand=True,
                controls=[
                    ft.Container(
                        expand=True,
                        alignment=ft.Alignment(0, -1),
                        content=train_main_content,
                    ),
                    train_run_badge,
                ],
            ),
        )
        lazy_tab_layout_refs["train"]["main_content"] = train_main_content
        lazy_tab_layout_refs["train"]["badge"] = train_run_badge
        lazy_tab_view_cache["train"] = train_view
        return train_view

    validation_log_container = ft.Container(
        expand=True,
        height=260,
        content=validation_log,
    )

    validation_bottom_cards_height = 190
    validation_progress_card_body = ft.Container(
        height=validation_bottom_cards_height,
        content=ft.Column(
            spacing=10,
            controls=[
                ft.Text(
                    "Validation Progress",
                    size=14,
                    weight=ft.FontWeight.W_600,
                    color=LIQUID_TEXT,
                ),
                validation_progress,
                validation_status,
            ],
        ),
    )
    validation_feed_card_body = ft.Container(
        height=validation_bottom_cards_height,
        content=ft.Column(
            expand=True,
            spacing=12,
            controls=[
                ft.Text(
                    "Validation Feed",
                    size=14,
                    weight=ft.FontWeight.W_600,
                    color=LIQUID_TEXT,
                ),
                validation_log_container,
            ],
        ),
    )

    def _build_validation_view() -> ft.Control:
        nonlocal validation_main_content_ref, validation_paths_panel_ref, validation_preview_panel_ref
        cached_view = lazy_tab_view_cache.get("validation")
        if cached_view is not None:
            return cached_view

        validation_paths_panel = _glass_panel(
            padding=18,
            content=ft.Column(
                spacing=12,
                controls=[
                    ft.Text(
                        "Validation Setup",
                        size=14,
                        weight=ft.FontWeight.W_600,
                        color=LIQUID_TEXT,
                    ),
                    ft.ResponsiveRow(
                        columns=12,
                        run_spacing=12,
                        spacing=12,
                        controls=[
                            ft.Container(
                                col={"xs": 12, "sm": 6, "lg": 4},
                                padding=14,
                                border_radius=20,
                                bgcolor=ft.Colors.with_opacity(0.62, LIQUID_SURFACE_ALT),
                                border=ft.border.all(1, ft.Colors.with_opacity(0.30, ft.Colors.WHITE)),
                                content=ft.Column(
                                    spacing=10,
                                    tight=True,
                                    controls=[
                                        ft.Text(
                                            "Dataset",
                                            size=14,
                                            weight=ft.FontWeight.W_600,
                                            color=LIQUID_TEXT,
                                        ),
                                        validation_dataset_field,
                                        validation_dataset_button,
                                    ],
                                ),
                            ),
                            ft.Container(
                                col={"xs": 12, "sm": 6, "lg": 4},
                                padding=14,
                                border_radius=20,
                                bgcolor=ft.Colors.with_opacity(0.62, LIQUID_SURFACE_ALT),
                                border=ft.border.all(1, ft.Colors.with_opacity(0.30, ft.Colors.WHITE)),
                                content=ft.Column(
                                    spacing=10,
                                    tight=True,
                                    controls=[
                                        ft.Text(
                                            "Model",
                                            size=14,
                                            weight=ft.FontWeight.W_600,
                                            color=LIQUID_TEXT,
                                        ),
                                        validation_model_dropdown,
                                        validation_external_model_field,
                                        ft.Row(
                                            wrap=True,
                                            spacing=8,
                                            controls=[
                                                validation_external_model_button,
                                                validation_dropdown_model_button,
                                            ],
                                        ),
                                    ],
                                ),
                            ),
                            ft.Container(
                                col={"xs": 12, "sm": 6, "lg": 4},
                                padding=14,
                                border_radius=20,
                                bgcolor=ft.Colors.with_opacity(0.62, LIQUID_SURFACE_ALT),
                                border=ft.border.all(1, ft.Colors.with_opacity(0.30, ft.Colors.WHITE)),
                                content=ft.Column(
                                    spacing=10,
                                    tight=True,
                                    controls=[
                                        ft.Text(
                                            "Options + Output",
                                            size=14,
                                            weight=ft.FontWeight.W_600,
                                            color=LIQUID_TEXT,
                                        ),
                                        validation_output_dir_field,
                                        validation_output_button,
                                        validation_label_column_field,
                                        validation_mode_dropdown,
                                        validation_target_class_container,
                                    ],
                                ),
                            ),
                        ],
                    ),
                ],
            ),
        )
        validation_paths_panel_ref = validation_paths_panel

        validation_main_content = ft.Container(
            width=1460,
            content=ft.Column(
                expand=True,
                scroll=ft.ScrollMode.AUTO,
                spacing=16,
                controls=[
                    validation_intro_panel,
                    validation_paths_panel,
                    ft.Container(
                        padding=ft.padding.symmetric(horizontal=18, vertical=16),
                        border_radius=22,
                        bgcolor=ft.Colors.with_opacity(0.62, LIQUID_SURFACE_ALT),
                        border=ft.border.all(1, ft.Colors.with_opacity(0.30, ft.Colors.WHITE)),
                        content=ft.Column(
                            spacing=10,
                            controls=[
                                ft.Text(
                                    "Validation Output Preview",
                                    size=14,
                                    weight=ft.FontWeight.W_600,
                                    color=LIQUID_TEXT,
                                ),
                                validation_output_preview,
                            ],
                        ),
                    ),
                    ft.ResponsiveRow(
                        columns=12,
                        run_spacing=16,
                        controls=[
                            ft.Container(
                                col={"xs": 12, "lg": 4},
                                content=_glass_panel(
                                    padding=18,
                                    content=validation_progress_card_body,
                                ),
                            ),
                            ft.Container(
                                col={"xs": 12, "lg": 8},
                                content=_glass_panel(
                                    padding=18,
                                    content=validation_feed_card_body,
                                ),
                            ),
                        ],
                    ),
                ],
            ),
        )
        validation_main_content_ref = validation_main_content
        validation_column = validation_main_content.content
        if isinstance(validation_column, ft.Column) and len(validation_column.controls) >= 3:
            validation_preview_panel_ref = validation_column.controls[2]

        validation_run_badge = ft.Container(
            right=24,
            bottom=20,
            content=_glass_panel(
                padding=10,
                content=ft.Row(
                    spacing=10,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=[
                        validation_spinner,
                        validation_run_button,
                    ],
                ),
            ),
        )

        validation_view = _glass_panel(
            expand=True,
            content=ft.Stack(
                expand=True,
                controls=[
                    ft.Container(
                        expand=True,
                        alignment=ft.Alignment(0, -1),
                        content=validation_main_content,
                    ),
                    validation_run_badge,
                ],
            ),
        )
        lazy_tab_layout_refs["validation"]["main_content"] = validation_main_content
        lazy_tab_layout_refs["validation"]["badge"] = validation_run_badge
        lazy_tab_view_cache["validation"] = validation_view
        return validation_view

    def _build_history_view() -> ft.Control:
        nonlocal history_main_content_ref
        cached_view = lazy_tab_view_cache.get("history")
        if cached_view is not None:
            return cached_view

        history_main_content = ft.Container(
            width=1460,
            padding=ft.padding.only(bottom=90),
            content=ft.Column(
                expand=True,
                scroll=ft.ScrollMode.AUTO,
                spacing=16,
                controls=[
                    ft.ResponsiveRow(
                        columns=12,
                        run_spacing=16,
                        controls=[
                            ft.Container(
                                col={"xs": 12, "lg": 6},
                                content=_glass_panel(
                                    padding=18,
                                    content=ft.Column(
                                        spacing=10,
                                        controls=[
                                            ft.Column(
                                                spacing=4,
                                                controls=[
                                                    ft.Text(
                                                        "History",
                                                        size=14,
                                                        weight=ft.FontWeight.W_600,
                                                        color=LIQUID_TEXT,
                                                    ),
                                                    history_summary,
                                                ],
                                            ),
                                            history_status,
                                            ft.Text(
                                                "Table shows date, model, location, and status. Select a row for full details.",
                                                size=11,
                                                color=LIQUID_MUTED,
                                            ),
                                            history_table_container,
                                        ],
                                    ),
                                ),
                            ),
                            ft.Container(
                                col={"xs": 12, "lg": 6},
                                content=_glass_panel(
                                    padding=18,
                                    content=ft.Column(
                                        spacing=10,
                                        controls=[
                                            ft.Text(
                                                "Run Details",
                                                size=14,
                                                weight=ft.FontWeight.W_600,
                                                color=LIQUID_TEXT,
                                            ),
                                            history_details_container,
                                        ],
                                    ),
                                ),
                            ),
                        ],
                    ),
                ],
            ),
        )
        history_main_content_ref = history_main_content

        history_refresh_badge = ft.Container(
            right=24,
            bottom=20,
            content=_glass_panel(
                padding=10,
                content=ft.Row(
                    spacing=8,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    controls=[history_refresh_button],
                ),
            ),
        )

        history_view = _glass_panel(
            expand=True,
            content=ft.Stack(
                expand=True,
                controls=[
                    ft.Container(
                        expand=True,
                        alignment=ft.Alignment(0, -1),
                        content=history_main_content,
                    ),
                    history_refresh_badge,
                ],
            ),
        )
        lazy_tab_layout_refs["history"]["main_content"] = history_main_content
        lazy_tab_layout_refs["history"]["badge"] = history_refresh_badge
        lazy_tab_view_cache["history"] = history_view
        return history_view

    tab_view_builders: dict[str, Callable[[], ft.Control]] = {
        "apply": lambda: apply_view,
        "train": _build_train_view,
        "validation": _build_validation_view,
        "history": _build_history_view,
    }
    active_view_host = ft.Container(
        expand=True,
        content=apply_view,
    )
    app_status_card.padding = ft.padding.symmetric(horizontal=16, vertical=12)
    app_status_card.border_radius = 22
    app_status_card.content = ft.Row(
        spacing=12,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
        controls=[
            app_status_dot,
            app_status_ring,
            ft.Column(
                spacing=2,
                controls=[
                    app_status_title,
                    app_status,
                ],
            ),
        ],
    )

    active_tab = {"name": "apply"}
    menu_state = {"open": False}
    menu_title = ft.Text(
        "ICE CREAMS Studio",
        size=30,
        weight=ft.FontWeight.W_700,
        color=LIQUID_TEXT,
    )
    menu_subtitle = ft.Text(
        "Intertidal Classification of Europe with a clean desktop workflow for inference, validation, batch processing, and model retraining.",
        size=13,
        color=LIQUID_SUBTEXT,
    )
    menu_shortcuts = ft.Text(
        "Shortcuts: Ctrl+1 Apply, Ctrl+2 Train, Ctrl+3 Validation, Ctrl+4 History, Ctrl+Enter Run",
        size=11,
        color=LIQUID_MUTED,
    )

    apply_tab_label = ft.Text("Apply", size=14, weight=ft.FontWeight.W_600)
    train_tab_label = ft.Text("Train", size=14, weight=ft.FontWeight.W_600)
    validation_tab_label = ft.Text("Validation", size=14, weight=ft.FontWeight.W_600)
    history_tab_label = ft.Text("History", size=14, weight=ft.FontWeight.W_600)

    def _activate_tab_from_menu(tab_name: str) -> None:
        set_active_tab(tab_name, refresh=False)
        set_menu_open(False, refresh=False)
        _apply_responsive_layout()
        _refresh_ui_surface(
            active_view_host,
            side_menu_backdrop,
            side_menu_overlay,
            menu_about_button,
            menu_update_button,
            menu_toggle_badge,
        )

    apply_tab_button = ft.Container(
        border_radius=14,
        padding=ft.padding.symmetric(horizontal=18, vertical=12),
        on_click=lambda _: _activate_tab_from_menu("apply"),
        content=apply_tab_label,
    )
    train_tab_button = ft.Container(
        border_radius=14,
        padding=ft.padding.symmetric(horizontal=18, vertical=12),
        on_click=lambda _: _activate_tab_from_menu("train"),
        content=train_tab_label,
    )
    validation_tab_button = ft.Container(
        border_radius=14,
        padding=ft.padding.symmetric(horizontal=18, vertical=12),
        on_click=lambda _: _activate_tab_from_menu("validation"),
        content=validation_tab_label,
    )
    history_tab_button = ft.Container(
        border_radius=14,
        padding=ft.padding.symmetric(horizontal=18, vertical=12),
        on_click=lambda _: _activate_tab_from_menu("history"),
        content=history_tab_label,
    )
    menu_tab_buttons = [apply_tab_button, train_tab_button, validation_tab_button, history_tab_button]

    menu_toggle_icon = ft.Icon(ft.Icons.MENU, size=20, color=LIQUID_TEXT)
    menu_toggle_title = ft.Text("Menu", size=14, weight=ft.FontWeight.W_700, color=LIQUID_TEXT)
    menu_toggle_indicator = ft.Container(
        width=9,
        height=9,
        border_radius=999,
        bgcolor=LIQUID_ACCENT,
    )
    menu_toggle_button = ft.Container(
        border_radius=16,
        padding=ft.padding.symmetric(horizontal=12, vertical=8),
        bgcolor=ft.Colors.with_opacity(0.62, LIQUID_SURFACE_ALT),
        border=ft.border.all(1, ft.Colors.with_opacity(0.42, ft.Colors.WHITE)),
        content=ft.Row(
            spacing=10,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                menu_toggle_icon,
                menu_toggle_title,
                menu_toggle_indicator,
            ],
        ),
    )
    menu_toggle_badge = ft.Container(
        left=24,
        top=20,
        on_click=lambda _: set_menu_open(not menu_state["open"]),
        content=_glass_panel(
            padding=6,
            content=menu_toggle_button,
        ),
    )
    side_menu_bottom_spacer = ft.Container(height=140)

    side_menu_panel = ft.Container(
        width=360,
        content=_glass_panel(
            expand=True,
            padding=18,
            variant="sidebar",
            accent=LIQUID_ACCENT,
            content=ft.Column(
                spacing=14,
                expand=True,
                controls=[
                    ft.Text("Menu", size=12, color=LIQUID_MUTED, weight=ft.FontWeight.W_600),
                    menu_title,
                    menu_subtitle,
                    app_status_card,
                    ft.Container(
                        height=1,
                        bgcolor=ft.Colors.with_opacity(0.28, ft.Colors.WHITE),
                    ),
                    ft.Text("Workflows", size=12, color=LIQUID_MUTED, weight=ft.FontWeight.W_600),
                    ft.Column(
                        spacing=8,
                        controls=menu_tab_buttons,
                    ),
                    ft.Container(expand=True),
                    menu_shortcuts,
                    ft.Container(
                        height=1,
                        bgcolor=ft.Colors.with_opacity(0.28, ft.Colors.WHITE),
                    ),
                    ft.Text("About", size=12, color=LIQUID_MUTED, weight=ft.FontWeight.W_600),
                    ft.Text(
                        "This work is a UI made by Simon Oiry for the ICE CREAMS algorithm developed by Bede Davies.",
                        size=11,
                        color=LIQUID_SUBTEXT,
                        max_lines=4,
                    ),
                    side_menu_bottom_spacer,
                ],
            ),
        ),
    )
    side_menu_host = ft.Container(
        width=380,
        alignment=ft.Alignment(-1, -1),
        padding=ft.padding.only(left=12, top=0, right=0, bottom=0),
        content=side_menu_panel,
    )
    side_menu_backdrop = ft.Container(
        left=0,
        top=0,
        right=0,
        bottom=0,
        visible=False,
        content=ft.Stack(
            expand=True,
            controls=[
                ft.Container(
                    expand=True,
                    blur=MODAL_SCRIM_BLUR,
                    bgcolor=MODAL_SCRIM_BG,
                ),
                ft.Container(
                    expand=True,
                    bgcolor=ft.Colors.TRANSPARENT,
                    on_click=lambda _: set_menu_open(False),
                ),
            ],
        ),
    )

    side_menu_overlay = ft.Container(
        left=0,
        top=0,
        bottom=0,
        visible=False,
        content=ft.Row(
            spacing=0,
            controls=[side_menu_host],
        ),
    )

    def set_menu_open(is_open: bool, refresh: bool = True) -> None:
        menu_state["open"] = is_open
        side_menu_backdrop.visible = is_open
        side_menu_overlay.visible = is_open
        menu_about_button.visible = is_open
        menu_update_button.visible = is_open
        if not is_open:
            about_popup_blocker.visible = False
        _sync_modal_shell_emphasis()
        menu_toggle_icon.name = ft.Icons.CLOSE if is_open else ft.Icons.MENU
        menu_toggle_title.value = "Menu"
        menu_toggle_indicator.bgcolor = "#41B883" if is_open else LIQUID_ACCENT
        menu_toggle_button.bgcolor = (
            ft.Colors.with_opacity(0.86, LIQUID_ACCENT)
            if is_open
            else ft.Colors.with_opacity(0.62, LIQUID_SURFACE_ALT)
        )
        menu_toggle_button.border = ft.border.all(
            1,
            ft.Colors.with_opacity(0.62 if is_open else 0.42, ft.Colors.WHITE),
        )
        if refresh:
            _refresh_ui_surface(
                shell,
                side_menu_backdrop,
                side_menu_overlay,
                menu_about_button,
                menu_update_button,
                menu_toggle_badge,
                about_popup_blocker,
            )

    def set_active_tab(tab_name: str, refresh: bool = True) -> None:
        resolved_tab_name = tab_name if tab_name in tab_view_builders else "apply"
        active_tab["name"] = resolved_tab_name
        selected_view = tab_view_builders[resolved_tab_name]()
        if resolved_tab_name == "history" and not history_view_initialized["value"]:
            refresh_history_view(force_reload=True, refresh_ui=False)
            history_view_initialized["value"] = True
        if active_view_host.content is not selected_view:
            active_view_host.content = selected_view
        apply_selected = resolved_tab_name == "apply"
        train_selected = resolved_tab_name == "train"
        validation_selected = resolved_tab_name == "validation"
        history_selected = resolved_tab_name == "history"

        apply_tab_button.bgcolor = (
            ft.Colors.with_opacity(0.82, LIQUID_ACCENT)
            if apply_selected
            else ft.Colors.with_opacity(0.56, LIQUID_SURFACE_ALT)
        )
        train_tab_button.bgcolor = (
            ft.Colors.with_opacity(0.82, LIQUID_ACCENT)
            if train_selected
            else ft.Colors.with_opacity(0.56, LIQUID_SURFACE_ALT)
        )
        validation_tab_button.bgcolor = (
            ft.Colors.with_opacity(0.82, LIQUID_ACCENT)
            if validation_selected
            else ft.Colors.with_opacity(0.56, LIQUID_SURFACE_ALT)
        )
        history_tab_button.bgcolor = (
            ft.Colors.with_opacity(0.82, LIQUID_ACCENT)
            if history_selected
            else ft.Colors.with_opacity(0.56, LIQUID_SURFACE_ALT)
        )

        apply_tab_button.border = ft.border.all(
            1,
            ft.Colors.with_opacity(0.52 if apply_selected else 0.30, ft.Colors.WHITE),
        )
        train_tab_button.border = ft.border.all(
            1,
            ft.Colors.with_opacity(0.52 if train_selected else 0.30, ft.Colors.WHITE),
        )
        validation_tab_button.border = ft.border.all(
            1,
            ft.Colors.with_opacity(0.52 if validation_selected else 0.30, ft.Colors.WHITE),
        )
        history_tab_button.border = ft.border.all(
            1,
            ft.Colors.with_opacity(0.52 if history_selected else 0.30, ft.Colors.WHITE),
        )

        apply_tab_label.color = LIQUID_TEXT if apply_selected else LIQUID_SUBTEXT
        train_tab_label.color = LIQUID_TEXT if train_selected else LIQUID_SUBTEXT
        validation_tab_label.color = LIQUID_TEXT if validation_selected else LIQUID_SUBTEXT
        history_tab_label.color = LIQUID_TEXT if history_selected else LIQUID_SUBTEXT
        if refresh:
            _apply_responsive_layout()
            _refresh_ui_surface(active_view_host, side_menu_overlay)

    def _run_active_tab_via_shortcut() -> None:
        if state["busy"]:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        active_name = active_tab["name"]
        if active_name == "apply":
            loop.create_task(run_apply(None))
        elif active_name == "train":
            loop.create_task(run_training(None))
        elif active_name == "validation":
            loop.create_task(run_validation(None))
        elif active_name == "history":
            refresh_history_view(force_reload=True)

    def on_page_keyboard(event: ft.KeyboardEvent) -> None:
        key_value = str(getattr(event, "key", "")).lower().replace(" ", "")
        if key_value in {"escape", "esc"} and history_map_popup_blocker.visible:
            close_history_map_popup(None)
            return
        if key_value in {"escape", "esc"} and update_popup_blocker.visible:
            close_update_popup(None)
            return
        if key_value in {"escape", "esc"} and about_popup_blocker.visible:
            close_about_popup(None)
            return
        if key_value in {"escape", "esc"} and menu_state["open"]:
            set_menu_open(False)
            return
        if not bool(getattr(event, "ctrl", False)):
            return
        if key_value == "1":
            set_active_tab("apply")
            return
        if key_value == "2":
            set_active_tab("train")
            return
        if key_value == "3":
            set_active_tab("validation")
            return
        if key_value == "4":
            set_active_tab("history")
            return
        if key_value in {"enter", "numpadenter", "numenter"}:
            _run_active_tab_via_shortcut()
            return
        if key_value == "r" and active_tab["name"] == "history":
            refresh_history_view(force_reload=True)

    set_active_tab("apply", refresh=False)

    shell_padding = ft.padding.only(left=24, top=76, right=24, bottom=24)
    shell = ft.Container(
        expand=True,
        padding=shell_padding,
        content=active_view_host,
    )

    batch_table_container = ft.Container(
        height=360,
        content=ft.Column(
            scroll=ft.ScrollMode.AUTO,
            controls=[batch_popup_table],
        ),
    )
    validation_table_container = ft.Container(
        height=360,
        content=ft.Column(
            scroll=ft.ScrollMode.AUTO,
            controls=[validation_popup_table],
        ),
    )

    batch_popup_blocker.left = 0
    batch_popup_blocker.top = 0
    batch_popup_blocker.right = 0
    batch_popup_blocker.bottom = 0
    batch_popup_blocker.visible = False
    batch_popup_dialog_container = ft.Container(
        width=1160,
        padding=ft.padding.symmetric(horizontal=12, vertical=8),
        content=_glass_panel(
            padding=24,
            variant="modal",
            accent="#F5A623",
            content=ft.Column(
                spacing=14,
                tight=True,
                controls=[
                    ft.Row(
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        controls=[
                            batch_popup_title,
                            batch_popup_close_button,
                        ],
                    ),
                    batch_popup_summary,
                    batch_popup_formats,
                    batch_popup_dates,
                    batch_popup_warning,
                    ft.Container(
                        padding=12,
                        border_radius=18,
                        bgcolor=ft.Colors.with_opacity(0.62, LIQUID_SURFACE_ALT),
                        content=batch_table_container,
                    ),
                ],
            ),
        ),
    )
    batch_popup_dialog_host = ft.Container(
        expand=True,
        padding=ft.padding.symmetric(horizontal=24, vertical=24),
        alignment=ft.Alignment(0, 0),
        content=batch_popup_dialog_container,
    )

    batch_popup_blocker.content = ft.Stack(
        expand=True,
        controls=[
            ft.Container(
                expand=True,
                blur=MODAL_SCRIM_BLUR,
                bgcolor=MODAL_SCRIM_BG,
            ),
            ft.Container(
                expand=True,
                bgcolor=ft.Colors.TRANSPARENT,
                on_click=close_batch_popup,
            ),
            batch_popup_dialog_host,
        ],
    )

    validation_popup_blocker.left = 0
    validation_popup_blocker.top = 0
    validation_popup_blocker.right = 0
    validation_popup_blocker.bottom = 0
    validation_popup_blocker.visible = False
    validation_popup_dialog_container = ft.Container(
        width=1160,
        padding=ft.padding.symmetric(horizontal=12, vertical=8),
        content=_glass_panel(
            padding=24,
            variant="modal",
            accent=LIQUID_ACCENT,
            content=ft.Column(
                spacing=14,
                tight=True,
                controls=[
                    ft.Row(
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        controls=[
                            validation_popup_title,
                            validation_popup_close_button,
                        ],
                    ),
                    validation_popup_accuracy,
                    validation_popup_note,
                    ft.Text(
                        "Confusion matrix (rows = true labels, columns = predicted labels)",
                        size=12,
                        color=LIQUID_SUBTEXT,
                    ),
                    ft.Container(
                        padding=12,
                        border_radius=18,
                        bgcolor=ft.Colors.with_opacity(0.62, LIQUID_SURFACE_ALT),
                        content=validation_table_container,
                    ),
                ],
            ),
        ),
    )
    validation_popup_dialog_host = ft.Container(
        expand=True,
        padding=ft.padding.symmetric(horizontal=24, vertical=24),
        alignment=ft.Alignment(0, 0),
        content=validation_popup_dialog_container,
    )
    validation_popup_blocker.content = ft.Stack(
        expand=True,
        controls=[
            ft.Container(
                expand=True,
                blur=MODAL_SCRIM_BLUR,
                bgcolor=MODAL_SCRIM_BG,
            ),
            ft.Container(
                expand=True,
                bgcolor=ft.Colors.TRANSPARENT,
                on_click=close_validation_popup,
            ),
            validation_popup_dialog_host,
        ],
    )

    history_map_popup_blocker.left = 0
    history_map_popup_blocker.top = 0
    history_map_popup_blocker.right = 0
    history_map_popup_blocker.bottom = 0
    history_map_popup_blocker.visible = False
    history_map_popup_dialog_container = ft.Container(
        width=1240,
        padding=ft.padding.symmetric(horizontal=12, vertical=8),
        content=_glass_panel(
            padding=24,
            variant="modal",
            accent=LIQUID_ACCENT,
            content=ft.Column(
                spacing=14,
                tight=True,
                controls=[
                    ft.Row(
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        controls=[
                            history_map_popup_title,
                            history_map_popup_close_button,
                        ],
                    ),
                    history_map_popup_summary,
                    history_map_popup_bounds,
                    ft.Row(
                        alignment=ft.MainAxisAlignment.START,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        controls=[history_map_popup_external_button],
                    ),
                    ft.Container(
                        padding=8,
                        border_radius=18,
                        bgcolor=ft.Colors.with_opacity(0.62, LIQUID_SURFACE_ALT),
                        content=history_map_popup_content_host,
                    ),
                    ft.Text(
                        "The polygon is reconstructed from the four WGS84 extent corners stored in the run history JSON.",
                        size=12,
                        color=LIQUID_SUBTEXT,
                    ),
                ],
            ),
        ),
    )
    history_map_popup_dialog_host = ft.Container(
        expand=True,
        padding=ft.padding.symmetric(horizontal=24, vertical=24),
        alignment=ft.Alignment(0, 0),
        content=history_map_popup_dialog_container,
    )
    history_map_popup_blocker.content = ft.Stack(
        expand=True,
        controls=[
            ft.Container(
                expand=True,
                blur=MODAL_SCRIM_BLUR,
                bgcolor=MODAL_SCRIM_BG,
            ),
            ft.Container(
                expand=True,
                bgcolor=ft.Colors.TRANSPARENT,
                on_click=close_history_map_popup,
            ),
            history_map_popup_dialog_host,
        ],
    )

    update_popup_blocker.left = 0
    update_popup_blocker.top = 0
    update_popup_blocker.right = 0
    update_popup_blocker.bottom = 0
    update_popup_blocker.visible = False
    update_popup_dialog_container = ft.Container(
        width=760,
        padding=ft.padding.symmetric(horizontal=12, vertical=8),
        content=_glass_panel(
            padding=24,
            variant="modal",
            accent="#41B883",
            content=ft.Column(
                spacing=14,
                tight=True,
                controls=[
                    ft.Row(
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        controls=[
                            ft.Column(
                                spacing=2,
                                controls=[
                                    update_popup_title,
                                    ft.Text(
                                        "Hosted from the ICE_CREAMS_STUDIO repository",
                                        size=12,
                                        color=LIQUID_SUBTEXT,
                                    ),
                                ],
                            ),
                            update_popup_close_button,
                        ],
                    ),
                    update_popup_summary,
                    ft.Container(
                        padding=16,
                        border_radius=18,
                        bgcolor=ft.Colors.with_opacity(0.62, LIQUID_SURFACE_ALT),
                        border=ft.border.all(1, ft.Colors.with_opacity(0.28, ft.Colors.WHITE)),
                        content=update_popup_details,
                    ),
                    update_popup_status,
                    ft.Row(
                        wrap=True,
                        spacing=10,
                        run_spacing=10,
                        alignment=ft.MainAxisAlignment.END,
                        controls=[
                            update_popup_release_notes_button,
                            update_popup_later_button,
                            update_popup_install_button,
                        ],
                    ),
                ],
            ),
        ),
    )
    update_popup_dialog_host = ft.Container(
        expand=True,
        padding=ft.padding.symmetric(horizontal=24, vertical=24),
        alignment=ft.Alignment(0, 0),
        content=update_popup_dialog_container,
    )
    update_popup_blocker.content = ft.Stack(
        expand=True,
        controls=[
            ft.Container(
                expand=True,
                blur=MODAL_SCRIM_BLUR,
                bgcolor=MODAL_SCRIM_BG_STRONG,
            ),
            ft.Container(
                expand=True,
                bgcolor=ft.Colors.TRANSPARENT,
                on_click=close_update_popup,
            ),
            update_popup_dialog_host,
        ],
    )

    overlay_blocker.left = 0
    overlay_blocker.top = 0
    overlay_blocker.right = 0
    overlay_blocker.bottom = 0
    overlay_blocker.visible = False
    overlay_dialog_container = ft.Container(
        width=980,
        padding=ft.padding.symmetric(horizontal=12, vertical=8),
        content=_glass_panel(
            padding=26,
            variant="modal",
            accent="#F5A623",
            content=ft.Column(
                spacing=16,
                tight=True,
                controls=[
                    ft.Row(
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        controls=[
                            ft.Row(
                                spacing=14,
                                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                                controls=[
                                    overlay_spinner,
                                    ft.Column(
                                        spacing=4,
                                        tight=True,
                                        controls=[
                                            overlay_title,
                                            overlay_counter,
                                        ],
                                    ),
                                ],
                            ),
                            ft.Container(
                                padding=ft.padding.symmetric(horizontal=12, vertical=8),
                                border_radius=999,
                                bgcolor=ft.Colors.with_opacity(0.34, LIQUID_SURFACE_ALT),
                                content=overlay_percent,
                            ),
                        ],
                    ),
                    overlay_progress,
                    ft.Container(
                        padding=16,
                        border_radius=18,
                        bgcolor=ft.Colors.with_opacity(0.62, LIQUID_SURFACE_ALT),
                        content=ft.Column(
                            spacing=6,
                            tight=True,
                            controls=[
                                overlay_step_label,
                                overlay_detail,
                            ],
                        ),
                    ),
                    ft.ResponsiveRow(
                        columns=12,
                        spacing=12,
                        run_spacing=12,
                        controls=[
                            ft.Container(
                                col={"xs": 12, "md": 12},
                                padding=14,
                                border_radius=18,
                                bgcolor=ft.Colors.with_opacity(0.62, LIQUID_SURFACE_ALT),
                                content=ft.Column(
                                    spacing=6,
                                    tight=True,
                                    controls=[
                                        overlay_job_label,
                                        overlay_job_value,
                                    ],
                                ),
                            ),
                            ft.Container(
                                col={"xs": 12, "md": 6},
                                padding=14,
                                border_radius=18,
                                bgcolor=ft.Colors.with_opacity(0.62, LIQUID_SURFACE_ALT),
                                content=ft.Column(
                                    spacing=6,
                                    tight=True,
                                    controls=[
                                        overlay_input_label,
                                        overlay_input_value,
                                    ],
                                ),
                            ),
                            ft.Container(
                                col={"xs": 12, "md": 6},
                                padding=14,
                                border_radius=18,
                                bgcolor=ft.Colors.with_opacity(0.62, LIQUID_SURFACE_ALT),
                                content=ft.Column(
                                    spacing=6,
                                    tight=True,
                                    controls=[
                                        overlay_output_label,
                                        overlay_output_value,
                                    ],
                                ),
                            ),
                        ],
                    ),
                    ft.Text(
                        "Inputs are locked while the current task is running.",
                        size=12,
                        color=LIQUID_SUBTEXT,
                    ),
                ],
            ),
        ),
    )
    overlay_dialog_host = ft.Container(
        expand=True,
        padding=ft.padding.symmetric(horizontal=24, vertical=24),
        alignment=ft.Alignment(0, 0),
        content=overlay_dialog_container,
    )

    overlay_blocker.content = ft.Stack(
        expand=True,
        controls=[
            ft.Container(
                expand=True,
                blur=MODAL_SCRIM_BLUR,
                bgcolor=MODAL_SCRIM_BG_STRONG,
            ),
            overlay_dialog_host,
        ],
    )

    about_popup_blocker.left = 0
    about_popup_blocker.top = 0
    about_popup_blocker.right = 0
    about_popup_blocker.bottom = 0
    about_popup_blocker.visible = False
    about_popup_dialog_host = ft.Container(
        expand=True,
        padding=ft.padding.symmetric(horizontal=24, vertical=24),
        alignment=ft.Alignment(0, 0),
        content=about_popup_dialog_container,
    )
    about_popup_blocker.content = ft.Stack(
        expand=True,
        controls=[
            ft.Container(
                expand=True,
                blur=MODAL_SCRIM_BLUR,
                bgcolor=MODAL_SCRIM_BG_STRONG,
            ),
            ft.Container(
                expand=True,
                bgcolor=ft.Colors.TRANSPARENT,
                on_click=close_about_popup,
            ),
            about_popup_dialog_host,
        ],
    )

    def _viewport_dimension(raw_value: object, fallback: int) -> int:
        try:
            value = int(float(raw_value))
        except (TypeError, ValueError):
            return fallback
        return value if value > 0 else fallback

    def _current_viewport_size() -> tuple[int, int]:
        window_width = getattr(page.window, "width", None)
        window_height = getattr(page.window, "height", None)
        return (
            _viewport_dimension(page.width or window_width, 1600),
            _viewport_dimension(page.height or window_height, 900),
        )

    def _device_pixel_ratio() -> float:
        """Return the display device-pixel ratio (Windows zoom level / 100).

        Flet reports page.width / page.height in *logical* pixels, which are
        physical pixels divided by this ratio.  At 200 % Windows scaling a
        1920-pixel-wide screen only has 960 logical pixels, which would
        incorrectly trigger the compact-layout breakpoints.  Multiplying by
        the ratio before comparing against breakpoints restores the intended
        physical-pixel behaviour while leaving all control sizes in logical
        pixels (Flutter DPI-scales those automatically).
        """
        try:
            dpr = float(
                getattr(page, "device_pixel_ratio", None)
                or getattr(page.window, "device_pixel_ratio", None)
                or 1.0
            )
            return max(0.5, dpr)
        except Exception:
            return 1.0

    def _apply_responsive_layout() -> None:
        viewport_width, viewport_height = _current_viewport_size()
        dpr = _device_pixel_ratio()

        # Compare against breakpoints using physical-equivalent pixels so that
        # layout decisions are independent of the Windows display-scaling level.
        effective_width = int(viewport_width * dpr)
        effective_height = int(viewport_height * dpr)
        compact_width = effective_width < 1380
        compact_height = effective_height < 980
        tight_height = effective_height < 860
        compact_layout = compact_width or compact_height

        horizontal_padding = max(
            8,
            min(24 if compact_layout else 28, int(viewport_width * (0.014 if compact_layout else 0.018))),
        )
        vertical_padding = max(
            8,
            min(18 if compact_layout else 22, int(viewport_height * (0.014 if compact_layout else 0.018))),
        )
        top_padding = max(
            48 if compact_layout else 64,
            min(78 if compact_layout else 96, int(viewport_height * (0.062 if compact_layout else 0.09))),
        )
        shell.padding = ft.padding.only(
            left=horizontal_padding,
            top=top_padding,
            right=horizontal_padding,
            bottom=vertical_padding,
        )
        content_width = max(
            320,
            min(1400 if compact_layout else 1460, viewport_width - (horizontal_padding * 2) - 10),
        )
        section_spacing = 12 if compact_layout else 16
        panel_padding = 14 if compact_layout else 18
        panel_padding_vertical = 12 if compact_layout else 16
        main_bottom_padding = 52 if tight_height else (68 if compact_layout else 90)
        apply_main_content.width = content_width
        apply_main_content.padding = ft.padding.only(bottom=main_bottom_padding)
        if isinstance(apply_main_content.content, ft.Column):
            apply_main_content.content.spacing = section_spacing
        apply_intro_panel.padding = panel_padding
        apply_paths_panel.padding = panel_padding
        if isinstance(apply_paths_panel.content, ft.Column):
            apply_paths_panel.content.spacing = 10 if compact_layout else 12
        apply_model_panel.padding = ft.padding.symmetric(
            horizontal=panel_padding,
            vertical=panel_padding_vertical,
        )
        if isinstance(apply_model_panel.content, ft.Column):
            apply_model_panel.content.spacing = 8 if compact_layout else 10
        for tab_name in ("train", "validation", "history"):
            tab_main_content = lazy_tab_layout_refs[tab_name]["main_content"]
            if tab_main_content is not None:
                tab_main_content.width = content_width
        train_intro_panel.padding = panel_padding
        validation_intro_panel.padding = panel_padding

        if train_main_content_ref is not None:
            train_main_content_ref.padding = ft.padding.only(bottom=main_bottom_padding)
            if isinstance(train_main_content_ref.content, ft.Column):
                train_main_content_ref.content.spacing = section_spacing
        if train_paths_panel_ref is not None:
            train_paths_panel_ref.padding = panel_padding
            if isinstance(train_paths_panel_ref.content, ft.Column):
                train_paths_panel_ref.content.spacing = 10 if compact_layout else 12
                if len(train_paths_panel_ref.content.controls) >= 2:
                    responsive_row = train_paths_panel_ref.content.controls[1]
                    if isinstance(responsive_row, ft.ResponsiveRow):
                        train_paths_card_height_dynamic = max(
                            196,
                            min(232 if compact_layout else 248, int(viewport_height * (0.22 if compact_layout else 0.26))),
                        )
                        for card in responsive_row.controls:
                            if isinstance(card, ft.Container):
                                card.height = train_paths_card_height_dynamic
        if isinstance(train_settings_panel_ref, ft.Container):
            train_settings_panel_ref.padding = ft.padding.symmetric(
                horizontal=panel_padding,
                vertical=panel_padding_vertical,
            )
            if isinstance(train_settings_panel_ref.content, ft.Column):
                train_settings_panel_ref.content.spacing = 10 if compact_layout else 14

        if validation_main_content_ref is not None:
            validation_main_content_ref.padding = ft.padding.only(bottom=main_bottom_padding)
            if isinstance(validation_main_content_ref.content, ft.Column):
                validation_main_content_ref.content.spacing = section_spacing
        if validation_paths_panel_ref is not None:
            validation_paths_panel_ref.padding = panel_padding
            if isinstance(validation_paths_panel_ref.content, ft.Column):
                validation_paths_panel_ref.content.spacing = 10 if compact_layout else 12
        if isinstance(validation_preview_panel_ref, ft.Container):
            validation_preview_panel_ref.padding = ft.padding.symmetric(
                horizontal=panel_padding,
                vertical=panel_padding_vertical,
            )
            if isinstance(validation_preview_panel_ref.content, ft.Column):
                validation_preview_panel_ref.content.spacing = 8 if compact_layout else 10

        if history_main_content_ref is not None:
            history_main_content_ref.padding = ft.padding.only(bottom=main_bottom_padding)
            if isinstance(history_main_content_ref.content, ft.Column):
                history_main_content_ref.content.spacing = section_spacing

        menu_host_padding = max(8, min(20, int(viewport_width * 0.012)))
        side_menu_host.padding = ft.padding.only(
            left=menu_host_padding,
            top=0,
            right=0,
            bottom=0,
        )
        menu_width = max(280, min(430, int(viewport_width * 0.30)))
        side_menu_host.width = menu_width + menu_host_padding
        side_menu_host.height = viewport_height
        side_menu_panel.width = menu_width
        side_menu_panel.height = viewport_height
        side_menu_overlay.height = viewport_height
        side_menu_bottom_spacer.height = max(96, int(viewport_height * 0.14))
        menu_tab_width = max(220, menu_width - 36)
        for menu_tab_button in menu_tab_buttons:
            menu_tab_button.width = menu_tab_width
        menu_toggle_badge.left = 10 if viewport_width < 900 else 20
        menu_toggle_badge.top = max(8, top_padding - 54)
        menu_about_button.right = 10 if viewport_width < 900 else 20
        menu_about_button.top = menu_toggle_badge.top
        menu_about_button.visible = menu_state["open"]
        menu_update_button.right = menu_about_button.right
        menu_update_button.top = menu_about_button.top + 56
        menu_update_button.visible = menu_state["open"]

        popup_padding_h = max(10, min(24, int(viewport_width * 0.018)))
        popup_padding_v = max(10, min(22, int(viewport_height * 0.018)))
        batch_popup_dialog_host.padding = ft.padding.symmetric(
            horizontal=popup_padding_h, vertical=popup_padding_v
        )
        validation_popup_dialog_host.padding = ft.padding.symmetric(
            horizontal=popup_padding_h, vertical=popup_padding_v
        )
        history_map_popup_dialog_host.padding = ft.padding.symmetric(
            horizontal=popup_padding_h, vertical=popup_padding_v
        )
        update_popup_dialog_host.padding = ft.padding.symmetric(
            horizontal=popup_padding_h, vertical=popup_padding_v
        )
        overlay_dialog_host.padding = ft.padding.symmetric(
            horizontal=popup_padding_h, vertical=popup_padding_v
        )
        about_popup_dialog_host.padding = ft.padding.symmetric(
            horizontal=popup_padding_h, vertical=popup_padding_v
        )
        batch_popup_dialog_container.width = max(
            340, min(1160, viewport_width - (popup_padding_h * 2) - 12)
        )
        validation_popup_dialog_container.width = max(
            340, min(1160, viewport_width - (popup_padding_h * 2) - 12)
        )
        history_map_popup_dialog_container.width = max(
            360, min(1240, viewport_width - (popup_padding_h * 2) - 12)
        )
        update_popup_dialog_container.width = max(
            340, min(820, viewport_width - (popup_padding_h * 2) - 12)
        )
        overlay_dialog_container.width = max(
            320, min(980, viewport_width - (popup_padding_h * 2) - 12)
        )
        about_popup_dialog_container.width = max(
            360, min(1240, viewport_width - (popup_padding_h * 2) - 12)
        )
        history_map_popup_content_host.height = max(
            300, min(620, viewport_height - (popup_padding_v * 2) - 220)
        )

        cards_height = max(
            150 if compact_layout else 170,
            min(220 if compact_layout else 240, int(viewport_height * (0.20 if compact_layout else 0.24))),
        )
        apply_preview_card_body.height = cards_height
        apply_feed_card_body.height = cards_height
        train_progress_card_body.height = cards_height
        train_feed_card_body.height = cards_height
        validation_progress_card_body.height = cards_height
        validation_feed_card_body.height = cards_height

        train_log_container.height = max(80, cards_height - 46)
        validation_log_container.height = max(80, cards_height - 46)
        batch_table_container.height = max(180, min(400, int(viewport_height * 0.40)))
        validation_table_container.height = max(180, min(400, int(viewport_height * 0.40)))
        history_panel_height = max(
            300 if compact_layout else 360,
            min(680 if compact_layout else 720, int(viewport_height * (0.64 if compact_layout else 0.72))),
        )
        history_table_container.height = history_panel_height
        history_details_container.height = history_panel_height

        apply_run_badge.right = 12 if viewport_width < 900 else 24
        apply_run_badge.bottom = 12 if viewport_height < 760 else 20
        for tab_name in ("train", "validation", "history"):
            tab_badge = lazy_tab_layout_refs[tab_name]["badge"]
            if tab_badge is not None:
                tab_badge.right = 12 if viewport_width < 900 else 24
                tab_badge.bottom = 12 if viewport_height < 760 else 20

    async def _stabilize_initial_layout() -> None:
        last_size = (-1, -1)
        for delay_seconds in (0.05, 0.20, 0.75):
            await asyncio.sleep(delay_seconds)
            current_size = _current_viewport_size()
            if current_size == last_size:
                continue
            last_size = current_size
            _apply_responsive_layout()
            request_ui_refresh(force=True)

    def on_page_resize(_: ft.ControlEvent | None = None) -> None:
        _apply_responsive_layout()
        request_ui_refresh()

    page.on_resize = on_page_resize
    page.on_media_change = on_page_resize
    page.on_keyboard_event = on_page_keyboard
    _apply_responsive_layout()

    background_controls: list[ft.Control] = [
        ft.Container(
            expand=True,
            gradient=ft.LinearGradient(
                begin=ft.Alignment(-1, -1),
                end=ft.Alignment(1, 1),
                colors=[
                    "#E9F3FF",
                    "#D9E9FB",
                    "#D3E4F9",
                    "#EDF5FF",
                ],
            ),
        ),
        ft.Container(
            expand=True,
            bgcolor=ft.Colors.with_opacity(0.12, "#FFFFFF"),
        ),
    ]
    if SHOW_AMBIENT_BACKGROUND_BLOBS:
        background_controls.extend(
            [
                ft.Container(
                    width=360,
                    height=360,
                    left=-90,
                    top=60,
                    border_radius=999,
                    blur=(140, 140),
                    bgcolor=ft.Colors.with_opacity(0.12, LIQUID_ACCENT),
                ),
                ft.Container(
                    width=300,
                    height=300,
                    right=-40,
                    top=120,
                    border_radius=999,
                    blur=(130, 130),
                    bgcolor=ft.Colors.with_opacity(0.10, "#A5C8FF"),
                ),
                ft.Container(
                    width=420,
                    height=420,
                    right=120,
                    bottom=-180,
                    border_radius=999,
                    blur=(165, 165),
                    bgcolor=ft.Colors.with_opacity(0.10, "#FFD9B8"),
                ),
                ft.Container(
                    width=280,
                    height=280,
                    left=120,
                    bottom=-130,
                    border_radius=999,
                    blur=(150, 150),
                    bgcolor=ft.Colors.with_opacity(0.10, "#B5D6FF"),
                ),
            ]
        )

    page.add(
        ft.Stack(
            expand=True,
            controls=[
                *background_controls,
                shell,
                menu_toggle_badge,
                side_menu_backdrop,
                side_menu_overlay,
                menu_about_button,
                menu_update_button,
                about_popup_blocker,
                update_popup_blocker,
                batch_popup_blocker,
                validation_popup_blocker,
                history_map_popup_blocker,
                overlay_blocker,
            ],
        )
    )

    set_app_status("ready", "Ready for a new run.")
    refresh_apply_preview()
    refresh_apply_run_button_state()
    refresh_validation_preview()
    append_log(apply_log, "Ready. Select inputs and start the apply workflow.")
    append_log(
        train_log,
        "Ready. Select training CSV files and/or folders containing CSV files to train a new model.",
    )
    append_log(
        validation_log,
        "Ready. Select validation dataset and model to export predictions and metrics CSV files.",
    )
    page.update()
    page.run_task(_stabilize_initial_layout)
    _close_startup_splash()
    page.run_task(_run_startup_update_check)


if __name__ == "__main__":
    ft.run(main)
