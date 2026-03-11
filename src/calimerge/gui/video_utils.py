"""
Video file discovery and encoding utilities.

Handles both new naming (port{N}_{serial}.mp4) and legacy naming (port_N.mp4).
Provides hardware-accelerated video encoding via ffmpeg with cv2 fallback.
"""

from __future__ import annotations

import re
import subprocess
import shutil
from pathlib import Path
from typing import Literal
from dataclasses import dataclass


def find_video_for_port(
    folder: Path,
    port: int,
    serial: str | None = None,
    extensions: tuple[str, ...] = ("mp4", "avi", "mov"),
) -> Path | None:
    """
    Find a video file for a given camera port, with backwards-compatible matching.

    Search order:
    1. New format with exact serial: port_{N}_{sanitized_serial}.ext
    2. Legacy format: port_{N}.ext
    3. New format with any serial: port_{N}_*.ext

    Returns the first match found, or None.
    """
    for ext in extensions:
        # 1. New format, exact serial match
        if serial:
            sanitized = serial.replace("&", "-")
            path = folder / f"port_{port}_{sanitized}.{ext}"
            if path.exists():
                return path

        # 2. Legacy format
        path = folder / f"port_{port}.{ext}"
        if path.exists():
            return path

        # 3. New format, any serial
        matches = sorted(folder.glob(f"port_{port}_*.{ext}"))
        if matches:
            return matches[0]

    return None


def discover_videos(
    folder: Path,
    cameras: dict | None = None,
    extensions: tuple[str, ...] = ("mp4", "avi", "mov"),
) -> dict[int, Path]:
    """
    Discover all video files in a folder, mapping port numbers to paths.

    If cameras dict is provided (port -> CameraState), uses serial numbers
    for matching. Otherwise, discovers all port-numbered videos in the folder.

    Returns dict of port -> video_path.
    """
    result: dict[int, Path] = {}

    if cameras:
        # Match using known cameras (prefer serial-based matching)
        for port, cam_state in cameras.items():
            serial = getattr(cam_state.info, "serial_number", None)
            path = find_video_for_port(folder, port, serial, extensions)
            if path:
                result[port] = path
    else:
        # Discover all videos in folder (both new and legacy formats)
        for ext in extensions:
            for video_file in folder.glob(f"port_*.{ext}"):
                parsed = parse_video_filename(video_file.name)
                if parsed is not None:
                    port, _ = parsed
                    if port not in result:
                        result[port] = video_file

    return result


def parse_video_filename(filename: str) -> tuple[int, str] | None:
    """
    Parse a video filename to extract port number and serial.

    Handles:
    - New format: "port_{N}_{serial}.mp4" -> (N, serial)
    - Legacy format: "port_{N}.mp4" -> (N, "")

    Returns (port, serial) or None if not a valid port video filename.
    """
    stem = Path(filename).stem

    # New format: port_0_7-1b959837-0-0000
    m = re.match(r"^port_(\d+)_(.+)$", stem)
    if m:
        return int(m.group(1)), m.group(2)

    # Legacy format: port_0
    m = re.match(r"^port_(\d+)$", stem)
    if m:
        return int(m.group(1)), ""

    return None


# ============================================================================
# Video Encoding with Hardware Acceleration
# ============================================================================

@dataclass
class EncoderInfo:
    """Information about available encoders."""
    ffmpeg_path: str | None
    has_h264_hw: bool
    has_hevc_hw: bool
    has_prores_hw: bool
    h264_hw_encoder: str | None = None   # e.g. "h264_nvenc", "h264_amf", "h264_qsv"
    hevc_hw_encoder: str | None = None
    has_libx264: bool = False


_encoder_info: EncoderInfo | None = None


def detect_encoders() -> EncoderInfo:
    """Detect available video encoders (cached)."""
    global _encoder_info
    if _encoder_info is not None:
        return _encoder_info

    ffmpeg_path = shutil.which("ffmpeg")
    has_h264_hw = False
    has_hevc_hw = False
    has_prores_hw = False
    h264_hw_encoder = None
    hevc_hw_encoder = None
    has_libx264 = False

    if ffmpeg_path:
        try:
            result = subprocess.run(
                [ffmpeg_path, "-encoders"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            output = result.stdout + result.stderr

            # H.264 hardware: check all platform encoders (prefer order)
            for enc in ("h264_nvenc", "h264_amf", "h264_qsv", "h264_videotoolbox"):
                if enc in output:
                    has_h264_hw = True
                    h264_hw_encoder = enc
                    break

            # HEVC hardware
            for enc in ("hevc_nvenc", "hevc_amf", "hevc_qsv", "hevc_videotoolbox"):
                if enc in output:
                    has_hevc_hw = True
                    hevc_hw_encoder = enc
                    break

            has_prores_hw = "prores_videotoolbox" in output
            has_libx264 = "libx264" in output
        except Exception:
            pass

    _encoder_info = EncoderInfo(
        ffmpeg_path=ffmpeg_path,
        has_h264_hw=has_h264_hw,
        has_hevc_hw=has_hevc_hw,
        has_prores_hw=has_prores_hw,
        h264_hw_encoder=h264_hw_encoder,
        hevc_hw_encoder=hevc_hw_encoder,
        has_libx264=has_libx264,
    )
    return _encoder_info


Codec = Literal["h264", "hevc", "prores", "mpeg4"]


# ── Video writer handle (plain struct, no methods) ──

@dataclass
class VideoWriterHandle:
    """Opaque handle for an active video writer."""
    kind: Literal["ffmpeg", "cv2"]
    process: subprocess.Popen | None = None   # ffmpeg only
    cv2_writer: object | None = None          # cv2 only
    frame_count: int = 0


def _pick_encoder(codec: Codec, bitrate: str) -> tuple[str, list[str]]:
    """Pick best available encoder + args for the given codec."""
    info = detect_encoders()

    if codec == "h264":
        if info.has_h264_hw and info.h264_hw_encoder:
            return info.h264_hw_encoder, ["-b:v", bitrate]
        elif info.has_libx264:
            return "libx264", ["-preset", "fast", "-crf", "23"]
        else:
            return "mpeg4", ["-q:v", "5"]

    elif codec == "hevc":
        if info.has_hevc_hw and info.hevc_hw_encoder:
            return info.hevc_hw_encoder, ["-b:v", bitrate]
        else:
            return "libx265", ["-preset", "fast", "-crf", "28"]

    elif codec == "prores":
        if info.has_prores_hw:
            return "prores_videotoolbox", ["-profile:v", "0"]
        else:
            return "prores_ks", ["-profile:v", "0"]

    else:  # mpeg4 fallback
        return "mpeg4", ["-q:v", "5"]


def _open_ffmpeg_writer(
    output_path: Path, width: int, height: int, fps: int,
    codec: Codec, bitrate: str,
    metadata: dict[str, str] | None = None,
) -> VideoWriterHandle:
    """Start an ffmpeg subprocess that accepts raw BGR24 on stdin."""
    info = detect_encoders()
    if not info.ffmpeg_path:
        raise RuntimeError("ffmpeg not found")

    encoder, encoder_args = _pick_encoder(codec, bitrate)

    metadata_args = []
    if metadata:
        for key, value in metadata.items():
            metadata_args.extend(["-metadata", f"{key}={value}"])

    cmd = [
        info.ffmpeg_path,
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", encoder,
        *encoder_args,
        "-pix_fmt", "yuv420p",
        *metadata_args,
        str(output_path),
    ]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return VideoWriterHandle(kind="ffmpeg", process=proc)


def _open_cv2_writer(
    output_path: Path, width: int, height: int, fps: int,
) -> VideoWriterHandle:
    """Open an OpenCV VideoWriter using mp4v (always available, no openh264 needed)."""
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    return VideoWriterHandle(kind="cv2", cv2_writer=writer)


def create_video_writer(
    output_path: Path,
    width: int,
    height: int,
    fps: int,
    codec: Codec = "h264",
    bitrate: str = "8M",
    prefer_hardware: bool = True,
    metadata: dict[str, str] | None = None,
) -> VideoWriterHandle:
    """
    Create a video writer with automatic encoder selection.

    Prefers ffmpeg (hw or sw H.264) when available,
    falls back to cv2 with mp4v (no openh264 dependency).

    metadata: optional dict of key=value pairs embedded in the MP4 container
    (ffmpeg only; ignored for cv2 fallback).
    """
    info = detect_encoders()

    if prefer_hardware and info.ffmpeg_path:
        try:
            return _open_ffmpeg_writer(
                output_path, width, height, fps, codec, bitrate, metadata
            )
        except Exception:
            pass

    return _open_cv2_writer(output_path, width, height, fps)


def write_frame(handle: VideoWriterHandle, frame) -> bool:
    """Write a BGR frame to the video writer. Returns True on success."""
    if handle.kind == "ffmpeg":
        if handle.process is None or handle.process.stdin is None:
            return False
        try:
            data = frame.tobytes() if hasattr(frame, "tobytes") else bytes(frame)
            handle.process.stdin.write(data)
            handle.frame_count += 1
            return True
        except BrokenPipeError:
            return False
    else:
        handle.cv2_writer.write(frame)
        handle.frame_count += 1
        return True


def release_writer(handle: VideoWriterHandle):
    """Close the video writer and free resources."""
    if handle.kind == "ffmpeg":
        if handle.process is not None:
            try:
                if handle.process.stdin:
                    handle.process.stdin.close()
                handle.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                handle.process.kill()
            finally:
                handle.process = None
    else:
        if handle.cv2_writer is not None:
            handle.cv2_writer.release()
            handle.cv2_writer = None
