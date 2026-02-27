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
    1. New format with exact serial: port{N}_{serial}.ext
    2. Legacy format: port_{N}.ext
    3. New format with any serial: port{N}_*.ext

    Returns the first match found, or None.
    """
    for ext in extensions:
        # 1. New format, exact serial match
        if serial:
            path = folder / f"port{port}_{serial}.{ext}"
            if path.exists():
                return path

        # 2. Legacy format
        path = folder / f"port_{port}.{ext}"
        if path.exists():
            return path

        # 3. New format, any serial
        matches = sorted(folder.glob(f"port{port}_*.{ext}"))
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
        # Discover all videos in folder
        for ext in extensions:
            # New format: port{N}_{serial}.ext
            for video_file in folder.glob(f"port*_*.{ext}"):
                parsed = parse_video_filename(video_file.name)
                if parsed is not None:
                    port, _ = parsed
                    if port not in result:
                        result[port] = video_file

            # Legacy format: port_{N}.ext
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
    - New format: "port{N}_{serial}.mp4" -> (N, serial)
    - Legacy format: "port_{N}.mp4" -> (N, "")

    Returns (port, serial) or None if not a valid port video filename.
    """
    stem = Path(filename).stem

    # New format: port0_ABC123DEF
    m = re.match(r"^port(\d+)_(.+)$", stem)
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

    if ffmpeg_path:
        try:
            result = subprocess.run(
                [ffmpeg_path, "-encoders"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            output = result.stdout + result.stderr
            has_h264_hw = "h264_videotoolbox" in output or "h264_nvenc" in output
            has_hevc_hw = "hevc_videotoolbox" in output or "hevc_nvenc" in output
            has_prores_hw = "prores_videotoolbox" in output
        except Exception:
            pass

    _encoder_info = EncoderInfo(
        ffmpeg_path=ffmpeg_path,
        has_h264_hw=has_h264_hw,
        has_hevc_hw=has_hevc_hw,
        has_prores_hw=has_prores_hw,
    )
    return _encoder_info


Codec = Literal["h264", "hevc", "prores", "mpeg4"]


class FFmpegWriter:
    """
    Video writer using ffmpeg subprocess with hardware encoding.

    Frames are piped to ffmpeg's stdin as raw BGR24 data.
    The pipe buffer provides natural decoupling between capture and encoding.
    """

    def __init__(
        self,
        output_path: Path,
        width: int,
        height: int,
        fps: int,
        codec: Codec = "h264",
        bitrate: str = "8M",
    ):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec
        self.bitrate = bitrate
        self.process: subprocess.Popen | None = None
        self.frame_count = 0

        self._start_ffmpeg()

    def _get_encoder(self) -> tuple[str, list[str]]:
        """Get encoder name and extra args based on codec and hardware availability."""
        info = detect_encoders()

        if self.codec == "h264":
            if info.has_h264_hw:
                return "h264_videotoolbox", ["-b:v", self.bitrate]
            else:
                return "libx264", ["-preset", "fast", "-crf", "23"]

        elif self.codec == "hevc":
            if info.has_hevc_hw:
                return "hevc_videotoolbox", ["-b:v", self.bitrate]
            else:
                return "libx265", ["-preset", "fast", "-crf", "28"]

        elif self.codec == "prores":
            if info.has_prores_hw:
                return "prores_videotoolbox", ["-profile:v", "0"]  # Proxy profile
            else:
                return "prores_ks", ["-profile:v", "0"]

        else:  # mpeg4 fallback
            return "mpeg4", ["-q:v", "5"]

    def _start_ffmpeg(self):
        """Start ffmpeg subprocess."""
        info = detect_encoders()
        if not info.ffmpeg_path:
            raise RuntimeError("ffmpeg not found")

        encoder, encoder_args = self._get_encoder()

        cmd = [
            info.ffmpeg_path,
            "-y",  # Overwrite output
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "pipe:0",  # Read from stdin
            "-c:v", encoder,
            *encoder_args,
            "-pix_fmt", "yuv420p",  # Compatibility
            str(self.output_path),
        ]

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame) -> bool:
        """
        Write a frame (numpy array, BGR format).

        Returns True on success, False if pipe is broken.
        """
        if self.process is None or self.process.stdin is None:
            return False

        try:
            # Ensure contiguous array
            if hasattr(frame, "tobytes"):
                data = frame.tobytes()
            else:
                data = bytes(frame)

            self.process.stdin.write(data)
            self.frame_count += 1
            return True
        except BrokenPipeError:
            return False

    def release(self):
        """Close the writer and wait for ffmpeg to finish."""
        if self.process is None:
            return

        try:
            if self.process.stdin:
                self.process.stdin.close()
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
        finally:
            self.process = None


class CV2Writer:
    """Fallback video writer using OpenCV (software encoding)."""

    def __init__(
        self,
        output_path: Path,
        width: int,
        height: int,
        fps: int,
        codec: Codec = "h264",
    ):
        import cv2

        # Map codec to fourcc
        fourcc_map = {
            "h264": "avc1",
            "hevc": "hvc1",
            "mpeg4": "mp4v",
            "prores": "ap4h",
        }
        fourcc_str = fourcc_map.get(codec, "mp4v")
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)

        self.writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height),
        )
        self.frame_count = 0

    def write(self, frame) -> bool:
        """Write a frame."""
        self.writer.write(frame)
        self.frame_count += 1
        return True

    def release(self):
        """Close the writer."""
        self.writer.release()


def create_video_writer(
    output_path: Path,
    width: int,
    height: int,
    fps: int,
    codec: Codec = "h264",
    bitrate: str = "8M",
    prefer_hardware: bool = True,
) -> FFmpegWriter | CV2Writer:
    """
    Create a video writer with automatic encoder selection.

    Prefers ffmpeg with hardware encoding when available,
    falls back to cv2.VideoWriter.
    """
    info = detect_encoders()

    if prefer_hardware and info.ffmpeg_path:
        try:
            return FFmpegWriter(output_path, width, height, fps, codec, bitrate)
        except Exception:
            pass  # Fall through to cv2

    return CV2Writer(output_path, width, height, fps, codec)
