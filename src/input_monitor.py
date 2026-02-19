#!/usr/bin/env python3
"""
Physical Input Monitor - detects laptop keyboard/mouse usage.
When physical input detected, pauses app's simulated input.
Linux: /dev/input/event* monitoring (zero external deps, requires 'input' group).
Win/Mac: pynput listeners with timestamp-based virtual event filtering.
"""

import os
import sys
import struct
import threading
import logging
import time as _time

logger = logging.getLogger("sanketra")

# Linux input event constants
_EV_SYN = 0x00
_EV_KEY = 0x01
_EV_REL = 0x02
_EV_ABS = 0x03
_REL_X = 0x00
_REL_Y = 0x01
_MOUSE_MOVE_THRESHOLD = 5  # units — ignore sensor jitter when mouse is stationary

# struct input_event: struct timeval (long, long) + __u16 type + __u16 code + __s32 value
_INPUT_EVENT_FORMAT = '@llHHi'
_INPUT_EVENT_SIZE = struct.calcsize(_INPUT_EVENT_FORMAT)

# Virtual device name substrings to exclude (our own + common virtual devices)
_VIRTUAL_DEVICE_NAMES = [
    'sanketra',
    'ydotool',
    'virtual',
    'xdotool',
    'pynput',
    'uinput',
]


def _is_virtual_device(name):
    """Check if device name matches known virtual devices"""
    name_lower = name.lower()
    return any(virt in name_lower for virt in _VIRTUAL_DEVICE_NAMES)


def _get_device_name(event_num):
    """Read device name from sysfs"""
    try:
        path = f"/sys/class/input/event{event_num}/device/name"
        with open(path, 'r') as f:
            return f.read().strip()
    except (OSError, IOError):
        return None


def _has_input_capability(event_num):
    """Check if device has keyboard (EV_KEY) or mouse (EV_REL) capability"""
    try:
        path = f"/sys/class/input/event{event_num}/device/capabilities/ev"
        with open(path, 'r') as f:
            caps = int(f.read().strip(), 16)
        # Check EV_KEY (bit 1) or EV_REL (bit 2)
        has_key = bool(caps & (1 << _EV_KEY))
        has_rel = bool(caps & (1 << _EV_REL))
        return has_key or has_rel
    except (OSError, IOError, ValueError):
        return False


class PhysicalInputMonitor:
    """
    Monitors physical keyboard/mouse input on Linux via /dev/input/event*.
    Calls pause_callback() when physical input is detected.
    Daemon thread, zero external dependencies.
    """

    def __init__(self, pause_callback, poll_timeout=0.25):
        self._pause_callback = pause_callback
        self._poll_timeout = poll_timeout
        self._device_fds = []  # list of (fd, path, name)
        self._thread = None
        self._running = False
        self._active = False

    def start(self):
        """Enumerate physical input devices and start monitoring thread"""
        devices = self._enumerate_devices()
        if not devices:
            logger.warning("[InputMonitor] No physical input devices found or accessible")
            return False

        # Open devices
        for event_num, name in devices:
            path = f"/dev/input/event{event_num}"
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
                self._device_fds.append((fd, path, name))
                logger.info(f"[InputMonitor] Monitoring: {name} ({path})")
            except PermissionError:
                logger.warning(f"[InputMonitor] Permission denied: {path} ({name}) — add user to 'input' group")
            except OSError as e:
                logger.warning(f"[InputMonitor] Cannot open {path}: {e}")

        if not self._device_fds:
            logger.warning("[InputMonitor] No devices could be opened — feature disabled")
            return False

        # Start monitor thread
        self._running = True
        self._active = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True, name="input-monitor")
        self._thread.start()
        logger.info(f"[InputMonitor] Started — monitoring {len(self._device_fds)} device(s)")
        return True

    def stop(self):
        """Stop monitoring and close device files"""
        self._running = False
        self._active = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        for fd, path, name in self._device_fds:
            try:
                os.close(fd)
            except OSError:
                pass
        self._device_fds.clear()
        logger.info("[InputMonitor] Stopped")

    def _enumerate_devices(self):
        """Find physical keyboard/mouse devices"""
        devices = []
        try:
            entries = os.listdir('/dev/input/')
        except (OSError, PermissionError):
            logger.warning("[InputMonitor] Cannot read /dev/input/")
            return devices

        for entry in sorted(entries):
            if not entry.startswith('event'):
                continue
            try:
                event_num = int(entry[5:])
            except ValueError:
                continue

            name = _get_device_name(event_num)
            if name is None:
                continue

            # Skip virtual devices
            if _is_virtual_device(name):
                logger.debug(f"[InputMonitor] Skipping virtual: {name} (event{event_num})")
                continue

            # Only monitor devices with keyboard or mouse capability
            if not _has_input_capability(event_num):
                continue

            devices.append((event_num, name))

        return devices

    def _monitor_loop(self):
        """Thread: poll device fds for input events"""
        import select  # Linux-only — not available on Windows
        while self._running:
            if not self._device_fds:
                break

            try:
                fds = [fd for fd, _, _ in self._device_fds]
                readable, _, _ = select.select(fds, [], [], self._poll_timeout)

                for fd in readable:
                    if self._process_events(fd):
                        # Physical input detected — call pause
                        try:
                            self._pause_callback()
                        except Exception as e:
                            logger.debug(f"[InputMonitor] Callback error: {e}")
                        # Drain remaining readable fds without re-triggering
                        for other_fd in readable:
                            if other_fd != fd:
                                self._flush_fd(other_fd)
                        break  # One trigger per select cycle is enough

            except (OSError, ValueError) as e:
                # Device disconnected or fd became invalid
                self._cleanup_dead_fds()
            except Exception:
                pass  # Don't crash on transient errors

    def _process_events(self, fd):
        """Read events from fd, return True if physical input detected"""
        rel_x, rel_y = 0, 0
        try:
            while True:
                data = os.read(fd, _INPUT_EVENT_SIZE)
                if len(data) < _INPUT_EVENT_SIZE:
                    break
                _, _, ev_type, code, value = struct.unpack(_INPUT_EVENT_FORMAT, data)
                if ev_type == _EV_KEY and value != 0:
                    self._flush_fd(fd)
                    return True
                elif ev_type == _EV_REL:
                    if code == _REL_X:
                        rel_x += value
                    elif code == _REL_Y:
                        rel_y += value
                    else:
                        # scroll wheel — trigger immediately
                        self._flush_fd(fd)
                        return True
                elif ev_type == _EV_ABS and value != 0:
                    self._flush_fd(fd)
                    return True
        except BlockingIOError:
            pass
        except OSError:
            return False
        return abs(rel_x) >= _MOUSE_MOVE_THRESHOLD or abs(rel_y) >= _MOUSE_MOVE_THRESHOLD

    def _flush_fd(self, fd):
        """Drain all pending events from fd without processing"""
        try:
            while True:
                data = os.read(fd, _INPUT_EVENT_SIZE * 64)
                if len(data) < _INPUT_EVENT_SIZE:
                    break
        except (BlockingIOError, OSError):
            pass

    def _cleanup_dead_fds(self):
        """Remove disconnected device fds"""
        alive = []
        for fd, path, name in self._device_fds:
            try:
                os.fstat(fd)
                alive.append((fd, path, name))
            except OSError:
                logger.warning(f"[InputMonitor] Device disconnected: {name} ({path})")
                try:
                    os.close(fd)
                except OSError:
                    pass
        self._device_fds = alive

    @property
    def is_active(self):
        return self._active and self._running


# =============================================================================
#                    PYNPUT MONITOR (Windows / macOS)
# =============================================================================

# Timestamp-based virtual event suppression:
# When our app generates input via pynput, mark_app_output() is called.
# The monitor ignores events within _APP_OUTPUT_WINDOW of the last mark.
_last_app_output_time = 0.0
_APP_OUTPUT_WINDOW = 0.15  # 150ms grace period


def mark_app_output():
    """Call before generating input via pynput — suppresses monitor for 150ms."""
    global _last_app_output_time
    _last_app_output_time = _time.time()


def _in_grace_period():
    return _time.time() - _last_app_output_time < _APP_OUTPUT_WINDOW


class PynputInputMonitor:
    """
    Monitors physical keyboard/mouse on Windows/macOS via pynput listeners.
    Uses timestamp grace period to filter out app's own pynput-generated events.
    """

    _MOUSE_MOVE_THRESHOLD = 5  # pixels — ignore jitter / palm brushes

    def __init__(self, pause_callback, debounce=0.25):
        self._pause_callback = pause_callback
        self._debounce = debounce
        self._last_trigger = 0.0
        self._last_mouse_x = 0
        self._last_mouse_y = 0
        self._kb_listener = None
        self._mouse_listener = None
        self._running = False
        self._active = False

    def start(self):
        try:
            from pynput import keyboard, mouse
        except ImportError:
            logger.warning("[InputMonitor] pynput not available — input guard disabled")
            return False

        self._running = True
        self._active = True

        self._kb_listener = keyboard.Listener(
            on_press=self._on_key_event,
            on_release=self._on_key_event,
        )
        self._kb_listener.daemon = True
        self._kb_listener.start()

        self._mouse_listener = mouse.Listener(
            on_move=self._on_mouse_move,
            on_click=self._on_mouse_click,
            on_scroll=self._on_mouse_scroll,
        )
        self._mouse_listener.daemon = True
        self._mouse_listener.start()

        logger.info("[InputMonitor] Started pynput monitor (keyboard + mouse)")
        return True

    def stop(self):
        self._running = False
        self._active = False
        if self._kb_listener:
            try:
                self._kb_listener.stop()
            except Exception:
                pass
        if self._mouse_listener:
            try:
                self._mouse_listener.stop()
            except Exception:
                pass
        logger.info("[InputMonitor] Stopped pynput monitor")

    def _trigger_pause(self):
        """Debounced pause trigger — max once per debounce interval."""
        if not self._running:
            return
        if _in_grace_period():
            return
        now = _time.time()
        if now - self._last_trigger < self._debounce:
            return
        self._last_trigger = now
        try:
            self._pause_callback()
        except Exception as e:
            logger.debug(f"[InputMonitor] Callback error: {e}")

    def _on_key_event(self, key):
        self._trigger_pause()

    def _on_mouse_move(self, x, y):
        dx = abs(x - self._last_mouse_x)
        dy = abs(y - self._last_mouse_y)
        self._last_mouse_x = x
        self._last_mouse_y = y
        if dx >= self._MOUSE_MOVE_THRESHOLD or dy >= self._MOUSE_MOVE_THRESHOLD:
            self._trigger_pause()

    def _on_mouse_click(self, x, y, button, pressed):
        if pressed:
            self._trigger_pause()

    def _on_mouse_scroll(self, x, y, dx, dy):
        self._trigger_pause()

    @property
    def is_active(self):
        return self._active and self._running


# =============================================================================
#                              MODULE API
# =============================================================================

_monitor = None


def start_monitor(pause_callback):
    """Start physical input monitor. Returns True if started."""
    global _monitor

    if sys.platform == 'linux':
        _monitor = PhysicalInputMonitor(pause_callback)
    else:
        # Windows / macOS — use pynput listeners
        _monitor = PynputInputMonitor(pause_callback)

    started = _monitor.start()
    if not started:
        _monitor = None
    return started


def stop_monitor():
    """Stop the physical input monitor"""
    global _monitor
    if _monitor:
        _monitor.stop()
        _monitor = None


def is_monitor_active():
    """Check if monitor is currently running"""
    return _monitor is not None and _monitor.is_active
