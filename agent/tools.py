"""
LangGraph tools.
Hardware commands go to ESP32 via WebSocket broadcast.
Motor commands go to the robot car via HTTP (ESP32 WebServer on port 80).
YouTube opens on the phone via tool_call emit (not webbrowser on PC).
"""

import logging
import requests
from urllib.parse import quote_plus
from langchain_core.tools import tool

from server.websocket import broadcast_command
from config.settings import settings

logger = logging.getLogger(__name__)


# ── ESP32 hardware tools ───────────────────────────────────────────────────────

@tool
def shake_hand() -> str:
    """Make the robot shake hands."""
    logger.info("[TOOL] shake_hand")
    broadcast_command("shake_hand")
    return "Handshake gesture activated"


@tool
def wave() -> str:
    """Make the robot wave hello."""
    logger.info("[TOOL] wave")
    broadcast_command("wave")
    return "Wave gesture activated"


@tool
def blink_led(times: int = 3) -> str:
    """Blink the LED a given number of times."""
    logger.info("[TOOL] blink_led | times=%d", times)
    broadcast_command("blink_led", times=times)
    return f"LED blinked {times} times"


@tool
def led_on() -> str:
    """Turn the ESP32 LED on."""
    logger.info("[TOOL] led_on")
    broadcast_command("led_on")
    return "LED turned ON"


@tool
def led_off() -> str:
    """Turn the ESP32 LED off."""
    logger.info("[TOOL] led_off")
    broadcast_command("led_off")
    return "LED turned OFF"


@tool
def set_servo(angle: int) -> str:
    """Move the servo to a specific angle (0-180 degrees)."""
    angle = max(0, min(180, angle))
    logger.info("[TOOL] set_servo | angle=%d", angle)
    broadcast_command("set_servo", angle=angle)
    return f"Servo moved to {angle} degrees"


# ── Robot car motor tool ───────────────────────────────────────────────────────

@tool
def motor_move(direction: str, speed: int = 85) -> str:
    """
    Move the robot car.
    direction: forward | backward | left | right | stop
    speed: 0-100 (percent)
    """
    direction = direction.lower().strip()
    speed     = max(0, min(100, speed))

    _DIR_MAP = {
        "forward":  "F",
        "backward": "B",
        "back":     "B",
        "left":     "L",
        "right":    "R",
        "stop":     "S",
    }

    cmd = _DIR_MAP.get(direction, "S")
    url = f"{settings.esp32_car_url}/cmd?m={cmd}&s={speed}"

    logger.info("[TOOL] motor_move | direction=%s cmd=%s speed=%d", direction, cmd, speed)

    try:
        resp = requests.get(url, timeout=settings.car_cmd_timeout)
        resp.raise_for_status()
        return f"Car moving {direction} at {speed}% speed"
    except requests.exceptions.Timeout:
        logger.warning("motor_move: ESP32 car HTTP timeout")
        return f"Car command sent (timeout — car may still respond)"
    except Exception:
        logger.exception("motor_move: HTTP error")
        return "Car command failed — check ESP32 connection"


# ── Weather ────────────────────────────────────────────────────────────────────

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    logger.info("[TOOL] get_weather | city=%s", city)
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=5,
        ).json()

        if not geo.get("results"):
            return f"City '{city}' not found."

        r       = geo["results"][0]
        lat, lon = r["latitude"], r["longitude"]
        name    = r["name"]
        country = r.get("country", "")

        wx = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,weathercode,windspeed_10m,precipitation",
                "temperature_unit": "celsius",
                "windspeed_unit": "kmh",
                "timezone": "auto",
            },
            timeout=5,
        ).json()

        c    = wx["current"]
        temp = c["temperature_2m"]
        hum  = c["relative_humidity_2m"]
        wind = c["windspeed_10m"]
        rain = c["precipitation"]
        code = c["weathercode"]

        WMO = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Fog", 51: "Light drizzle", 61: "Light rain", 63: "Rain",
            65: "Heavy rain", 71: "Light snow", 73: "Snow", 80: "Showers",
            95: "Thunderstorm",
        }
        desc = WMO.get(code, f"Code {code}")

        return (
            f"{name}, {country}: {desc}, {temp}°C, "
            f"humidity {hum}%, wind {wind} km/h, rain {rain} mm"
        )
    except Exception:
        logger.exception("get_weather failed")
        return "Weather info unavailable"


# ── YouTube — emits open_url to the phone app via Socket.IO ──────────────────

@tool
def search_youtube(query: str) -> str:
    """
    Search YouTube. The phone app will open the result in the browser.
    Emits an open_url tool_call to the connected Socket.IO app client.
    """
    logger.info("[TOOL] search_youtube | query=%s", query)
    url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
    # Import here to avoid circular import at module load time
    try:
        from server.api import sio as _sio, _event_loop
        import asyncio
        if _event_loop and not _event_loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                _sio.emit("tool_call", {"tool": "open_url", "params": {"url": url}}),
                _event_loop,
            )
    except Exception:
        logger.exception("search_youtube: failed to emit to app")
    return f"Opening YouTube search: {query}"


ALL_TOOLS = [
    shake_hand, wave, blink_led, led_on, led_off,
    set_servo, motor_move, get_weather, search_youtube,
]
