import logging
import requests
import webbrowser
from urllib.parse import quote_plus
from langchain_core.tools import tool
from server.websocket import broadcast_command

logger = logging.getLogger(__name__)

YOUTUBE_SEARCH_API = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_API_KEY    = ""   # optional — leave empty to use scrape fallback


@tool
def shake_hand() -> str:
    """Make the robot shake hands. Call when user asks for a handshake or greeting gesture."""
    logger.info("[TOOL] shake_hand")
    broadcast_command("shake_hand")
    return "Handshake gesture activated"


@tool
def wave() -> str:
    """Make the robot wave. Call when user asks to wave or say hello."""
    logger.info("[TOOL] wave")
    broadcast_command("wave")
    return "Wave gesture activated"


@tool
def blink_led(times: int = 3) -> str:
    """Blink the LED a given number of times to signal attention or acknowledgement."""
    logger.info("[TOOL] blink_led | times=%d", times)
    broadcast_command("blink_led", times=times)
    return f"LED blinked {times} times"

@tool
def led_on() -> str:
    """Turn the ESP32 built-in LED ON."""
    logger.info("[TOOL] led_on")
    broadcast_command("led_on")
    return "LED turned ON"

@tool
def led_off() -> str:
    """Turn the ESP32 built-in LED OFF."""
    logger.info("[TOOL] led_off")
    broadcast_command("led_off")
    return "LED turned OFF"


@tool
def set_servo(angle: int) -> str:
    """Move the servo to a specific angle (0-180 degrees). Use for pointing or rotating."""
    angle = max(0, min(180, angle))
    logger.info("[TOOL] set_servo | angle=%d", angle)
    broadcast_command("set_servo", angle=angle)
    return f"Servo moved to {angle} degrees"


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city. Call when user asks about weather, temperature,
    rain, humidity, or climate for any location."""
    logger.info("[TOOL] get_weather | city=%s", city)
    try:
        # Step 1: geocode city name → lat/lon
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=5,
        ).json()

        if not geo.get("results"):
            return f"City '{city}' not found."

        r       = geo["results"][0]
        lat     = r["latitude"]
        lon     = r["longitude"]
        name    = r["name"]
        country = r.get("country", "")

        # Step 2: fetch current weather
        wx = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude":            lat,
                "longitude":           lon,
                "current":             "temperature_2m,relative_humidity_2m,weathercode,windspeed_10m,precipitation",
                "temperature_unit":    "celsius",
                "windspeed_unit":      "kmh",
                "timezone":            "auto",
            },
            timeout=5,
        ).json()

        c    = wx["current"]
        temp = c["temperature_2m"]
        hum  = c["relative_humidity_2m"]
        wind = c["windspeed_10m"]
        rain = c["precipitation"]
        code = c["weathercode"]

        # WMO weather code → simple description
        WMO = {
            0: "साफ आसमान", 1: "लगभग साफ", 2: "आंशिक बादल", 3: "बादल छाए",
            45: "कोहरा", 48: "कोहरा",
            51: "हल्की बूंदाबांदी", 53: "बूंदाबांदी", 55: "तेज बूंदाबांदी",
            61: "हल्की बारिश", 63: "बारिश", 65: "तेज बारिश",
            71: "हल्की बर्फ", 73: "बर्फबारी", 75: "तेज बर्फबारी",
            80: "बौछारें", 81: "बौछारें", 82: "तेज बौछारें",
            95: "गरज के साथ बारिश", 96: "ओलावृष्टि", 99: "तेज ओलावृष्टि",
        }
        desc = WMO.get(code, f"कोड {code}")

        return (
            f"{name}, {country}: {desc}, तापमान {temp}°C, "
            f"नमी {hum}%, हवा {wind} km/h, वर्षा {rain} mm"
        )

    except Exception as e:
        logger.exception("Weather fetch failed")
        return f"मौसम जानकारी उपलब्ध नहीं: {e}"


@tool
def search_youtube(query: str) -> str:
    """Search YouTube and auto-play the first result. Call when user wants to watch
    or play a video/song/music on YouTube."""
    logger.info("[TOOL] search_youtube | query=%s", query)

    video_id = None

    # Try YouTube Data API v3 first (needs API key)
    if YOUTUBE_API_KEY:
        try:
            r = requests.get(
                YOUTUBE_SEARCH_API,
                params={
                    "part":       "id",
                    "q":          query,
                    "type":       "video",
                    "maxResults": 1,
                    "key":        YOUTUBE_API_KEY,
                },
                timeout=5,
            ).json()
            items = r.get("items", [])
            if items:
                video_id = items[0]["id"]["videoId"]
        except Exception:
            logger.warning("YouTube API failed, falling back to scrape")

    # Fallback: scrape YouTube search page for first video ID (no API key needed)
    if not video_id:
        try:
            html = requests.get(
                f"https://www.youtube.com/results?search_query={quote_plus(query)}",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=5,
            ).text
            # YouTube embeds video IDs as /watch?v=XXXXXXXXXXX in the HTML
            idx = html.find('/watch?v=')
            if idx != -1:
                video_id = html[idx + 9: idx + 20]
        except Exception:
            logger.warning("YouTube scrape failed")

    if video_id:
        url = f"https://www.youtube.com/watch?v={video_id}&autoplay=1"
        webbrowser.open(url)
        return f"Playing YouTube: {query}"
    else:
        # Last resort: open search results
        webbrowser.open(f"https://www.youtube.com/results?search_query={quote_plus(query)}")
        return f"Opened YouTube search: {query}"


ALL_TOOLS = [shake_hand, wave, blink_led, led_on, led_off, set_servo, get_weather, search_youtube]
