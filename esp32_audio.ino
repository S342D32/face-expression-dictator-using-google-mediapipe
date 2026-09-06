/*
=========================================================
 ESP32-S3 LangGraph Voice Agent — streaming sentence TTS
=========================================================
*/

#define WEBSOCKETS_MAX_DATA_SIZE (8 * 1024)

#include <WiFi.h>
#include <WebSocketsClient.h>
#include <driver/i2s.h>
#include <ESP32Servo.h>
#include <ArduinoJson.h>
#include <Wire.h>
#include <Adafruit_VL53L0X.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <cstring>
#include <math.h>

// ---------------- WiFi ----------------
#define WIFI_SSID "Sourajit"
#define WIFI_PASS "Nayak@2002"
#define PC_HOST   "10.118.180.202"
#define PC_PORT   8765

// ---------------- Speaker (MAX98357A) ----------------
#define SPK_BCLK 5
#define SPK_LRC  6
#define SPK_DOUT 7

// ---------------- LED + Servo ----------------
#define LED_PIN   48
#define SERVO_PIN 10

// ---------------- I2C + VL53L0X + OLED ----------------
#define I2C_SDA          8
#define I2C_SCL          9
#define OLED_ADDR        0x3C
#define DIST_INTERVAL_MS 200

Adafruit_VL53L0X vl53;
Adafruit_SSD1306 oled(128, 64, &Wire, -1);
bool     vl53Ready  = false;
uint32_t lastDistMs = 0;

Servo myServo;

// ---------------- Microphone (INMP441) ----------------
#define MIC_BCLK 15
#define MIC_WS   16
#define MIC_DATA 17

#define SAMPLE_RATE 16000
#define MIC_BUF_LEN 512

int32_t micRaw[MIC_BUF_LEN];
int16_t micPCM[MIC_BUF_LEN];

// ---- TTS receive buffer (PSRAM) ----
// Each sentence chunk is small (~1-3 sec), 12 sec max is plenty
#define RX_BUF_MAX (16000 * 2 * 12)

uint8_t  *rxBuf       = nullptr;
uint32_t  rxExpected  = 0;
uint32_t  rxReceived  = 0;
bool      rxHasHeader = false;

// ---- Volume (software gain, 1.0 = original, 2.0 = double, max ~4.0 before clipping) ----
#define VOLUME 2.5f

// ---- Playback task signalling ----
volatile bool rxInterrupt = false;   // set true when new header arrives → stop current playback
TaskHandle_t  playTaskHandle = nullptr;

struct PlayJob {
  uint8_t  *data;
  uint32_t  len;
};

// Double-buffer: one being filled, one being played
uint8_t *playBuf[2]  = {nullptr, nullptr};
volatile int fillIdx = 0;   // index into playBuf[] currently being filled by WS

QueueHandle_t playQueue;    // sends PlayJob to playback task

WebSocketsClient ws;
bool wsConnected = false;
bool micMuted    = false;   // true while TTS is playing on speaker

// =====================================================
// OLED — show expression based on distance
// =====================================================
void showOledExpression(int cm) {
  oled.clearDisplay();
  oled.setTextColor(SSD1306_WHITE);

  if (cm < 0) {
    // Sensor error
    oled.setTextSize(2);
    oled.setCursor(20, 20);
    oled.print("  --cm");
  } else if (cm < 15) {
    // Too close — shocked
    oled.setTextSize(3);
    oled.setCursor(30, 5);  oled.print("O_O");
    oled.setTextSize(1);
    oled.setCursor(10, 40); oled.print("WHOA! TOO CLOSE!");
    oled.setCursor(25, 52); oled.printf("%d cm", cm);
  } else if (cm < 30) {
    // Close — suspicious
    oled.setTextSize(3);
    oled.setCursor(30, 5);  oled.print(">_>");
    oled.setTextSize(1);
    oled.setCursor(18, 40); oled.print("Getting closer...");
    oled.setCursor(25, 52); oled.printf("%d cm", cm);
  } else if (cm < 60) {
    // Medium — happy
    oled.setTextSize(3);
    oled.setCursor(30, 5);  oled.print("^_^");
    oled.setTextSize(1);
    oled.setCursor(28, 40); oled.print("Hello there!");
    oled.setCursor(25, 52); oled.printf("%d cm", cm);
  } else {
    // Far — neutral/sleeping
    oled.setTextSize(3);
    oled.setCursor(30, 5);  oled.print("-_-");
    oled.setTextSize(1);
    oled.setCursor(22, 40); oled.print("Waiting...");
    oled.setCursor(25, 52); oled.printf("%d cm", cm);
  }

  oled.display();
}

// =====================================================
// Command handlers
// =====================================================
void cmdBlinkLed(int times) {
  for (int i = 0; i < times; i++) {
    digitalWrite(LED_PIN, HIGH); delay(200);
    digitalWrite(LED_PIN, LOW);  delay(200);
  }
}

void cmdSetServo(int angle) {
  myServo.write(angle);
}

void cmdShakeHand() {
  // Shake: sweep servo back and forth 3 times
  for (int i = 0; i < 3; i++) {
    myServo.write(45);  delay(300);
    myServo.write(135); delay(300);
  }
  myServo.write(90);
}

void cmdWave() {
  // Wave: fast sweep 5 times
  for (int i = 0; i < 5; i++) {
    myServo.write(60);  delay(200);
    myServo.write(120); delay(200);
  }
  myServo.write(90);
}

void handleCommand(const char *json) {
  StaticJsonDocument<256> doc;
  if (deserializeJson(doc, json) != DeserializationError::Ok) {
    Serial.println("Bad JSON command");
    return;
  }
  const char *cmd = doc["cmd"];
  Serial.printf("CMD: %s\n", cmd);

  if      (strcmp(cmd, "blink_led")  == 0) cmdBlinkLed(doc["times"] | 3);
  else if (strcmp(cmd, "set_servo")  == 0) cmdSetServo(doc["angle"] | 90);
  else if (strcmp(cmd, "shake_hand") == 0) cmdShakeHand();
  else if (strcmp(cmd, "wave")       == 0) cmdWave();
  else if (strcmp(cmd, "led_on")     == 0) digitalWrite(LED_PIN, HIGH);
  else if (strcmp(cmd, "led_off")    == 0) digitalWrite(LED_PIN, LOW);
  else if (strcmp(cmd, "mic_mute")   == 0) { micMuted = true;  Serial.println("Mic muted"); }
  else if (strcmp(cmd, "mic_unmute") == 0) { micMuted = false; Serial.println("Mic unmuted"); }
  else Serial.printf("Unknown cmd: %s\n", cmd);
}

// =====================================================
// Speaker setup
// =====================================================
void setupSpeaker() {
  i2s_config_t cfg = {
    .mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
    .sample_rate          = SAMPLE_RATE,
    .bits_per_sample      = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format       = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count        = 8,
    .dma_buf_len          = 512,
    .use_apll             = false,
    .tx_desc_auto_clear   = true,
    .fixed_mclk           = 0
  };
  i2s_pin_config_t pins = {
    .mck_io_num   = I2S_PIN_NO_CHANGE,
    .bck_io_num   = SPK_BCLK,
    .ws_io_num    = SPK_LRC,
    .data_out_num = SPK_DOUT,
    .data_in_num  = I2S_PIN_NO_CHANGE
  };
  i2s_driver_install(I2S_NUM_0, &cfg, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &pins);
  i2s_zero_dma_buffer(I2S_NUM_0);
  Serial.println("Speaker Ready");
}

// =====================================================
// Microphone setup
// =====================================================
void setupMic() {
  i2s_config_t cfg = {
    .mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate          = SAMPLE_RATE,
    .bits_per_sample      = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format       = I2S_CHANNEL_FMT_ONLY_RIGHT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags     = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count        = 8,
    .dma_buf_len          = 256,
    .use_apll             = false,
    .tx_desc_auto_clear   = false,
    .fixed_mclk           = 0
  };
  i2s_pin_config_t pins = {
    .mck_io_num   = I2S_PIN_NO_CHANGE,
    .bck_io_num   = MIC_BCLK,
    .ws_io_num    = MIC_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num  = MIC_DATA
  };
  i2s_driver_install(I2S_NUM_1, &cfg, 0, NULL);
  i2s_set_pin(I2S_NUM_1, &pins);
  size_t dummy;
  for (int i = 0; i < 20; i++)
    i2s_read(I2S_NUM_1, micRaw, sizeof(micRaw), &dummy, 20 / portTICK_PERIOD_MS);
  Serial.println("Microphone Ready");
}

// =====================================================
// Playback task — runs on Core 1, plays one sentence at a time
// Interrupted immediately when rxInterrupt is set
// =====================================================
void playbackTask(void *) {
  PlayJob job;
  while (true) {
    if (xQueueReceive(playQueue, &job, portMAX_DELAY) != pdTRUE) continue;

    rxInterrupt = false;
    Serial.printf("Playing %u bytes\n", job.len);

    // Apply software volume gain into a small stack buffer
    static int16_t gainBuf[512];
    uint32_t offset = 0;
    while (offset < job.len) {
      if (rxInterrupt) {
        i2s_zero_dma_buffer(I2S_NUM_0);
        Serial.println("Playback interrupted by new sentence");
        break;
      }
      uint32_t samples = min((uint32_t)512, (job.len - offset) / 2);
      const int16_t *src = (const int16_t *)(job.data + offset);
      for (uint32_t i = 0; i < samples; i++) {
        int32_t v = (int32_t)(src[i] * VOLUME);
        if (v >  32767) v =  32767;
        if (v < -32768) v = -32768;
        gainBuf[i] = (int16_t)v;
      }
      size_t written = 0;
      i2s_write(I2S_NUM_0, gainBuf, samples * 2, &written, portMAX_DELAY);
      offset += written;
    }

    if (!rxInterrupt)
      Serial.println("Playback done");
  }
}

// =====================================================
// WebSocket callback
// Protocol per sentence:
//   Frame 1 : 4 bytes — little-endian uint32 total PCM length
//   Frame 2+: raw PCM chunks (4KB each)
// =====================================================
void onWsEvent(WStype_t type, uint8_t *payload, size_t length) {
  switch (type) {

    case WStype_CONNECTED:
      wsConnected = true;
      Serial.printf("WS Connected | heap=%u psram=%u\n",
                    ESP.getFreeHeap(), ESP.getFreePsram());
      break;

    case WStype_DISCONNECTED:
      wsConnected  = false;
      rxHasHeader  = false;
      rxExpected   = 0;
      rxReceived   = 0;
      rxInterrupt  = true;
      Serial.println("WS Disconnected");
      break;

    case WStype_TEXT:
      handleCommand((const char *)payload);
      break;

    case WStype_BIN:
      // Header frame: 4 bytes = total PCM length for this sentence
      if (length == 4 && !rxHasHeader) {
        uint32_t incoming =
          (uint32_t)payload[0]         |
          ((uint32_t)payload[1] << 8)  |
          ((uint32_t)payload[2] << 16) |
          ((uint32_t)payload[3] << 24);

        if (incoming == 0 || incoming > RX_BUF_MAX) {
          Serial.printf("Bad size: %u\n", incoming);
          return;
        }

        // Signal current playback to stop — new sentence coming
        rxInterrupt = true;

        // Switch fill buffer
        fillIdx    = 1 - fillIdx;
        rxExpected = incoming;
        rxReceived = 0;
        rxHasHeader = true;
        Serial.printf("Sentence header: %u bytes\n", rxExpected);
        return;
      }

      // Audio data chunks
      if (rxHasHeader && length > 0) {
        uint32_t toCopy = min((uint32_t)length, rxExpected - rxReceived);
        memcpy(playBuf[fillIdx] + rxReceived, payload, toCopy);
        rxReceived += toCopy;

        if (rxReceived >= rxExpected) {
          rxHasHeader = false;
          PlayJob job = { playBuf[fillIdx], rxExpected };
          xQueueOverwrite(playQueue, &job);   // replace any pending job
          Serial.printf("Sentence queued: %u bytes\n", rxExpected);
        }
      }
      break;

    default:
      break;
  }
}

// =====================================================
// Setup
// =====================================================
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n=== ESP32-S3 Audio Bridge ===");

  if (!psramFound()) { Serial.println("NO PSRAM!"); while(1) delay(1000); }

  // Allocate two play buffers in PSRAM (double-buffer)
  playBuf[0] = (uint8_t *)ps_malloc(RX_BUF_MAX);
  playBuf[1] = (uint8_t *)ps_malloc(RX_BUF_MAX);
  if (!playBuf[0] || !playBuf[1]) { Serial.println("PSRAM alloc failed!"); while(1) delay(1000); }
  Serial.printf("PSRAM OK: 2 x %d bytes\n", RX_BUF_MAX);

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.printf("\nWiFi OK | IP: %s\n", WiFi.localIP().toString().c_str());

  setupSpeaker();
  setupMic();

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  myServo.attach(SERVO_PIN);
  myServo.write(90);
  Serial.println("LED + Servo Ready");

  // I2C + VL53L0X + OLED
  Wire.begin(I2C_SDA, I2C_SCL);
  if (oled.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    oled.clearDisplay();
    oled.setTextSize(1);
    oled.setTextColor(SSD1306_WHITE);
    oled.setCursor(0, 0);
    oled.println("Booting...");
    oled.display();
    Serial.println("OLED Ready");
  } else {
    Serial.println("OLED not found");
  }
  if (vl53.begin()) {
    vl53Ready = true;
    Serial.println("VL53L0X Ready");
  } else {
    Serial.println("VL53L0X not found");
  }

  // Playback queue — depth 1, xQueueOverwrite replaces stale jobs
  playQueue = xQueueCreate(1, sizeof(PlayJob));
  xTaskCreatePinnedToCore(playbackTask, "play", 4096, nullptr, 2, &playTaskHandle, 1);

  ws.begin(PC_HOST, PC_PORT, "/");
  ws.onEvent(onWsEvent);
  ws.setReconnectInterval(3000);
  ws.enableHeartbeat(15000, 3000, 2);

  Serial.println("Ready");
}

// =====================================================
// Loop
// =====================================================
void loop() {
  ws.loop();

  // VL53L0X — poll every DIST_INTERVAL_MS, send to PC + update OLED
  if (vl53Ready && (millis() - lastDistMs >= DIST_INTERVAL_MS)) {
    lastDistMs = millis();
    VL53L0X_RangingMeasurementData_t measure;
    vl53.rangingTest(&measure, false);
    int cm = -1;
    if (measure.RangeStatus != 4) {   // 4 = out of range
      cm = measure.RangeMilliMeter / 10;
    }
    showOledExpression(cm);
    if (wsConnected && cm >= 0) {
      // Send distance as JSON text frame to PC
      char buf[48];
      snprintf(buf, sizeof(buf), "{\"event\":\"distance\",\"cm\":%d}", cm);
      ws.sendTXT(buf);
    }
  }

  if (!wsConnected) return;

  size_t bytesRead = 0;
  i2s_read(I2S_NUM_1, micRaw, sizeof(micRaw), &bytesRead, 10 / portTICK_PERIOD_MS);
  if (bytesRead == 0 || micMuted) return;

  int samples = bytesRead / sizeof(int32_t);
  for (int i = 0; i < samples; i++) {
    int32_t s = micRaw[i] >> 8;
    if (s >  32767) s =  32767;
    if (s < -32768) s = -32768;
    micPCM[i] = (int16_t)s;
  }
  ws.sendBIN((uint8_t *)micPCM, samples * sizeof(int16_t));
}
