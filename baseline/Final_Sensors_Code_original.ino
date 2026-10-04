#include <WiFi.h>
#include <Firebase_ESP_Client.h>
#include <DHT.h>

// Required by the Firebase library
#include <addons/TokenHelper.h>
#include <addons/RTDBHelper.h>

// --- SENSOR CONFIGURATION ---
#define DHTPIN 4
#define LDRPIN 34
#define WATERPIN 35
#define DHTTYPE DHT22
#define PHPIN 32      // <-- Analog pH Sensor Pin

// TDS UART Pins (Hardware Serial 2)
#define RXD2 16
#define TXD2 17

DHT dht(DHTPIN, DHTTYPE);
HardwareSerial SerialTDS(2);

// --- FIREBASE & WIFI CONFIGURATION ---
// IMPORTANT: Update these with your real credentials before uploading
#define WIFI_SSID "test"
#define WIFI_PASSWORD "12345678"
#define API_KEY "AIzaSyBgQEsRK4YFcZymGNb_iwRx5te5ykW8zXM"
#define DATABASE_URL "https://hydrosmart-pbl-default-rtdb.asia-southeast1.firebasedatabase.app/" 
#define USER_EMAIL "test2@test.com"
#define USER_PASSWORD "12345678"

FirebaseData fbdo;
FirebaseAuth auth;
FirebaseConfig config;

// Timers for non-blocking delays
unsigned long lastSerialMillis = 0;
unsigned long lastFirebaseMillis = 0;

float currentTDS = 0.0;
float waterTemp = 0.0;
float currentPH = 0.0; // <-- Variable to hold pH
float phOffset = -1.50;

// Command to request data from the UGE TDS sensor
byte requestCommand[] = {0xA0, 0x00, 0x00, 0x00, 0x00, 0xA0};
byte calibCommand[]   = {0xA0, 0x00, 0x00, 0x00, 0x00, 0xA6}; // Secret A6 Calibration Command

// --- TDS CALIBRATION FUNCTION ---
void calibrateTDS() {
  Serial.println(">>> Sending UGE Calibration Sequence...");
  SerialTDS.write(calibCommand, sizeof(calibCommand));
  delay(1000); // Give the UGE chip time to zero itself
  Serial.println(">>> TDS Calibration Complete!");
}

void setup() {
  Serial.begin(115200);
  
  // Initialize TDS Serial (9600 baud)
  SerialTDS.begin(9600, SERIAL_8N1, RXD2, TXD2);
  delay(100);

  calibrateTDS();

  dht.begin();
  analogReadResolution(12);

  // 1. Connect to Wi-Fi
  Serial.print("Connecting to Wi-Fi");
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(300);
  }
  Serial.println("\nConnected to Wi-Fi!");

  // 2. Initialize Firebase
  config.api_key = API_KEY;
  auth.user.email = USER_EMAIL;
  auth.user.password = USER_PASSWORD;
  config.database_url = DATABASE_URL;
  config.token_status_callback = tokenStatusCallback; 
  
  Firebase.reconnectNetwork(true);
  fbdo.setBSSLBufferSize(4096, 1024);
  fbdo.setResponseSize(2048);
  
  Firebase.begin(&config, &auth);
  Firebase.setDoubleDigits(2); 
  config.timeout.serverResponse = 10 * 1000;

Serial.println("--- System Online: [A-Temp | Hum | Light | W-Lvl | TDS | W-Temp | pH] ---");}

void loop() {
  // --- LOCAL READING & SERIAL PRINT (Every 2 Seconds) ---
  if (millis() - lastSerialMillis > 2000) {
    lastSerialMillis = millis();

    // 1. TDS Request & Parse
    SerialTDS.write(requestCommand, sizeof(requestCommand));
    delay(150); 
    
    // Check buffer and align strictly with the '0xAA' starting byte
    while (SerialTDS.available()) {
      if (SerialTDS.read() == 0xAA) {
        if (SerialTDS.available() >= 4) {
          byte tdsHi = SerialTDS.read();
          byte tdsLo = SerialTDS.read();
          byte tempHi = SerialTDS.read();
          byte tempLo = SerialTDS.read();
          
          currentTDS = (tdsHi << 8) | tdsLo; 
          // Using the manufacturer's /100.0 math!
          WaterTemp = ((tempHi << 8) | tempLo) / 100.0; 
        }
        break; // Exit the while loop once we successfully parse the packet
      }
    }
    // 2. Read pH Sensor
    int phRaw = analogRead(PHPIN);
    float phVoltage = phRaw * (3.3 / 4095.0); // Convert 12-bit ESP32 ADC to Voltage
    currentPH = (3.5 * phVoltage) + phOffset; // Standard op-amp conversion equation

    //3. Read Other Sensors
    float h = dht.readHumidity();
    float t = dht.readTemperature();
    int light = analogRead(LDRPIN);
    int waterLvl = analogRead(WATERPIN);

    // 4. Single Line Print Output
    Serial.print("DATA: ");
    Serial.print(t);      Serial.print("C | Hum:");
    Serial.print(h);      Serial.print("% | L:");
    Serial.print(light);  Serial.print(" | W-Lvl:");
    Serial.print(waterLvl); Serial.print(" | TDS:");
    Serial.print(currentTDS); Serial.print("ppm | W-Temp:");
    Serial.print(WaterTemp); Serial.println("C | pH:");
    Serial.print(currentPH); Serial.println("");

    // --- FIREBASE CLOUD SYNC (Every 10 Seconds) ---
    if (Firebase.ready() && (millis() - lastFirebaseMillis > 10000 || lastFirebaseMillis == 0)) {
      lastFirebaseMillis = millis();

      if (!isnan(h) && !isnan(t)) {
        Firebase.RTDB.setFloat(&fbdo, F("/sensors/ambient/temp"), t);
        Firebase.RTDB.setFloat(&fbdo, F("/sensors/ambient/humidity"), h);
      }
      
      Firebase.RTDB.setInt(&fbdo, F("/sensors/light"), light);
      Firebase.RTDB.setInt(&fbdo, F("/sensors/water_level"), waterLvl);
      Firebase.RTDB.setFloat(&fbdo, F("/sensors/water/tds"), currentTDS);
      Firebase.RTDB.setFloat(&fbdo, F("/sensors/water/temp"), (waterTemp-200));
      Firebase.RTDB.setFloat(&fbdo, F("/sensors/water/ph"), currentPH); // <-- Push new pH Data

      Serial.println(">>> CLOUD SYNC: Data successfully pushed to Firebase! <<<");
    }
  }
}
