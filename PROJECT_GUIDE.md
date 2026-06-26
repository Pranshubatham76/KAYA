# Project Overview & Guide

## Directory Structure
Here is an overview of what each folder and file does in your workspace:

- **`android/`**: Contains the native Android edge-compute application. All mobile-specific code and resources live here.
- **`backend/`**: Contains the backend API application (likely FastAPI) and infrastructure configurations like `docker-compose.yml`.
- **`dashboard/`**: Contains the frontend web application codebase, built with modern web technologies like Vite and TailwindCSS.
- **`ml/`**: Houses the machine learning models, training scripts, and related components.
- **`doc/`**: Contains project documentation, diagrams, or architectural notes.
- **`gradle-8.3/`**: A standalone extraction of the Gradle build tool, which allows you to compile the Android app without needing to download a system-wide Gradle or Android Studio.
- **`gradle.zip`**: The compressed archive of the Gradle 8.3 build tool.
- **`.venv/`**: A Python virtual environment used to manage Python dependencies for the backend and ML services.

---

## How to Start the Applications

### 1. Backend
Open a terminal (PowerShell or Command Prompt) and run:
```powershell
cd d:\files\backend
# Activate your Python virtual environment
..\.venv\Scripts\activate
# Start the backend server (assuming uvicorn is used)
uvicorn app.main:app --reload
```
*(Alternatively, if you are using Docker, you can run `docker-compose up --build` inside the `backend` folder).*

### 2. Frontend Dashboard
Open a new terminal window and run:
```powershell
cd d:\files\dashboard
# Install node dependencies (only needed the first time)
npm install
# Start the development server
npm run dev
```

### 3. Android Application (Building)
You can compile your Android app using the included Gradle tool.
Open a new terminal window and run:
```powershell
cd d:\files\android
# Run the Gradle assemble command using the standalone Gradle binary
..\gradle-8.3\bin\gradle.bat assembleDebug
```
Once the build is complete, your debug APK will be generated at:
`d:\files\android\app\build\outputs\apk\debug\app-debug.apk`

---

## Step-by-Step Guide: Testing the Android App *Without* Android Studio

If you want to test your Android application on a physical device without installing the massive Android Studio IDE, follow these steps:

### Step 1: Download the Android SDK Platform-Tools
Instead of the full IDE, you only need the **Platform-Tools** (which includes `adb` - Android Debug Bridge).
1. Go to the [Android Platform-Tools download page](https://developer.android.com/tools/releases/platform-tools).
2. Download the ZIP file for Windows.
3. Extract the ZIP file to a convenient location on your PC (e.g., `C:\platform-tools`).
4. **Optional but recommended:** Add `C:\platform-tools` to your Windows System `PATH` environment variable so you can run `adb` commands from anywhere.

### Step 2: Prepare Your Physical Android Device
1. On your Android phone, go to **Settings > About phone**.
2. Scroll down to the **Build number** and tap it 7 times until you see a message saying "You are now a developer!".
3. Go back to the main Settings menu, and navigate to **System > Developer options**.
4. Scroll down and enable **USB debugging**.

### Step 3: Connect and Install the App via CLI
1. Connect your phone to your computer using a USB cable.
2. Open a terminal (PowerShell or Command Prompt) and type:
   ```powershell
   adb devices
   ```
3. Look at your phone's screen. A prompt will ask you to **"Allow USB debugging"** from your computer's RSA key fingerprint. Check "Always allow" and tap **OK**.
4. Run `adb devices` again. You should see your device listed with the word `device` next to it.
5. Install the APK you built earlier by running:
   ```powershell
   cd d:\files\android
   adb install app\build\outputs\apk\debug\app-debug.apk
   ```
6. The app will install and you can launch it directly from your phone's app drawer!

### 💡 Alternative Method (No Cables / No ADB)
If you don't want to bother with `adb` at all, you can use this simple workaround:
1. Build the APK using the Gradle command (`..\gradle-8.3\bin\gradle.bat assembleDebug`).
2. Transfer the resulting `app-debug.apk` file to your Android phone (you can upload it to Google Drive, email it to yourself, or use a USB thumb drive).
3. Open a File Manager on your phone, find the APK, and tap it to install. *(You may be prompted to allow installations from "Unknown Sources" in your phone's settings).*
