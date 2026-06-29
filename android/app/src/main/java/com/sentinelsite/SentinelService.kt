package com.sentinelsite

import android.app.NotificationChannel
import android.app.NotificationManager
import androidx.lifecycle.LifecycleService
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import com.sentinelsite.audio.AudioBufferManager
import com.sentinelsite.audio.YAMNetInferenceEngine
import com.sentinelsite.fusion.EventTriggerController
import com.sentinelsite.fusion.FusionGate
import com.sentinelsite.imu.IMUManager
import com.sentinelsite.imu.LocationManager
import com.sentinelsite.vision.CameraManager
import kotlinx.coroutines.*

class SentinelService : LifecycleService() {

    private lateinit var audioBufferManager: AudioBufferManager
    private lateinit var yamnetEngine: YAMNetInferenceEngine
    private lateinit var imuManager: IMUManager
    private lateinit var fusionGate: FusionGate
    private lateinit var cameraManager: CameraManager
    private lateinit var locationManager: LocationManager
    private lateinit var eventTriggerController: EventTriggerController
    
    private val serviceScope = CoroutineScope(Dispatchers.Default + Job())
    private var isMonitoring = false

    override fun onCreate() {
        super.onCreate()
        audioBufferManager = AudioBufferManager()
        yamnetEngine = YAMNetInferenceEngine(this)
        fusionGate = FusionGate()
        cameraManager = CameraManager(this, this)
        locationManager = LocationManager(this)
        eventTriggerController = EventTriggerController(this, audioBufferManager, cameraManager, locationManager)
        
        imuManager = IMUManager(this) { timestamp ->
            if (fusionGate.onImuJerk(timestamp)) {
                // For simplicity, passing dummy yamnet data if fusion triggered from IMU side first
                eventTriggerController.handleNearMissTrigger(timestamp, -1, 0f)
            }
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        createNotificationChannel()
        val notification = NotificationCompat.Builder(this, "SENTINEL_CHANNEL")
            .setContentTitle("SentinelSite Active")
            .setContentText("Passive monitoring is running.")
            .setSmallIcon(android.R.drawable.ic_menu_camera)
            .build()
            
        startForeground(1, notification)
        startMonitoring()
        
        return START_STICKY
    }

    private fun startMonitoring() {
        if (isMonitoring) return
        isMonitoring = true
        
        audioBufferManager.startRecording()
        imuManager.start()
        
        serviceScope.launch {
            while (isActive && isMonitoring) {
                delay(1000) // Every 1 second
                val window = audioBufferManager.getRecentWindow(0.96f)
                val result = yamnetEngine.classify(window)
                
                // Simple threshold logic instead of full AnomalyScorer for now
                if (result.confidence > 0.6f && isConstructionAnomaly(result.classId)) {
                    if (fusionGate.onAudioAnomaly(System.currentTimeMillis())) {
                        eventTriggerController.handleNearMissTrigger(
                            System.currentTimeMillis(), 
                            result.classId, 
                            result.confidence
                        )
                    }
                }
            }
        }
    }
    
    private fun isConstructionAnomaly(classId: Int): Boolean {
        // e.g. 373=Crash, 374=Bang, 376=Thud, 44=Shout
        return classId in listOf(373, 374, 376, 44, 45, 378, 388, 375, 414)
    }

    override fun onDestroy() {
        super.onDestroy()
        isMonitoring = false
        audioBufferManager.stopRecording()
        imuManager.stop()
        yamnetEngine.close()
        serviceScope.cancel()
    }

    override fun onBind(intent: Intent): IBinder? {
        super.onBind(intent)
        return null
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                "SENTINEL_CHANNEL",
                "Sentinel Service",
                NotificationManager.IMPORTANCE_LOW
            )
            val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            manager.createNotificationChannel(channel)
        }
    }
}
