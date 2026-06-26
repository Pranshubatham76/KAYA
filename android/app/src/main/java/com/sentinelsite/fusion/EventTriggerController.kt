package com.sentinelsite.fusion

import android.content.Context
import android.util.Log
import com.sentinelsite.audio.AudioBufferManager
import com.sentinelsite.vision.CameraManager
import com.sentinelsite.imu.LocationManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.UUID

class EventTriggerController(
    private val context: Context,
    private val audioBufferManager: AudioBufferManager,
    private val cameraManager: CameraManager,
    private val locationManager: LocationManager
    // uploadQueueManager injected later
) {
    private val scope = CoroutineScope(Dispatchers.IO)

    fun handleNearMissTrigger(
        timestamp: Long, 
        yamnetClassId: Int, 
        yamnetScore: Float
    ) {
        Log.i("EventTrigger", "Near miss triggered at $timestamp")
        scope.launch {
            // 1. Freeze audio
            val audioData = audioBufferManager.getFullBuffer()
            val audioFile = saveAudioBufferToFile(audioData, timestamp)

            // 2. Request Camera Frame (actual)
            var frameFile: File? = null
            try {
                frameFile = cameraManager.captureFrame(timestamp)
            } catch (e: Exception) {
                Log.e("EventTrigger", "Failed to capture frame", e)
            }

            // 3. Get GPS (actual)
            val location = locationManager.getCurrentLocation()
            val lat = location?.latitude ?: 37.7749
            val lon = location?.longitude ?: -122.4194

            // 4. Build payload and queue it
            val payload = NearMissPayloadBuilder.build(
                eventId = UUID.randomUUID().toString(),
                timestamp = timestamp,
                lat = lat,
                lon = lon,
                audioFilePath = audioFile.absolutePath,
                frameFilePath = frameFile?.absolutePath ?: "",
                yamnetClass = yamnetClassId,
                yamnetScore = yamnetScore,
                visualClass = -1, // No visual classification yet
                visualScore = 0f,
                workerId = "worker_1",
                deviceId = "device_xyz"
            )
            
            // TODO: UploadQueueManager.enqueue(payload)
            Log.i("EventTrigger", "Event payload built: $payload")
        }
    }
    
    private fun saveAudioBufferToFile(audioData: FloatArray, timestamp: Long): File {
        val file = File(context.cacheDir, "event_${timestamp}.wav")
        // Very basic conversion of FloatArray to PCM16
        val shortArray = ShortArray(audioData.size)
        for (i in audioData.indices) {
            var scaled = audioData[i] * 32767.0f
            if (scaled > 32767f) scaled = 32767f
            if (scaled < -32768f) scaled = -32768f
            shortArray[i] = scaled.toInt().toShort()
        }
        val byteBuffer = ByteBuffer.allocate(shortArray.size * 2)
        byteBuffer.order(ByteOrder.LITTLE_ENDIAN)
        byteBuffer.asShortBuffer().put(shortArray)
        file.writeBytes(byteBuffer.array())
        return file
    }
}
