package com.sentinelsite.upload

import android.content.Context
import android.util.Log
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import io.ktor.client.*
import io.ktor.client.engine.android.*
import io.ktor.client.request.forms.*
import io.ktor.client.statement.*
import io.ktor.http.*
import java.io.File

class SyncWorker(
    appContext: Context, 
    workerParams: WorkerParameters
) : CoroutineWorker(appContext, workerParams) {

    private val db = androidx.room.Room.databaseBuilder(
        applicationContext,
        SentinelDatabase::class.java, "sentinel-db"
    ).build()

    private val client = HttpClient(Android)

    override suspend fun doWork(): Result {
        val pendingEvents = db.nearMissDao().getPendingUploads()
        if (pendingEvents.isEmpty()) return Result.success()

        var allSuccess = true

        for (event in pendingEvents) {
            try {
                val response: HttpResponse = client.submitFormWithBinaryData(
                    url = "http://10.0.2.2:8000/api/v1/events",
                    formData = formData {
                        append("payload", """
                            {
                                "event_id": "${event.eventId}",
                                "site_id": "site_1",
                                "timestamp": "${event.timestamp}",
                                "lat": ${event.lat},
                                "lon": ${event.lon},
                                "yamnet_class": ${event.yamnetClass},
                                "yamnet_confidence": ${event.yamnetScore},
                                "visual_class": ${event.visualClass},
                                "visual_confidence": ${event.visualScore},
                                "worker_id": "${event.workerId}",
                                "device_id": "${event.deviceId}"
                            }
                        """.trimIndent())
                        
                        val audioFile = File(event.audioFilePath)
                        if (audioFile.exists()) {
                            append("audio", audioFile.readBytes(), Headers.build {
                                append(HttpHeaders.ContentDisposition, "filename=\"audio.wav\"")
                            })
                        }
                    }
                )

                if (response.status.isSuccess()) {
                    db.nearMissDao().markAsUploaded(event.eventId)
                    File(event.audioFilePath).delete() // Cleanup
                } else {
                    allSuccess = false
                    Log.e("SyncWorker", "Failed to upload event ${event.eventId}: ${response.status}")
                }
            } catch (e: Exception) {
                allSuccess = false
                Log.e("SyncWorker", "Exception during upload", e)
            }
        }

        return if (allSuccess) Result.success() else Result.retry()
    }
}
