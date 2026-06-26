package com.sentinelsite.voice

import android.util.Log
import io.ktor.client.*
import io.ktor.client.engine.android.*
import io.ktor.client.request.*
import io.ktor.client.statement.*
import io.ktor.http.*
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

class VoiceCopilot {
    private val client = HttpClient(Android)
    
    // Simulate Wakeword engine (e.g. Porcupine) for now since it requires binary assets
    fun processAudioStream(audioChunk: FloatArray): Boolean {
        // Detect "Hey Sentinel"
        return false 
    }

    // Simulate Whisper STT inference since 150MB model isn't checked in
    fun transcribe(audioBuffer: FloatArray): String {
        Log.i("VoiceCopilot", "Transcribing audio...")
        return "Where is the nearest fire extinguisher?"
    }

    // Real call to Backend RAG intent router
    suspend fun queryCopilot(query: String, siteId: String, workerId: String): String {
        return try {
            val response: HttpResponse = client.post("http://10.0.2.2:8000/api/v1/voice/query") {
                contentType(ContentType.Application.Json)
                setBody("""{"site_id": "$siteId", "worker_id": "$workerId", "query": "$query"}""")
            }
            if (response.status.isSuccess()) {
                val jsonString = response.bodyAsText()
                val json = Json.parseToJsonElement(jsonString).jsonObject
                json["answer"]?.jsonPrimitive?.content ?: "Error parsing response"
            } else {
                "Failed to reach safety database."
            }
        } catch (e: Exception) {
            "Network error communicating with safety database."
        }
    }
}
