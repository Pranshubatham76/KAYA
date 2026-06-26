package com.sentinelsite.model

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.net.URL

class ModelUpdateManager(private val context: Context) {
    
    // Check if new model is available and download it to a staging area
    suspend fun checkForUpdates(modelUrl: String, version: String): Boolean = withContext(Dispatchers.IO) {
        try {
            val stagingDir = File(context.filesDir, "models_staging")
            if (!stagingDir.exists()) stagingDir.mkdirs()
            
            val modelFile = File(stagingDir, "yamnet_v${version}.tflite")
            
            URL(modelUrl).openStream().use { input ->
                FileOutputStream(modelFile).use { output ->
                    input.copyTo(output)
                }
            }
            
            // Mark as ready for next cold start
            context.getSharedPreferences("sentinel_prefs", Context.MODE_PRIVATE)
                .edit()
                .putString("pending_model_update", modelFile.absolutePath)
                .apply()
                
            Log.i("ModelUpdate", "Successfully downloaded model $version")
            return@withContext true
        } catch (e: Exception) {
            Log.e("ModelUpdate", "Failed to download model", e)
            return@withContext false
        }
    }

    // Called on app startup (cold start) to swap the model before inference engines initialize
    fun applyPendingUpdates() {
        val prefs = context.getSharedPreferences("sentinel_prefs", Context.MODE_PRIVATE)
        val pendingModelPath = prefs.getString("pending_model_update", null)
        
        if (pendingModelPath != null) {
            try {
                val stagingFile = File(pendingModelPath)
                if (stagingFile.exists()) {
                    val activeModelFile = File(context.filesDir, "yamnet_active.tflite")
                    stagingFile.copyTo(activeModelFile, overwrite = true)
                    stagingFile.delete()
                    Log.i("ModelUpdate", "Applied pending model update")
                }
            } catch (e: Exception) {
                Log.e("ModelUpdate", "Failed to apply model update", e)
            } finally {
                prefs.edit().remove("pending_model_update").apply()
            }
        }
    }
}
