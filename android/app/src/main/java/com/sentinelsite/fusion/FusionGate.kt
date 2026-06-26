package com.sentinelsite.fusion

class FusionGate {
    private var pendingAudioEventTime: Long? = null
    private var pendingImuEventTime: Long? = null
    
    private val fusionWindowMs = 2000L // ±2 seconds
    private val cooldownMs = 30000L // 30s cooldown
    private var lastTriggerTime: Long = 0L

    fun onAudioAnomaly(timestamp: Long): Boolean {
        if (System.currentTimeMillis() - lastTriggerTime < cooldownMs) return false
        
        pendingAudioEventTime = timestamp
        return checkFusion(timestamp)
    }

    fun onImuJerk(timestamp: Long): Boolean {
        if (System.currentTimeMillis() - lastTriggerTime < cooldownMs) return false
        
        pendingImuEventTime = timestamp
        return checkFusion(timestamp)
    }

    private fun checkFusion(currentTime: Long): Boolean {
        val audioTime = pendingAudioEventTime
        val imuTime = pendingImuEventTime

        if (audioTime != null && imuTime != null) {
            val diff = Math.abs(audioTime - imuTime)
            if (diff <= fusionWindowMs) {
                // NEAR MISS TRIGGERED
                lastTriggerTime = currentTime
                pendingAudioEventTime = null
                pendingImuEventTime = null
                return true
            }
        }

        // Cleanup old events
        if (audioTime != null && currentTime - audioTime > fusionWindowMs * 2) {
            pendingAudioEventTime = null
        }
        if (imuTime != null && currentTime - imuTime > fusionWindowMs * 2) {
            pendingImuEventTime = null
        }
        
        return false
    }
}
