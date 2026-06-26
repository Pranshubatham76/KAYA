package com.sentinelsite.imu

class JerkDetector(private val thresholdRadPerSecCubed: Float = 4.2f) {
    
    private var lastOmegaX = 0f
    private var lastOmegaY = 0f
    private var lastOmegaZ = 0f
    private var lastTime = 0L

    fun processSample(omegaX: Float, omegaY: Float, omegaZ: Float, timestampMs: Long): Boolean {
        if (lastTime == 0L) {
            lastOmegaX = omegaX
            lastOmegaY = omegaY
            lastOmegaZ = omegaZ
            lastTime = timestampMs
            return false
        }

        val dt = (timestampMs - lastTime) / 1000f // in seconds
        if (dt <= 0) return false

        val dOmegaX = (omegaX - lastOmegaX) / dt
        val dOmegaY = (omegaY - lastOmegaY) / dt
        val dOmegaZ = (omegaZ - lastOmegaZ) / dt

        val jerkMagnitude = Math.sqrt((dOmegaX * dOmegaX + dOmegaY * dOmegaY + dOmegaZ * dOmegaZ).toDouble()).toFloat()

        lastOmegaX = omegaX
        lastOmegaY = omegaY
        lastOmegaZ = omegaZ
        lastTime = timestampMs

        return jerkMagnitude > thresholdRadPerSecCubed
    }
}
