package com.sentinelsite.fusion

data class NearMissPayload(
    val eventId: String,
    val timestamp: Long,
    val lat: Double,
    val lon: Double,
    val audioFilePath: String,
    val frameFilePath: String,
    val yamnetClass: Int,
    val yamnetScore: Float,
    val visualClass: Int,
    val visualScore: Float,
    val workerId: String,
    val deviceId: String
)

object NearMissPayloadBuilder {
    fun build(
        eventId: String, timestamp: Long, lat: Double, lon: Double,
        audioFilePath: String, frameFilePath: String,
        yamnetClass: Int, yamnetScore: Float,
        visualClass: Int, visualScore: Float,
        workerId: String, deviceId: String
    ): NearMissPayload {
        return NearMissPayload(
            eventId, timestamp, lat, lon, audioFilePath, frameFilePath,
            yamnetClass, yamnetScore, visualClass, visualScore, workerId, deviceId
        )
    }
}
