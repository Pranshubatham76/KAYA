package com.sentinelsite.upload

import android.content.Context
import androidx.room.*
import com.sentinelsite.fusion.NearMissPayload
import kotlinx.coroutines.flow.Flow

@Entity(tableName = "near_miss_events")
data class NearMissEntity(
    @PrimaryKey val eventId: String,
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
    val deviceId: String,
    val isUploaded: Boolean = false
)

@Dao
interface NearMissDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(event: NearMissEntity)

    @Query("SELECT * FROM near_miss_events WHERE isUploaded = 0 ORDER BY timestamp ASC")
    suspend fun getPendingUploads(): List<NearMissEntity>

    @Query("UPDATE near_miss_events SET isUploaded = 1 WHERE eventId = :id")
    suspend fun markAsUploaded(id: String)
}

@Database(entities = [NearMissEntity::class], version = 1)
abstract class SentinelDatabase : RoomDatabase() {
    abstract fun nearMissDao(): NearMissDao
}

class UploadQueueManager(private val context: Context) {
    private val db = Room.databaseBuilder(
        context.applicationContext,
        SentinelDatabase::class.java, "sentinel-db"
    ).build()

    suspend fun enqueue(payload: NearMissPayload) {
        val entity = NearMissEntity(
            payload.eventId, payload.timestamp, payload.lat, payload.lon,
            payload.audioFilePath, payload.frameFilePath, payload.yamnetClass,
            payload.yamnetScore, payload.visualClass, payload.visualScore,
            payload.workerId, payload.deviceId
        )
        db.nearMissDao().insert(entity)
        // Would normally trigger SyncWorker here
    }
}
