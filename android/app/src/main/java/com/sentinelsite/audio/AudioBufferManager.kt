package com.sentinelsite.audio

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import java.nio.FloatBuffer
import java.util.concurrent.atomic.AtomicBoolean

class AudioBufferManager(
    private val sampleRate: Int = 16000,
    private val bufferDurationSeconds: Int = 30
) {
    private val bufferSize = sampleRate * bufferDurationSeconds
    private val circularBuffer = FloatArray(bufferSize)
    private var writeIndex = 0
    private val isRecording = AtomicBoolean(false)
    private var audioRecord: AudioRecord? = null

    @SuppressLint("MissingPermission")
    fun startRecording() {
        val minBufferSize = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )

        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            minBufferSize * 2
        )

        audioRecord?.startRecording()
        isRecording.set(true)

        Thread {
            val readBuffer = ShortArray(minBufferSize)
            while (isRecording.get()) {
                val readResult = audioRecord?.read(readBuffer, 0, readBuffer.size) ?: 0
                if (readResult > 0) {
                    for (i in 0 until readResult) {
                        // Normalize 16-bit PCM to float [-1.0, 1.0]
                        circularBuffer[writeIndex] = readBuffer[i] / 32768.0f
                        writeIndex = (writeIndex + 1) % bufferSize
                    }
                }
            }
        }.start()
    }

    fun stopRecording() {
        isRecording.set(false)
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
    }

    /**
     * Gets the most recent chunk of audio.
     */
    fun getRecentWindow(durationSeconds: Float): FloatArray {
        val windowSamples = (sampleRate * durationSeconds).toInt()
        val window = FloatArray(windowSamples)
        var readIndex = writeIndex - windowSamples
        if (readIndex < 0) {
            readIndex += bufferSize
        }
        for (i in 0 until windowSamples) {
            window[i] = circularBuffer[readIndex]
            readIndex = (readIndex + 1) % bufferSize
        }
        return window
    }

    /**
     * Freezes the entire 30s buffer.
     */
    fun getFullBuffer(): FloatArray {
        val buffer = FloatArray(bufferSize)
        var readIndex = writeIndex
        for (i in 0 until bufferSize) {
            buffer[i] = circularBuffer[readIndex]
            readIndex = (readIndex + 1) % bufferSize
        }
        return buffer
    }
}
