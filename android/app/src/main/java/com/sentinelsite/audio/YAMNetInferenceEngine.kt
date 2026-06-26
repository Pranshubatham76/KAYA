package com.sentinelsite.audio

import android.content.Context
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.FloatBuffer
import java.nio.MappedByteBuffer

class YAMNetInferenceEngine(context: Context) {
    private val interpreter: Interpreter
    
    // YAMNet requires exactly 0.96 seconds of audio at 16kHz (15600 samples)
    private val requiredSamples = 15360 

    init {
        val tfliteModel: MappedByteBuffer = FileUtil.loadMappedFile(context, "yamnet.tflite")
        val options = Interpreter.Options()
        // Use GPU or NPU delegate here if available in production
        options.setNumThreads(4)
        interpreter = Interpreter(tfliteModel, options)
    }

    data class ClassificationResult(val classId: Int, val confidence: Float)

    /**
     * Runs inference on the audio window.
     * Expected input: FloatArray of size 15360 [-1.0, 1.0]
     */
    fun classify(audioWindow: FloatArray): ClassificationResult {
        require(audioWindow.size == requiredSamples) { "Audio window must be exactly $requiredSamples samples" }

        val inputBuffer = FloatBuffer.wrap(audioWindow)
        // YAMNet has 3 outputs: scores, embeddings, spectrogram
        val scores = Array(1) { FloatArray(521) }
        val embeddings = Array(1) { FloatArray(1024) }
        val spectrogram = Array(1) { Array(96) { FloatArray(64) } }

        val outputMap = mapOf(
            0 to scores,
            1 to embeddings,
            2 to spectrogram
        )

        interpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputMap)

        // Find max score
        var maxIdx = -1
        var maxScore = -1f
        for (i in scores[0].indices) {
            if (scores[0][i] > maxScore) {
                maxScore = scores[0][i]
                maxIdx = i
            }
        }
        
        return ClassificationResult(maxIdx, maxScore)
    }

    fun close() {
        interpreter.close()
    }
}
