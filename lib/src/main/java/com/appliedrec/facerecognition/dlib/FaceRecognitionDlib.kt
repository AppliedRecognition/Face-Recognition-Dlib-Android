package com.appliedrec.facerecognition.dlib

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import com.appliedrec.verid3.common.Face
import com.appliedrec.verid3.common.FaceRecognition
import com.appliedrec.verid3.common.FaceTemplate
import com.appliedrec.verid3.common.IImage
import com.appliedrec.verid3.common.serialization.toBitmap
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import java.io.File
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

class FaceRecognitionDlib private constructor(dlibLandmarksModelPath: String, modelPath: String) : FaceRecognition<FaceTemplateVersionV16, FloatArray> {

    companion object {

        init {
            System.loadLibrary("FaceRecognitionDlib")
        }

        @JvmStatic
        suspend fun create(context: Context): FaceRecognitionDlib {
            val landmarkPredictorFileName = "shape_predictor_5_face_landmarks_8cf06d8d2c988ec6.dat"
            val landmarkPredictorFile = File(context.cacheDir, landmarkPredictorFileName)
            if (!landmarkPredictorFile.exists()) {
                landmarkPredictorFile.outputStream().use { outputStream ->
                    context.assets.open(landmarkPredictorFileName).use { inputStream ->
                        inputStream.copyTo(outputStream)
                    }
                }
            }
            val modelFileName = "dlib_face_recognition_resnet_v1.onnx"
            val modelFile = File(context.cacheDir, modelFileName)
            if (!modelFile.exists()) {
                modelFile.outputStream().use { outputStream ->
                    context.assets.open(modelFileName).use { inputStream ->
                        inputStream.copyTo(outputStream)
                    }
                }
            }
            return FaceRecognitionDlib(landmarkPredictorFile.absolutePath, modelFile.absolutePath)
        }
    }

    override val version: FaceTemplateVersionV16 = FaceTemplateVersionV16

    override val defaultThreshold: Float = 0.8f

    private var nativeContext: Long?

    init {
        nativeContext = createNativeContext(dlibLandmarksModelPath, modelPath)
    }

    override suspend fun createFaceRecognitionTemplates(
        faces: List<Face>,
        image: IImage
    ): List<FaceTemplate<FaceTemplateVersionV16, FloatArray>> {
        return nativeContext?.let { context ->
            faces.map { face ->
                val rect = Rect()
                face.bounds.round(rect)
                val templateData = createFaceTemplateFromBitmap(
                    context,
                    image.toBitmap(),
                    rect.left, rect.top, rect.right, rect.bottom,
                    150, 0.25f
                )
                FaceTemplateDlib(templateData)
            }
        } ?: throw IllegalStateException("Library closed")
    }

    override suspend fun compareFaceRecognitionTemplates(
        faceRecognitionTemplates: List<FaceTemplate<FaceTemplateVersionV16, FloatArray>>,
        template: FaceTemplate<FaceTemplateVersionV16, FloatArray>
    ): FloatArray = coroutineScope {
        require(faceRecognitionTemplates.all { it.data.size == template.data.size }) {
            "Face recognition templates must have the same length"
        }
        val a = template.data
        faceRecognitionTemplates.map { it.data }
            .chunked(100)
            .map { chunk ->
                async {
                    FloatArray(chunk.size) { idx ->
                        val b = chunk[idx]
                        val cos = innerProduct(a, b)
                        ((cos + 1f) * 0.5f).coerceIn(0f, 1f)
                    }
                }
            }
            .awaitAll()
            .reduce { acc, arr -> acc + arr }
    }

    private fun innerProduct(v1: FloatArray, v2: FloatArray): Float {
        return v1.zip(v2) { a, b -> a * b }.sum()
    }

    private fun norm(v: FloatArray): Float {
        return sqrt(innerProduct(v, v))
    }

    override suspend fun close() {
        super.close()
        nativeContext?.let { destroyNativeContext(it) }
        nativeContext = null
    }

    private external fun createFaceTemplateFromBitmap(context: Long, bitmap: Bitmap, left: Int, top: Int, right: Int, bottom: Int, size: Int, paddingF: Float): FloatArray

    private external fun createNativeContext(dlibLandmarksModelPath: String, modelPath: String): Long

    private external fun destroyNativeContext(context: Long)
}