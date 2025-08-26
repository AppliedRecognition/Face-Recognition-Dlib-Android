package com.appliedrec.facerecognition.dlib

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import com.appliedrec.verid3.common.Face
import com.appliedrec.verid3.common.FaceRecognition
import com.appliedrec.verid3.common.FaceTemplate
import com.appliedrec.verid3.common.IImage
import com.appliedrec.verid3.common.serialization.toBitmap
import java.io.File

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

    override val defaultThreshold: Float = 0.91f

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
    ): FloatArray {
        // precompute ||q||^2
        var q2 = 0f
        for (v in template.data) q2 += v * v

        return faceRecognitionTemplates.map { x ->
            // dot(q, x)
            var dot = 0f
            for (i in template.data.indices) dot += template.data[i] * x.data[i]

            // ||x||^2
            var x2 = 0f
            for (v in x.data) x2 += v * v

            // d^2 = ||q - x||^2  (clamped to avoid tiny negative from roundoff)
            val d2 = kotlin.math.max(0f, q2 + x2 - 2f * dot)

            // similarity in [0,1]: 1 - d^2/4  (cosine mapped to 0..1)
            val s = 1f - 0.25f * d2

            // clamp to [0,1] for numerical safety
            s.coerceIn(0f, 1f)
        }.toFloatArray()
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