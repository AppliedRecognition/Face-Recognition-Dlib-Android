package com.appliedrec.facerecognition.dlib

import android.graphics.Bitmap
import android.graphics.Rect
import com.appliedrec.verid3.common.SuspendingCloseable

class DlibLandmarks(modelPath: String) : SuspendingCloseable {

    var nativeContext: Long?

    init {
        nativeContext = createNativeContext(modelPath)
    }

    fun createAlignedFace(image: Bitmap, rect: Rect): FloatArray {
        return nativeContext?.let { context ->
            createAlignedFace(context, image, rect.left, rect.top, rect.right, rect.bottom, 150, 0.25f)
        } ?: throw IllegalStateException("Library closed")
    }

    private external fun createAlignedFace(
        context: Long,
        image: Bitmap,
        left: Int, top: Int, right: Int, bottom: Int,
        size: Int,
        padding: Float
    ): FloatArray

    private external fun createNativeContext(modelPath: String): Long

    private external fun destroyNativeContext(context: Long)

    override suspend fun close() {
        super.close()
        nativeContext?.let { destroyNativeContext(it) }
        nativeContext = null
    }
}