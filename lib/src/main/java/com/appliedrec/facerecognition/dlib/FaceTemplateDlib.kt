package com.appliedrec.facerecognition.dlib

import com.appliedrec.verid3.common.FaceTemplate

class FaceTemplateDlib(
    data: FloatArray
) : FaceTemplate<FaceTemplateVersionV16, FloatArray>(FaceTemplateVersionV16, data) {
    override fun equals(other: Any?): Boolean {
        return other is FaceTemplateDlib && other.data.contentEquals(data)
    }

    override fun hashCode(): Int {
        return 31 * version.hashCode() + data.contentHashCode()
    }
}