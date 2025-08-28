package com.appliedrec.facerecognition.dlib

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.appliedrec.verid3.common.FaceTemplate
import com.appliedrec.verid3.common.Image
import com.appliedrec.verid3.common.serialization.fromBitmap
import com.appliedrec.verid3.common.use
import com.appliedrec.verid3.facedetection.retinaface.FaceDetectionRetinaFace
import kotlinx.coroutines.runBlocking
import org.junit.Assert

import org.junit.Test
import org.junit.runner.RunWith

import java.io.InputStream
import kotlin.math.sqrt

/**
 * Instrumented test, which will execute on an Android device.
 *
 * See [testing documentation](http://d.android.com/tools/testing).
 */
@RunWith(AndroidJUnit4::class)
class ExampleInstrumentedTest {

    @Test
    fun testCreateFaceRecognition(): Unit = runBlocking {
        FaceRecognitionDlib.create(InstrumentationRegistry.getInstrumentation().targetContext)
    }

    @Test
    fun testExtractFaceTemplates(): Unit = runBlocking {
        FaceDetectionRetinaFace.create(InstrumentationRegistry.getInstrumentation().targetContext).use { faceDetection ->
            FaceRecognitionDlib.create(InstrumentationRegistry.getInstrumentation().targetContext)
                .use { recognition ->
                    for (fileName in listOf(
                        "subject1-01.jpg",
                        "subject1-02.jpg",
                        "subject2-01.jpg"
                    )) {
                        InstrumentationRegistry.getInstrumentation().context.assets.open(fileName)
                            .use { inputStream ->
                                val bitmap = BitmapFactory.decodeStream(inputStream)
                                val image = Image.fromBitmap(bitmap)
                                val face = faceDetection.detectFacesInImage(image, 1).first()
                                val templates = recognition.createFaceRecognitionTemplates(listOf(face), image)
                                Assert.assertEquals(templates.size, 1)
                            }
                    }
                }
        }
    }

    @Test
    fun testFaceTemplateIsNormalized(): Unit = runBlocking {
        FaceDetectionRetinaFace.create(InstrumentationRegistry.getInstrumentation().targetContext).use { faceDetection ->
            FaceRecognitionDlib.create(InstrumentationRegistry.getInstrumentation().targetContext)
                .use { recognition ->
                    for (fileName in listOf(
                        "subject1-01.jpg",
                        "subject1-02.jpg",
                        "subject2-01.jpg"
                    )) {
                        InstrumentationRegistry.getInstrumentation().context.assets.open(fileName)
                            .use { inputStream ->
                                val bitmap = BitmapFactory.decodeStream(inputStream)
                                val image = Image.fromBitmap(bitmap)
                                val face = faceDetection.detectFacesInImage(image, 1).first()
                                val templates = recognition.createFaceRecognitionTemplates(listOf(face), image)
                                Assert.assertEquals(templates.size, 1)
                                val norm = norm(templates.first().data)
                                Assert.assertEquals(1f, norm, 0.01f)
                            }
                    }
                }
        }
    }

    private fun innerProduct(v1: FloatArray, v2: FloatArray): Float {
        return v1.zip(v2) { a, b -> a * b }.sum()
    }

    private fun norm(v: FloatArray): Float {
        return sqrt(innerProduct(v, v))
    }

    @Test
    fun testCompareSubjectFaces(): Unit = runBlocking {
        FaceDetectionRetinaFace.create(InstrumentationRegistry.getInstrumentation().targetContext).use { faceDetection ->
            FaceRecognitionDlib.create(InstrumentationRegistry.getInstrumentation().targetContext)
                .use { recognition ->
                    val threshold = recognition.defaultThreshold
                    val subjectTemplates = mutableListOf<Pair<String,FaceTemplate<FaceTemplateVersionV16, FloatArray>>>()
                    for (fileName in listOf(
                        "subject1" to "-01.jpg",
                        "subject1" to "-02.jpg",
                        "subject2" to "-01.jpg"
                    )) {
                        InstrumentationRegistry.getInstrumentation().context.assets.open(fileName.first + fileName.second)
                            .use { inputStream ->
                                val bitmap = BitmapFactory.decodeStream(inputStream)
                                val image = Image.fromBitmap(bitmap)
                                val face = faceDetection.detectFacesInImage(image, 1).first()
                                val templates = recognition.createFaceRecognitionTemplates(listOf(face), image)
                                Assert.assertEquals(templates.size, 1)
                                subjectTemplates.add(fileName.first to templates[0])
                            }
                    }
                    for ((subject1, template1) in subjectTemplates) {
                        for ((subject2, template2) in subjectTemplates) {
                            if (template1 == template2) {
                                continue
                            }
                            val score = recognition.compareFaceRecognitionTemplates(listOf(template1), template2).first()
                            if (subject1 == subject2) {
                                Assert.assertTrue(score >= threshold)
                            } else {
                                Assert.assertTrue(score < threshold)
                            }
                        }
                    }
                }
        }
    }

//    @Test
//    fun generateLandmarkPredictorFileHash() {
//        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
//        val assetManager = appContext.assets
//        val inputStream = assetManager.open("shape_predictor_5_face_landmarks.dat")
//        val hash = inputStream.hash()
//        Log.d("Hash", hash)
//    }
}

fun InputStream.hash(bufferSize: Int = 8192): String {
    val buf = ByteArray(bufferSize)
    var hash = -3750763034362895579L  // FNV-1a 64-bit offset basis
    val prime = 1099511628211L      // FNV-1a prime

    while (true) {
        val read = this.read(buf)
        if (read == -1) break
        for (i in 0 until read) {
            hash = hash xor (buf[i].toLong() and 0xff)
            hash *= prime
        }
    }
    return java.lang.Long.toUnsignedString(hash, 16).padStart(16, '0')
}