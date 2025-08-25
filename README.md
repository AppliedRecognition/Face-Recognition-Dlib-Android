# Face Recognition for Ver-ID SDK using Dlib

## Installation

Add the following dependency in your **build.gradle.kts** file:

```kotlin
implementation(platform("com.appliedrec:verid-bom:3000.1.0"))
implementation("com.appliedrec:face-recognition-dlib")
```

## Usage

The library implements the [FaceRecognition](https://github.com/AppliedRecognition/Ver-ID-Common-Types-Android/blob/main/lib/src/main/java/com/appliedrec/verid3/common/FaceRecognition.kt) interface from the Ver-ID-Common-Types package. This makes it compatible with the Ver-ID SDK.

### Example: Create a face template from an image and face

Use a class that implements the [FaceDetection](https://github.com/AppliedRecognition/Ver-ID-Common-Types-Android/blob/main/lib/src/main/java/com/appliedrec/verid3/common/FaceDetection.kt) interface from the Ver-ID-Common-Types package. For example:

```kotlin
implementation("com.appliedrec:face-detection-retinaface")
```

To create an instance of [Image](https://github.com/AppliedRecognition/Ver-ID-Common-Types-Android/blob/main/lib/src/main/java/com/appliedrec/verid3/common/Image.kt) import the Ver-ID serialization library by adding the following dependency:

```kotlin
implementation("com.appliedrec:verid-serialization")
```

```kotlin
suspend fun detectFacesForRecognition(
    context: Context, 
    uri: Uri, 
    faceDetection: FaceDetection
): List<FaceTemplateArcFace> {
    // 1. Read image from URL
    val bitmap = context.contentResolver.openInputStream(uri)
        .use(BitmapFactory::decodeStream)
    // 2. Convert bitmap to Ver-ID image
    val image = Image.fromBitmap(bitmap)
    // 3. Detect up to 5 faces
    val faces = faceDetection.detectFacesInImage(image, 5)
    // 4. Create face recognition instance
    val templates = FaceRecognitionArcFace(context).use {
        // 5. Extract face templates
        it.createFaceRecognitionTemplates(faces, image) 
    }
    // 6. Return face templates
    return templates
}
```