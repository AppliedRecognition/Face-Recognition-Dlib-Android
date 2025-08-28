import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.vanniktech.publish)
    signing
}

version = "1.1.1"

android {
    namespace = "com.appliedrec.facerecognition.dlib"
    compileSdk = 36

    defaultConfig {
        minSdk = 26

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        consumerProguardFiles("consumer-rules.pro")

        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++17 -frtti -fexceptions"
            }
        }
        ndk {
            abiFilters += listOf("x86_64", "arm64-v8a")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlin {
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_11)
        }
    }
    packaging {
        jniLibs {
            pickFirsts.add("lib/arm64-v8a/libonnxruntime.so")
            pickFirsts.add("lib/x86_64/libonnxruntime.so")
        }
    }
    externalNativeBuild {
        cmake {
            path("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    ndkVersion = "28.2.13676358"
}

dependencies {

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.verid3.serialization)
    implementation(libs.kotlinx.coroutines.core)
    api(libs.verid3.common)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.face.detection.retinaface)
    androidTestImplementation(libs.verid3.serialization)
}


mavenPublishing {
    coordinates("com.appliedrec", "face-recognition-dlib")
    pom {
        name.set("Dlib face recognition for Ver-ID")
        description.set("Face recognition implementation for Ver-ID SDK using Dlib")
        url.set("https://github.com/AppliedRecognition/Face-Recognition-Dlib-Android")
        licenses {
            license {
                name.set("Commercial")
                url.set("https://raw.githubusercontent.com/AppliedRecognition/Face-Recognition-Dlib-Android/main/LICENCE.txt")
            }
        }
        developers {
            developer {
                id.set("appliedrec")
                name.set("Applied Recognition")
                email.set("support@appliedrecognition.com")
            }
        }
        scm {
            connection.set("scm:git:git://github.com/AppliedRecognition/Face-Recognition-Dlib-Android.git")
            developerConnection.set("scm:git:ssh://github.com/AppliedRecognition/Face-Recognition-Dlib-Android.git")
            url.set("https://github.com/AppliedRecognition/Face-Recognition-Dlib-Android")
        }
    }
    publishToMavenCentral(automaticRelease = true)
}

signing {
    useGpgCmd()
    sign(publishing.publications)
}