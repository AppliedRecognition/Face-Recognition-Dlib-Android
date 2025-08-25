//
// Created by Jakub Dolejs on 22/08/2025.
//
#pragma once
#include <jni.h>
#include <string>
#include <vector>

namespace dlib { class shape_predictor; }  // forward declare the type

class DlibLandmarks {
public:
    explicit DlibLandmarks(const std::string& modelPath);
    ~DlibLandmarks(); // defaulted in .cpp

    std::vector<float> createAlignedFace(
            JNIEnv* env,
            jobject bitmap,              // Android Bitmap in RGBA_8888
            jint left, jint top, jint right, jint bottom, // face bounds (inclusive)
            jint size,                   // chip size, e.g. 150
            jfloat paddingF              // e.g. 0.25f
    );

private:
    std::unique_ptr<dlib::shape_predictor> predictor_;
};

