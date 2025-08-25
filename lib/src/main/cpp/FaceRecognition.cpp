#include <jni.h>
#include <android/bitmap.h>
#include <string>
#include <vector>
#include <cassert>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include "DlibLandmarks.h"

//
// Created by Jakub Dolejs on 22/08/2025.
//

template <typename F>
class FinalAction {
public:
    explicit FinalAction(F f) : f_(std::move(f)), active_(true) {}
    ~FinalAction() { if (active_) f_(); }

    void dismiss() { active_ = false; }

private:
    F f_;
    bool active_;
};

template <typename F>
FinalAction<F> finally(F f) {
    return FinalAction<F>(std::move(f));
}

class FaceRecognition {
public:
    FaceRecognition(const std::string& modelPath, const std::string& landmarksModelPath)
        : env_(ORT_LOGGING_LEVEL_WARNING, "FaceRecognitionDlib"),
          sessionOptions_()
    {
        dlibLandmarks = std::make_unique<DlibLandmarks>(landmarksModelPath);
        sessionOptions_.SetInterOpNumThreads(1);
        sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        session_ = std::make_unique<Ort::Session>(env_, modelPath.c_str(), sessionOptions_);
        loadModelIO();
    }
    std::vector<float> createFaceRecognitionTemplate(std::vector<float>& inputData) {
        std::vector<int64_t> inputShape = {1, 150, 150, 3};
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
                memoryInfo, inputData.data(), inputData.size(),
                inputShape.data(), inputShape.size());

        auto outputTensors = session_->Run(
                Ort::RunOptions{nullptr},
                inputNames_.data(), &inputTensor, 1,
                outputNames_.data(), outputNames_.size());

        auto& output = outputTensors[0];
        const auto* outputData = output.GetTensorData<float>();
        auto typeInfo = output.GetTensorTypeAndShapeInfo();
        auto shape = typeInfo.GetShape();

        size_t total = 1;
        for (auto d : shape) total *= d;
        return {outputData, outputData + total};
    }
    std::unique_ptr<DlibLandmarks> dlibLandmarks;
private:
    Ort::Env env_;
    Ort::SessionOptions sessionOptions_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::vector<std::string> inputNamesStr_;
    std::vector<std::string> outputNamesStr_;
    std::vector<const char*> inputNames_;
    std::vector<const char*> outputNames_;


    void loadModelIO() {
        size_t inputCount  = session_->GetInputCount();
        size_t outputCount = session_->GetOutputCount();

        inputNamesStr_.clear();
        outputNamesStr_.clear();
        inputNames_.clear();
        outputNames_.clear();

        // Collect input names
        for (size_t i = 0; i < inputCount; ++i) {
            Ort::AllocatedStringPtr name = session_->GetInputNameAllocated(i, allocator_);
            inputNamesStr_.emplace_back(name.get()); // copy into std::string
        }
        for (auto& s : inputNamesStr_) {
            inputNames_.push_back(s.c_str());
        }

        // Collect output names
        for (size_t i = 0; i < outputCount; ++i) {
            Ort::AllocatedStringPtr name = session_->GetOutputNameAllocated(i, allocator_);
            outputNamesStr_.emplace_back(name.get());
        }
        for (auto& s : outputNamesStr_) {
            outputNames_.push_back(s.c_str());
        }
    }
};

extern "C"
JNIEXPORT jlong JNICALL
Java_com_appliedrec_facerecognition_dlib_FaceRecognitionDlib_createNativeContext(JNIEnv *env, jobject thiz, jstring landmarks_model_file, jstring model_file) {
    const char *modelPathCStr = env->GetStringUTFChars(model_file, nullptr);
    std::string modelPath(modelPathCStr);
    env->ReleaseStringUTFChars(model_file, modelPathCStr);
    const char *landmarksModelPathCStr = env->GetStringUTFChars(landmarks_model_file, nullptr);
    std::string landmarksModelPath(landmarksModelPathCStr);
    env->ReleaseStringUTFChars(landmarks_model_file, landmarksModelPathCStr);

    auto *recognition = new FaceRecognition(modelPath, landmarksModelPath);
    return reinterpret_cast<jlong>(recognition);
}
extern "C"
JNIEXPORT void JNICALL
Java_com_appliedrec_facerecognition_dlib_FaceRecognitionDlib_destroyNativeContext(JNIEnv *env,
                                                                                  jobject thiz,
                                                                                  jlong context) {
    auto *recognition = reinterpret_cast<FaceRecognition *>(context);
    delete recognition;
}

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_appliedrec_facerecognition_dlib_FaceRecognitionDlib_createFaceTemplateFromBitmap(
        JNIEnv *env, jobject thiz, jlong context, jobject bitmap, jint left, jint top, jint right, jint bottom, jint size, jfloat padding) {
    try {
        if (!env) {
            throw std::runtime_error("JNIEnv is null");
        }
        auto *recognition = reinterpret_cast<FaceRecognition *>(context);
        if (!recognition) {
            throw std::runtime_error("Face recognition context is null");
        }
        std::vector<float> inputTensor = recognition->dlibLandmarks->createAlignedFace(env, bitmap, left, top, right, bottom, size, padding);
        std::vector<float> templateData = recognition->createFaceRecognitionTemplate(inputTensor);
        auto outSize = static_cast<jsize>(templateData.size());
        jfloatArray faceTemplate = env->NewFloatArray(outSize);
        if (!faceTemplate) {
            throw std::runtime_error("Failed to allocate jfloatArray");
        }
        env->SetFloatArrayRegion(faceTemplate, 0, outSize, templateData.data());
        return faceTemplate;
    } catch (const std::exception &e) {
        if (env) {
            env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
        } else {
            throw e;
        }
        return nullptr;
    }
}