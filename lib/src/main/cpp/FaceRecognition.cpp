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
        Ort::MemoryInfo memoryInfo =
                Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
                memoryInfo, inputData.data(), inputData.size(),
                inputShape.data(), inputShape.size());

        auto outputTensors = session_->Run(
                Ort::RunOptions{nullptr},
                inputNames_.data(), &inputTensor, 1,
                outputNames_.data(), outputNames_.size());

        const auto& output = outputTensors[0];
        const auto* outputData = output.GetTensorData<float>();

        auto typeInfo = output.GetTensorTypeAndShapeInfo();
        auto shape    = typeInfo.GetShape();

        size_t total = 1;
        for (auto d : shape) total *= static_cast<size_t>(d);

        if (mean_.size() != total) {
            throw std::invalid_argument("mean vector length does not match model output length");
        }

        std::vector<float> templ(total);
        double sumsq = 0.0;

        for (size_t i = 0; i < total; ++i) {
            float v = outputData[i] - mean_[i];
            templ[i] = v;
            sumsq += static_cast<double>(v) * static_cast<double>(v);
        }

        auto norm = static_cast<float>(std::sqrt(sumsq));
        if (norm > 0.0f) {
            float inv = 1.0f / norm;
            for (float& x : templ) x *= inv;
        } else {
            for (float& x : templ) x = 0.0f;
        }

        return templ;
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
    inline static const std::vector<float> mean_ = {
            -0.1090f,0.0742f,0.0517f,-0.0375f,-0.0994f,-0.0329f,-0.0151f,-0.1079f,
            0.1378f,-0.0923f,0.2127f,-0.0365f,-0.2286f,-0.0445f,-0.0124f,0.1445f,

            -0.1405f,-0.1195f,-0.1007f,-0.0680f,0.0226f,0.0363f,0.0200f,0.0452f,
            -0.1115f,-0.3154f,-0.0861f,-0.0857f,0.0347f,-0.0633f,-0.0212f,0.0540f,

            -0.1759f,-0.0452f,0.0316f,0.0744f,-0.0404f,-0.0740f,0.1908f,0.0074f,
            -0.1750f,0.0011f,0.0608f,0.2374f,0.1846f,0.0242f,0.0188f,-0.0836f,

            0.1072f,-0.2355f,0.0457f,0.1380f,0.0863f,0.0695f,0.0580f,-0.1418f,
            0.0218f,0.1214f,-0.1886f,0.0353f,0.0607f,-0.0795f,-0.0504f,-0.0594f,

            0.2046f,0.1072f,-0.1132f,-0.1250f,0.1547f,-0.1550f,-0.0512f,0.0616f,
            -0.1190f,-0.1681f,-0.2682f,0.0425f,0.3917f,0.1305f,-0.1568f,0.0228f,

            -0.0711f,-0.0270f,0.0505f,0.0680f,-0.0632f,-0.0314f,-0.0845f,0.0344f,
            0.1964f,-0.0246f,-0.0093f,0.2210f,0.0085f,0.0091f,0.0245f,0.0508f,

            -0.0919f,-0.0210f,-0.1102f,-0.0185f,0.0413f,-0.0808f,0.0042f,0.0965f,
            -0.1852f,0.1417f,-0.0140f,-0.0215f,0.0028f,-0.0162f,-0.0834f,-0.0259f,

            0.1400f,-0.2383f,0.1883f,0.1652f,0.0180f,0.1376f,0.0564f,0.0727f,
            -0.0131f,-0.0284f,-0.1567f,-0.0831f,0.0615f,-0.0196f,0.0417f,0.0311f
    };

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