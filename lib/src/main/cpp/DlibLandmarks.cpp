#include "DlibLandmarks.h"

#include <android/bitmap.h>
#include <stdexcept>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>

// dlib includes only in the .cpp
#include <dlib/image_processing/shape_predictor.h>
#include <dlib/image_processing.h>
#include <dlib/image_transforms.h>
#include <dlib/array2d.h>
#include <dlib/pixel.h>
#include <dlib/geometry/rectangle.h>
#include <dlib/serialize.h>

using namespace dlib;

namespace {
    inline long clamp_long(double v, long lo, long hi) {
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        return static_cast<long>(llround(v));
    }

    // small RAII helper to ensure unlock
    struct PixelsLock {
        JNIEnv* env{};
        jobject bitmap{};
        bool locked{false};
        explicit PixelsLock(JNIEnv* e, jobject bm) : env(e), bitmap(bm) {}
        void unlock() {
            if (locked) { AndroidBitmap_unlockPixels(env, bitmap); locked = false; }
        }
        ~PixelsLock() { unlock(); }
    };
}

DlibLandmarks::DlibLandmarks(const std::string& modelPath) {
    predictor_ = std::make_unique<dlib::shape_predictor>();
    try {
        deserialize(modelPath) >> *predictor_;
    } catch (const std::exception& e) {
        predictor_.reset();
        throw std::runtime_error(std::string("Failed to load predictor: ") + e.what());
    }
}

DlibLandmarks::~DlibLandmarks() = default;

std::vector<float> DlibLandmarks::createAlignedFace(
        JNIEnv* env,
        jobject bitmap,
        jint left, jint top, jint right, jint bottom,
        jint size,
        jfloat paddingF
) {
    if (!predictor_) {
        throw std::runtime_error("Predictor not loaded");
    }
    if (bitmap == nullptr || size <= 1) {
        throw std::runtime_error("Invalid bitmap or size");
    }

    // 1) Read Android Bitmap pixels
    AndroidBitmapInfo info{};
    void* pixels = nullptr;
    if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS) {
        throw std::runtime_error("AndroidBitmap_getInfo failed");
    }
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        throw std::runtime_error("Bitmap must be RGBA_8888");
    }
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS) {
        throw std::runtime_error("AndroidBitmap_lockPixels failed");
    }
    PixelsLock lock(env, bitmap);
    lock.locked = true;

    const int width  = static_cast<int>(info.width);
    const int height = static_cast<int>(info.height);
    const int stride = static_cast<int>(info.stride);
    const uint8_t* src = static_cast<const uint8_t*>(pixels);

    // 2) Copy into dlib RGB image (unpremultiply alpha to avoid dark edges)
    dlib::array2d<rgb_pixel> img;
    img.set_size(height, width);
    for (int y = 0; y < height; ++y) {
        const uint32_t* row = reinterpret_cast<const uint32_t*>(src + y * stride);
        for (int x = 0; x < width; ++x) {
            const uint32_t rgba = row[x];
            uint8_t a = (rgba >> 24) & 0xff;
            uint8_t r = (rgba >> 16) & 0xff;
            uint8_t g = (rgba >>  8) & 0xff;
            uint8_t b = (rgba      ) & 0xff;

            if (a != 0 && a != 255) {
                r = static_cast<uint8_t>((static_cast<uint32_t>(r) * 255 + (a >> 1)) / a);
                g = static_cast<uint8_t>((static_cast<uint32_t>(g) * 255 + (a >> 1)) / a);
                b = static_cast<uint8_t>((static_cast<uint32_t>(b) * 255 + (a >> 1)) / a);
            }
            img[y][x] = rgb_pixel(r, g, b);
        }
    }

    // pixels unlocked automatically by RAII at end of scope

    // 3) Clamp face rectangle to image bounds
    const long L = clamp_long(left,   0, width  - 1);
    const long T = clamp_long(top,    0, height - 1);
    const long R = clamp_long(right,  0, width  - 1);
    const long B = clamp_long(bottom, 0, height - 1);
    if (R <= L || B <= T) {
        throw std::runtime_error("Face rectangle out of bounds");
    }

    // 4) Predict 5-point landmarks and compute chip details
    full_object_detection shape;
    try {
        shape = (*predictor_)(img, rectangle(L, T, R, B)); // pointer call
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Predictor failed: ") + e.what());
    }
    if (shape.num_parts() != 5) {
        throw std::runtime_error("Predictor did not return 5 points");
    }

    const double padding = static_cast<double>(paddingF);
    chip_details cd = get_face_chip_details(shape, size, padding);

    // 5) Extract aligned chip (RGB, size×size)
    dlib::matrix<rgb_pixel> chip;
    extract_image_chip(img, cd, chip); // chip.nr() == size, chip.nc() == size

    // 6) Pack to interleaved RGB, row-major, [0,1]
    const int H = size, W = size;
    std::vector<float> buffer(static_cast<size_t>(H) * W * 3);
    size_t idx = 0;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const rgb_pixel& p = chip(y, x);
            buffer[idx++] = static_cast<float>(p.red)   / 255.0f;
            buffer[idx++] = static_cast<float>(p.green) / 255.0f;
            buffer[idx++] = static_cast<float>(p.blue)  / 255.0f;
        }
    }
    return buffer;
}