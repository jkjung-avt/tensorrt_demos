#ifndef _YOLO_LAYER_H
#define _YOLO_LAYER_H

#include <cassert>
#include <vector>
#include <string>
#include <iostream>
#include "math_constants.h"
#include "NvInfer.h"

#define MAX_ANCHORS 6

#if NV_TENSORRT_MAJOR >= 8
#define NOEXCEPT noexcept
#else
#define NOEXCEPT
#endif

#define CHECK(status)                                           \
    do {                                                        \
        auto ret = status;                                      \
        if (ret != 0) {                                         \
            std::cerr << "Cuda failure in file '" << __FILE__   \
                      << "' line " << __LINE__                  \
                      << ": " << ret << std::endl;              \
            abort();                                            \
        }                                                       \
    } while (0)

namespace Yolo
{
    static constexpr float IGNORE_THRESH = 0.01f;

    struct alignas(float) Detection {
        float bbox[4];  // x, y, w, h
        float det_confidence;
        float class_id;
        float class_confidence;
    };
}

namespace nvinfer1
{
    class YoloLayerPlugin: public IPluginV2DynamicExt
    {
        public:
            YoloLayerPlugin(int yolo_width, int yolo_height, int num_anchors, float* anchors, int num_classes, int input_width, int input_height, float scale_x_y, int new_coords);
            YoloLayerPlugin(const void* data, size_t length);

            ~YoloLayerPlugin() override = default;

            IPluginV2DynamicExt* clone() const NOEXCEPT override;

            int initialize() NOEXCEPT override { return 0; }

            void terminate() NOEXCEPT override;

            void destroy() NOEXCEPT override { delete this; }

            size_t getSerializationSize() const NOEXCEPT override;

            void serialize(void* buffer) const NOEXCEPT override;

            int getNbOutputs() const NOEXCEPT override { return 1; }

            DimsExprs getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) NOEXCEPT override;

            size_t getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const NOEXCEPT override { return 0; }

            bool supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) NOEXCEPT override { return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT; }

            const char* getPluginType() const NOEXCEPT override { return "YoloLayer_TRT"; }

            const char* getPluginVersion() const NOEXCEPT override { return "1"; }

            void setPluginNamespace(const char* pluginNamespace) NOEXCEPT override { mPluginNamespace = pluginNamespace; }

            const char* getPluginNamespace() const NOEXCEPT override { return mPluginNamespace; }

            DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const NOEXCEPT override { return DataType::kFLOAT; }

            void attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) NOEXCEPT override {}

            //using IPluginV2IOExt::configurePlugin;
            void configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) NOEXCEPT override {}

            void detachFromContext() NOEXCEPT override {}

#if NV_TENSORRT_MAJOR >= 8
            int32_t enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) NOEXCEPT override;
#else
            int enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) NOEXCEPT override;
#endif

        private:
            void forwardGpu(const float* const* inputs, float* output, cudaStream_t stream, int batchSize = 1);

            int mThreadCount = 64;
            int mYoloWidth, mYoloHeight, mNumAnchors;
            float mAnchorsHost[MAX_ANCHORS * 2];
            float *mAnchors;  // allocated on GPU
            int mNumClasses;
            int mInputWidth, mInputHeight;
            float mScaleXY;
            int mNewCoords = 0;

            const char* mPluginNamespace;
    };

    class YoloPluginCreator : public IPluginCreator
    {
        public:
            YoloPluginCreator();

            ~YoloPluginCreator() override = default;

            const char* getPluginName() const NOEXCEPT override;

            const char* getPluginVersion() const NOEXCEPT override;

            const PluginFieldCollection* getFieldNames() NOEXCEPT override;

            IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) NOEXCEPT override;

            IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) NOEXCEPT override;

            void setPluginNamespace(const char* libNamespace) NOEXCEPT override
            {
                mNamespace = libNamespace;
            }

            const char* getPluginNamespace() const NOEXCEPT override
            {
                return mNamespace.c_str();
            }

        private:
            static PluginFieldCollection mFC;
            static std::vector<PluginField> mPluginAttributes;
            std::string mNamespace;
    };

    REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);
};

#endif
