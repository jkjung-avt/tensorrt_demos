#ifndef _YOLO_LAYER_H
#define _YOLO_LAYER_H

#include <cassert>
#include <vector>
#include <string>
#include <iostream>
#include "math_constants.h"
#include "NvInfer.h"
#include "NvInferVersion.h"

#define MAX_ANCHORS 6

#if NV_TENSORRT_MAJOR == 8
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
    class YoloLayerPlugin: public IPluginV2IOExt
    {
        public:
            YoloLayerPlugin(int yolo_width, int yolo_height, int num_anchors, float* anchors, int num_classes, int input_width, int input_height, float scale_x_y, int new_coords);
            YoloLayerPlugin(const void* data, size_t length);

            ~YoloLayerPlugin() override = default;

            int getNbOutputs() const NOEXCEPT override
            {
                return 1;
            }

            Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) NOEXCEPT override;

            int initialize() NOEXCEPT override;

            void terminate() NOEXCEPT override;

            virtual size_t getWorkspaceSize(int maxBatchSize) const NOEXCEPT override { return 0;}

#if NV_TENSORRT_MAJOR == 8
            virtual int32_t enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) NOEXCEPT override;
#else
            virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override;
#endif

            virtual size_t getSerializationSize() const NOEXCEPT override;

            virtual void serialize(void* buffer) const NOEXCEPT override;

            bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const NOEXCEPT override {
                return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
            }

            const char* getPluginType() const NOEXCEPT override;

            const char* getPluginVersion() const NOEXCEPT override;

            void destroy() NOEXCEPT override;

            IPluginV2IOExt* clone() const NOEXCEPT override;

            void setPluginNamespace(const char* pluginNamespace) NOEXCEPT override;

            const char* getPluginNamespace() const NOEXCEPT override;

            DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const NOEXCEPT override;

            bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const NOEXCEPT override;

            bool canBroadcastInputAcrossBatch(int inputIndex) const NOEXCEPT override;

            void attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) NOEXCEPT override;

#if NV_TENSORRT_MAJOR == 8
            void configurePlugin(PluginTensorDesc const* in, int32_t nbInput, PluginTensorDesc const* out, int32_t nbOutput) NOEXCEPT override;
#else
            void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) override TRTNOEXCEPT;
#endif

            void detachFromContext() NOEXCEPT override;

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

#if NV_TENSORRT_MAJOR < 8
        protected:
            using IPluginV2IOExt::configurePlugin;
#endif
    };

    class YoloPluginCreator : public IPluginCreator
    {
        public:
            YoloPluginCreator();

            ~YoloPluginCreator() override = default;

            const char* getPluginName() const NOEXCEPT override;

            const char* getPluginVersion() const NOEXCEPT override;

            const PluginFieldCollection* getFieldNames() NOEXCEPT override;

            IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* fc) NOEXCEPT override;

            IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) NOEXCEPT override;

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
