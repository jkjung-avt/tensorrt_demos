// trtNet.h
#ifndef __TRTNET_H__
#define __TRTNET_H__

#include <cassert>
#include <iostream>
#include <cstring>
#include <sstream>
#include <fstream>
#include <cuda_runtime_api.h>

#include "NvInfer.h"
#include "NvCaffeParser.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;

namespace trtnet {

    class Logger : public ILogger
    {
        void log(Severity severity, const char *msg) override
        {
            if (severity != Severity::kINFO)
                 std::cout << msg << std::endl;
        }
    };

    class IHostMemoryFromFile : public IHostMemory
    {
        public:
            IHostMemoryFromFile(std::string filename) {
                std::ifstream infile(filename, std::ifstream::binary |
                                               std::ifstream::ate);
                _s = infile.tellg();
                infile.seekg(0, std::ios::beg);
                _mem = malloc(_s);
                infile.read(reinterpret_cast<char*>(_mem), _s);
            }
#if NV_TENSORRT_MAJOR <= 5
            void* data() const { return _mem; }
            std::size_t size() const { return _s; }
            DataType type () const { return DataType::kFLOAT; } // not used
            void destroy() { free(_mem); }
#else  // NV_TENSORRT_MAJOR
            void* data() const noexcept { return _mem; }
            std::size_t size() const noexcept { return _s; }
            DataType type () const noexcept { return DataType::kFLOAT; } // not used
            void destroy() noexcept { free(_mem); }
#endif // NV_TENSORRT_MAJOR
        private:
            void *_mem{nullptr};
            std::size_t _s;
    };

    class TrtGooglenet
    {
        public:
            TrtGooglenet();
            // init from engine file
            void initEngine(std::string filePath, int dataDims[3], int probDims[3]);
            void forward(float *imgs, float *prob);
            void destroy();

        private:
            Logger _gLogger;
            IHostMemory *_gieModelStream{nullptr};
            IRuntime *_runtime;
            ICudaEngine *_engine;
            IExecutionContext *_context;
            cudaStream_t _stream;
            void *_gpu_buffers[2];
            int _blob_sizes[2];
	        int _binding_data;
	        int _binding_prob;

            void _initEngine(std::string filePath);
    };

    class TrtMtcnnDet
    {
        public:
            TrtMtcnnDet();
            // init from engine file
            void initDet1(std::string filePath, int dataDims[3], int prob1Dims[3], int boxesDims[3]);
            void initDet2(std::string filePath, int dataDims[3], int prob1Dims[3], int boxesDims[3]);
            void initDet3(std::string filePath, int dataDims[3], int prob1Dims[3], int boxesDims[3], int marksDims[3]);
            void setBatchSize(int value);
            int  getBatchSize();
            void forward(float *imgs, float *probs, float *boxes, float *);
            void destroy();

        private:
            Logger _gLogger;
            IHostMemory *_gieModelStream{nullptr};
            IRuntime *_runtime;
            ICudaEngine *_engine;
            IExecutionContext *_context;
            cudaStream_t _stream;
            void *_gpu_buffers[4];
            int _blob_sizes[4];
            int _num_bindings = 0;
	        int _binding_data;
	        int _binding_prob1;
	        int _binding_boxes;
	        int _binding_marks;
            int _batchsize = 0;

            void _initEngine(std::string filePath, const char *dataName, const char *prob1Name, const char *boxesName, const char *marksName);
            void _setBlobSizes(int dataDims[3], int prob1Dims[3], int boxesDims[3]);
    };

}  // namespace trtnet

#endif  // __TRTNET_H__
