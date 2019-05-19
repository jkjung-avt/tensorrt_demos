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

#define MAX_BINDINGS 3

using namespace nvinfer1;
using namespace nvcaffeparser1;

namespace mtcnn_trtnet {

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
            void* data() const { return _mem; }
            std::size_t size() const { return _s; }
            DataType type () const { return DataType::kFLOAT; } // not used
            void destroy() { free(_mem); }
        private:
            void *_mem{nullptr};
            std::size_t _s;
    };

    class TrtMtcnnDet
    {
        public:
            TrtMtcnnDet();
            // init from engine file
            void initEngine(std::string filePath, int dataDims[3], int prob1Dims[3], int boxesDims[3]);
            void setBatchSize(int value);
            int getBatchSize();
            void forward(float *imgs, float *probs, float *boxes);
            void destroy();

        private:
            Logger _gLogger;
            IHostMemory *_gieModelStream{nullptr};
            IRuntime *_runtime;
            ICudaEngine *_engine;
            IExecutionContext *_context;
            cudaStream_t _stream;
            void *_gpu_buffers[MAX_BINDINGS];
            int _blob_sizes[MAX_BINDINGS];
            int _inputIdx;  // index to the input binding
            int _batchsize = 0;

            void initEngine(std::string filePath);
    };

}  // namespace mtcnn_trtnet

#endif  // __TRTNET_H__
