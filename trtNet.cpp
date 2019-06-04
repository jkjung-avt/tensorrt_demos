// trtNet.cpp

#include "trtNet.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;

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

#define my_assert(EXP, MSG)                                     \
    do {                                                        \
        if (!(EXP)) {                                           \
            std::cerr << "Assertion fail in file '" << __FILE__ \
                      << "' line " << __LINE__                  \
                      << ": " << (MSG) << std:: endl;           \
            throw std::exception();                             \
        }                                                       \
    } while (0)


namespace trtnet {

    //
    // TrtGooglenet stuffs
    //

    TrtGooglenet::TrtGooglenet()
    {
        for (int i = 0; i < 2; i++) {
            _gpu_buffers[i] = nullptr;
        }
    }

    void TrtGooglenet::initEngine(std::string filePath)
    {
        _gieModelStream = new IHostMemoryFromFile(filePath);
        _runtime = createInferRuntime(_gLogger);
        my_assert(_runtime != nullptr, "_runtime is null");
        _engine = _runtime->deserializeCudaEngine(
            _gieModelStream->data(),
            _gieModelStream->size(),
            nullptr);
        my_assert(_engine != nullptr, "_enginer is null");
        my_assert(_engine->getNbBindings() == 2, "wrong number of bindings");
	    _binding_data = _engine->getBindingIndex("data");
        my_assert(_engine->bindingIsInput(_binding_data) == true, "bad type of binding 'data'");
	    _binding_prob = _engine->getBindingIndex("prob");
        my_assert(_engine->bindingIsInput(_binding_prob) == false, "bad type of binding 'prob'");
        _context = _engine->createExecutionContext();
        my_assert(_context != nullptr, "_context is null");
        _gieModelStream->destroy();
        CHECK(cudaStreamCreate(&_stream));
    }

    void TrtGooglenet::initEngine(std::string filePath, int dataDims[3], int probDims[3])
    {
        initEngine(filePath);
#if NV_TENSORRT_MAJOR == 3
        DimsCHW d;
        d = static_cast<DimsCHW&&>(_engine->getBindingDimensions(_binding_data));
        my_assert(d.nbDims == 3, "bad nbDims for 'data'");
        my_assert(d.c() == dataDims[0] && d.h() == dataDims[1] && d.w() == dataDims[2], "bad dims for 'data'");
        _blob_sizes[_binding_data] = d.c() * d.h() * d.w();
        d = static_cast<DimsCHW&&>(_engine->getBindingDimensions(_binding_prob));
        my_assert(d.nbDims == 3, "bad nbDims for 'prob'");
        my_assert(d.c() == prob1Dims[0] && d.h() == prob1Dims[1] && d.w() == prob1Dims[2], "bad dims for 'prob'");
        _blob_sizes[_binding_prob] = d.c() * d.h() * d.w();
#else   // NV_TENSORRT_MAJOR >= 4
        Dims3 d;
        d = static_cast<Dims3&&>(_engine->getBindingDimensions(_binding_data));
        my_assert(d.nbDims == 3, "bad nbDims for 'data'");
        my_assert(d.d[0] == dataDims[0] && d.d[1] == dataDims[1] && d.d[2] == dataDims[2], "bad dims for 'data'");
        _blob_sizes[_binding_data] = d.d[0] * d.d[1] * d.d[2];
        d = static_cast<Dims3&&>(_engine->getBindingDimensions(_binding_prob));
        my_assert(d.nbDims == 3, "bad nbDims for 'prob'");
        my_assert(d.d[0] == probDims[0] && d.d[1] == probDims[1] && d.d[2] == probDims[2], "bad dims for 'prob'");
        _blob_sizes[_binding_prob] = d.d[0] * d.d[1] * d.d[2];
#endif  // NV_TENSORRT_MAJOR

        for (int i = 0; i < 2; i++) {
            CHECK(cudaMalloc(&_gpu_buffers[i], _blob_sizes[i] * sizeof(float)));
        }
    }

    void TrtGooglenet::forward(float *imgs, float *prob)
    {
        CHECK(cudaMemcpyAsync(_gpu_buffers[_binding_data],
                              imgs,
                              _blob_sizes[_binding_data] * sizeof(float),
                              cudaMemcpyHostToDevice,
                              _stream));
        _context->enqueue(1, _gpu_buffers, _stream, nullptr);
        CHECK(cudaMemcpyAsync(prob,
                              _gpu_buffers[_binding_prob],
                              _blob_sizes[_binding_prob] * sizeof(float),
                              cudaMemcpyDeviceToHost,
                              _stream));
        cudaStreamSynchronize(_stream);
    }

    void TrtGooglenet::destroy()
    {
        cudaStreamDestroy(_stream);
        _context->destroy();
        _engine->destroy();
        _runtime->destroy();
    }

    //
    // TrtMtcnnDet stuffs
    //

    TrtMtcnnDet::TrtMtcnnDet()
    {
        for (int i = 0; i < 3; i++) {
            _gpu_buffers[i] = nullptr;
        }
    }

    void TrtMtcnnDet::initEngine(std::string filePath)
    {
        _gieModelStream = new IHostMemoryFromFile(filePath);
        _runtime = createInferRuntime(_gLogger);
        assert(_runtime != nullptr);
        _engine = _runtime->deserializeCudaEngine(
            _gieModelStream->data(),
            _gieModelStream->size(),
            nullptr);
        assert(_engine != nullptr);
        assert(_engine->getNbBindings() == 3);
        assert(_engine->bindingIsInput(0) == true);   // data
        assert(_engine->bindingIsInput(1) == false);  // prob1
        assert(_engine->bindingIsInput(2) == false);  // boxes
        _context = _engine->createExecutionContext();
        assert(_context != nullptr);
        _gieModelStream->destroy();
        CHECK(cudaStreamCreate(&_stream));
    }

    void TrtMtcnnDet::initEngine(std::string filePath, int dataDims[3], int prob1Dims[3], int boxesDims[3])
    {
        initEngine(filePath);
#if NV_TENSORRT_MAJOR == 3
        DimsCHW d;
        d = static_cast<DimsCHW&&>(_engine->getBindingDimensions(0));
        my_assert(d.nbDims == 3, "bad nbDims for 'data'");
        my_assert(d.c() == dataDims[0] && d.h() == dataDims[1] && d.w() == dataDims[2], "bad dims for 'data'");
        _blob_sizes[0] = d.c() * d.h() * d.w();
        d = static_cast<DimsCHW&&>(_engine->getBindingDimensions(1));
        my_assert(d.nbDims == 3, "bad nbDims for 'prob1'");
        my_assert(d.c() == prob1Dims[0] && d.h() == prob1Dims[1] && d.w() == prob1Dims[2], "bad dims for 'prob1'");
        _blob_sizes[1] = d.c() * d.h() * d.w();
        d = static_cast<DimsCHW&&>(_engine->getBindingDimensions(2));
        my_assert(d.nbDims == 3, "bad nbDims for 'boxes'");
        my_assert(d.c() == boxesDims[0] && d.h() == boxesDims[1] && d.w() == boxesDims[2], "bad dims for 'boxes'");
        _blob_sizes[2] = d.c() * d.h() * d.w();
#else   // NV_TENSORRT_MAJOR >= 4
        Dims3 d;
        d = static_cast<Dims3&&>(_engine->getBindingDimensions(0));
        my_assert(d.nbDims == 3, "bad nbDims for 'data'");
        my_assert(d.d[0] == dataDims[0] && d.d[1] == dataDims[1] && d.d[2] == dataDims[2], "bad dims for 'data'");
        _blob_sizes[0] = d.d[0] * d.d[1] * d.d[2];
        d = static_cast<Dims3&&>(_engine->getBindingDimensions(1));
        my_assert(d.nbDims == 3, "bad nbDims for 'prob1'");
        my_assert(d.d[0] == prob1Dims[0] && d.d[1] == prob1Dims[1] && d.d[2] == prob1Dims[2], "bad dims for 'prob1'");
        _blob_sizes[1] = d.d[0] * d.d[1] * d.d[2];
        d = static_cast<Dims3&&>(_engine->getBindingDimensions(2));
        my_assert(d.nbDims == 3, "bad nbDims for 'boxes'");
        my_assert(d.d[0] == boxesDims[0] && d.d[1] == boxesDims[1] && d.d[2] == boxesDims[2], "bad dims for 'boxes'");
        _blob_sizes[2] = d.d[0] * d.d[1] * d.d[2];
#endif  // NV_TENSORRT_MAJOR
    }

    void TrtMtcnnDet::setBatchSize(int value)
    {
        my_assert(value > 0 && value <= 256, "bad batch_size");
        if (value == _batchsize || _engine == nullptr)
            return;  // do nothing
        _batchsize = value;
        for (int i = 0; i < 3; i++) {
            if (_gpu_buffers[i] != nullptr) {
                CHECK(cudaFree(_gpu_buffers[i]));
                _gpu_buffers[i] = nullptr;
            }
        }
        for (int i = 0; i < 3; i++) {
            CHECK(cudaMalloc(&_gpu_buffers[i],
                             _batchsize * _blob_sizes[i] * sizeof(float)));
        }
    }

    int TrtMtcnnDet::getBatchSize()
    {
        return _batchsize;
    }

    void TrtMtcnnDet::forward(float *imgs, float *probs, float *boxes)
    {
        my_assert(_batchsize > 0, "_batchsize is not set");
        CHECK(cudaMemcpyAsync(_gpu_buffers[0],
                              imgs,
                              _batchsize * _blob_sizes[0] * sizeof(float),
                              cudaMemcpyHostToDevice,
                              _stream));
        _context->enqueue(_batchsize, _gpu_buffers, _stream, nullptr);
        CHECK(cudaMemcpyAsync(probs,
                              _gpu_buffers[1],
                              _batchsize * _blob_sizes[1] * sizeof(float),
                              cudaMemcpyDeviceToHost,
                              _stream));
        CHECK(cudaMemcpyAsync(boxes,
                              _gpu_buffers[2],
                              _batchsize * _blob_sizes[2] * sizeof(float),
                              cudaMemcpyDeviceToHost,
                              _stream));
        cudaStreamSynchronize(_stream);
    }

    void TrtMtcnnDet::destroy()
    {
        cudaStreamDestroy(_stream);
        _context->destroy();
        _engine->destroy();
        _runtime->destroy();
    }

}  // namespace trtnet
