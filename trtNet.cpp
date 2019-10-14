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

    void TrtGooglenet::_initEngine(std::string filePath)
    {
        _gieModelStream = new IHostMemoryFromFile(filePath);
        _runtime = createInferRuntime(_gLogger);
        my_assert(_runtime != nullptr, "_runtime is null");
        _engine = _runtime->deserializeCudaEngine(
            _gieModelStream->data(),
            _gieModelStream->size(),
            nullptr);
        my_assert(_engine != nullptr, "_engine is null");
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
        _initEngine(filePath);
#if NV_TENSORRT_MAJOR == 3
        DimsCHW d;
        d = static_cast<DimsCHW&&>(_engine->getBindingDimensions(_binding_data));
        my_assert(d.nbDims == 3, "bad nbDims for 'data'");
        my_assert(d.c() == dataDims[0] && d.h() == dataDims[1] && d.w() == dataDims[2], "bad dims for 'data'");
        _blob_sizes[_binding_data] = d.c() * d.h() * d.w();

        d = static_cast<DimsCHW&&>(_engine->getBindingDimensions(_binding_prob));
        my_assert(d.nbDims == 3, "bad nbDims for 'prob'");
        my_assert(d.c() == probDims[0] && d.h() == probDims[1] && d.w() == probDims[2], "bad dims for 'prob'");
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
        for (int i = 0; i < 2; i++) {
            if (_gpu_buffers[i] != nullptr) {
                CHECK(cudaFree(_gpu_buffers[i]));
                _gpu_buffers[i] = nullptr;
            }
        }
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
        for (int i = 0; i < 4; i++) {
            _gpu_buffers[i] = nullptr;
        }
    }

    void TrtMtcnnDet::_initEngine(std::string filePath, const char *dataName, const char *prob1Name, const char *boxesName, const char *marksName="unspecified")
    {
        _gieModelStream = new IHostMemoryFromFile(filePath);
        _runtime = createInferRuntime(_gLogger);
        my_assert(_runtime != nullptr, "_runtime is null");
        _engine = _runtime->deserializeCudaEngine(
            _gieModelStream->data(),
            _gieModelStream->size(),
            nullptr);
        my_assert(_engine != nullptr, "_engine is null");
        my_assert(_engine->getNbBindings() == _num_bindings, "wrong number of bindings");
	    _binding_data = _engine->getBindingIndex(dataName);
        my_assert(_engine->bindingIsInput(_binding_data) == true, "bad type of binding 'data'");
	    _binding_prob1 = _engine->getBindingIndex(prob1Name);
        my_assert(_engine->bindingIsInput(_binding_prob1) == false, "bad type of binding 'prob1'");
	    _binding_boxes = _engine->getBindingIndex(boxesName);
        my_assert(_engine->bindingIsInput(_binding_boxes) == false, "bad type of binding 'boxes'");
        if (_num_bindings == 4) {
	        _binding_marks = _engine->getBindingIndex(marksName);
            my_assert(_engine->bindingIsInput(_binding_marks) == false, "bad type of binding 'marks'");
        }
        _context = _engine->createExecutionContext();
        my_assert(_context != nullptr, "_context is null");
        _gieModelStream->destroy();
        CHECK(cudaStreamCreate(&_stream));
    }

    void TrtMtcnnDet::_setBlobSizes(int dataDims[3], int prob1Dims[3], int boxesDims[3])
    {
#if NV_TENSORRT_MAJOR == 3
        DimsCHW d;
        d = static_cast<DimsCHW&&>(_engine->getBindingDimensions(_binding_data));
        my_assert(d.nbDims == 3, "bad nbDims for 'data'");
        my_assert(d.c() == dataDims[0] && d.h() == dataDims[1] && d.w() == dataDims[2], "bad dims for 'data'");
        _blob_sizes[_binding_data] = d.c() * d.h() * d.w();

        d = static_cast<DimsCHW&&>(_engine->getBindingDimensions(_binding_prob1));
        my_assert(d.nbDims == 3, "bad nbDims for 'prob1'");
        my_assert(d.c() == prob1Dims[0] && d.h() == prob1Dims[1] && d.w() == prob1Dims[2], "bad dims for 'prob1'");
        _blob_sizes[_binding_prob1] = d.c() * d.h() * d.w();

        d = static_cast<DimsCHW&&>(_engine->getBindingDimensions(_binding_boxes));
        my_assert(d.nbDims == 3, "bad nbDims for 'boxes'");
        my_assert(d.c() == boxesDims[0] && d.h() == boxesDims[1] && d.w() == boxesDims[2], "bad dims for 'boxes'");
        _blob_sizes[_binding_boxes] = d.c() * d.h() * d.w();
#else   // NV_TENSORRT_MAJOR >= 4
        Dims3 d;
        d = static_cast<Dims3&&>(_engine->getBindingDimensions(_binding_data));
        my_assert(d.nbDims == 3, "bad nbDims for 'data'");
        my_assert(d.d[0] == dataDims[0] && d.d[1] == dataDims[1] && d.d[2] == dataDims[2], "bad dims for 'data'");
        _blob_sizes[_binding_data] = d.d[0] * d.d[1] * d.d[2];

        d = static_cast<Dims3&&>(_engine->getBindingDimensions(_binding_prob1));
        my_assert(d.nbDims == 3, "bad nbDims for 'prob1'");
        my_assert(d.d[0] == prob1Dims[0] && d.d[1] == prob1Dims[1] && d.d[2] == prob1Dims[2], "bad dims for 'prob1'");
        _blob_sizes[_binding_prob1] = d.d[0] * d.d[1] * d.d[2];

        d = static_cast<Dims3&&>(_engine->getBindingDimensions(_binding_boxes));
        my_assert(d.nbDims == 3, "bad nbDims for 'boxes'");
        my_assert(d.d[0] == boxesDims[0] && d.d[1] == boxesDims[1] && d.d[2] == boxesDims[2], "bad dims for 'boxes'");
        _blob_sizes[_binding_boxes] = d.d[0] * d.d[1] * d.d[2];
#endif  // NV_TENSORRT_MAJOR
    }

    void TrtMtcnnDet::initDet1(std::string filePath, int dataDims[3], int prob1Dims[3], int boxesDims[3])
    {
        _num_bindings = 3;
        _initEngine(filePath, "data", "prob1", "conv4-2");
        _setBlobSizes(dataDims, prob1Dims, boxesDims);
    }

    void TrtMtcnnDet::initDet2(std::string filePath, int dataDims[3], int prob1Dims[3], int boxesDims[3])
    {
        _num_bindings = 3;
        _initEngine(filePath, "data", "prob1", "conv5-2");
        _setBlobSizes(dataDims, prob1Dims, boxesDims);
    }

    void TrtMtcnnDet::initDet3(std::string filePath, int dataDims[3], int prob1Dims[3], int boxesDims[3], int marksDims[3])
    {
        _num_bindings = 4;
        _initEngine(filePath, "data", "prob1", "conv6-2", "conv6-3");
        _setBlobSizes(dataDims, prob1Dims, boxesDims);

#if NV_TENSORRT_MAJOR == 3
        DimsCHW d;
        d = static_cast<DimsCHW&&>(_engine->getBindingDimensions(_binding_marks));
        my_assert(d.nbDims == 3, "bad nbDims for 'marks'");
        my_assert(d.c() == marksDims[0] && d.h() == marksDims[1] && d.w() == marksDims[2], "bad dims for 'marks'");
        _blob_sizes[_binding_marks] = d.c() * d.h() * d.w();
#else   // NV_TENSORRT_MAJOR >= 4
        Dims3 d;
        d = static_cast<Dims3&&>(_engine->getBindingDimensions(_binding_marks));
        my_assert(d.nbDims == 3, "bad nbDims for 'marks'");
        my_assert(d.d[0] == marksDims[0] && d.d[1] == marksDims[1] && d.d[2] == marksDims[2], "bad dims for 'marks'");
        _blob_sizes[_binding_marks] = d.d[0] * d.d[1] * d.d[2];
#endif  // NV_TENSORRT_MAJOR
    }

    void TrtMtcnnDet::setBatchSize(int value)
    {
        my_assert(value > 0 && value <= 1024, "bad batch_size");
        if (value == _batchsize || _engine == nullptr)
            return;  // do nothing
        _batchsize = value;
        for (int i = 0; i < _num_bindings; i++) {
            if (_gpu_buffers[i] != nullptr) {
                CHECK(cudaFree(_gpu_buffers[i]));
                _gpu_buffers[i] = nullptr;
            }
        }
        for (int i = 0; i < _num_bindings; i++) {
            CHECK(cudaMalloc(&_gpu_buffers[i],
                             _batchsize * _blob_sizes[i] * sizeof(float)));
        }
    }

    int TrtMtcnnDet::getBatchSize()
    {
        return _batchsize;
    }

    void TrtMtcnnDet::forward(float *imgs, float *probs, float *boxes, float *marks=nullptr)
    {
        my_assert(_batchsize > 0, "_batchsize is not set");
        CHECK(cudaMemcpyAsync(_gpu_buffers[_binding_data],
                              imgs,
                              _batchsize * _blob_sizes[_binding_data] * sizeof(float),
                              cudaMemcpyHostToDevice,
                              _stream));
        _context->enqueue(_batchsize, _gpu_buffers, _stream, nullptr);
        CHECK(cudaMemcpyAsync(probs,
                              _gpu_buffers[_binding_prob1],
                              _batchsize * _blob_sizes[_binding_prob1] * sizeof(float),
                              cudaMemcpyDeviceToHost,
                              _stream));
        CHECK(cudaMemcpyAsync(boxes,
                              _gpu_buffers[_binding_boxes],
                              _batchsize * _blob_sizes[_binding_boxes] * sizeof(float),
                              cudaMemcpyDeviceToHost,
                              _stream));
        if (_num_bindings == 4) {
            my_assert(marks != nullptr, "pointer 'marks' is null");
            CHECK(cudaMemcpyAsync(marks,
                                  _gpu_buffers[_binding_marks],
                                  _batchsize * _blob_sizes[_binding_marks] * sizeof(float),
                                  cudaMemcpyDeviceToHost,
                                  _stream));
        }
        cudaStreamSynchronize(_stream);
    }

    void TrtMtcnnDet::destroy()
    {
        for (int i = 0; i < _num_bindings; i++) {
            if (_gpu_buffers[i] != nullptr) {
                CHECK(cudaFree(_gpu_buffers[i]));
                _gpu_buffers[i] = nullptr;
            }
        }
        cudaStreamDestroy(_stream);
        _context->destroy();
        _engine->destroy();
        _runtime->destroy();
    }

}  // namespace trtnet
