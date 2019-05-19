from libcpp.string cimport string

cdef extern from 'trtNet.cpp' namespace 'mtcnn_trtnet':
    pass

cdef extern from 'trtNet.h' namespace 'mtcnn_trtnet':
    cdef cppclass TrtMtcnnDet:
        TrtMtcnnDet() except +
        void initEngine(string, int *, int *, int *)
        void setBatchSize(int)
        int  getBatchSize()
        void forward(float *, float *, float *)
        void destroy()
