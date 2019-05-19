from libcpp.string cimport string

cdef extern from 'trtNet.cpp' namespace 'trtnet':
    pass

cdef extern from 'trtNet.h' namespace 'trtnet':
    cdef cppclass TrtGooglenet:
        TrtGooglenet() except +
        void initEngine(string, int *, int *)
        void forward(float *, float *)
        void destroy()

    cdef cppclass TrtMtcnnDet:
        TrtMtcnnDet() except +
        void initEngine(string, int *, int *, int *)
        void setBatchSize(int)
        int  getBatchSize()
        void forward(float *, float *, float *)
        void destroy()
