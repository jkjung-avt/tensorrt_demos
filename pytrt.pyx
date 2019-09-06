import cython

import numpy as np
cimport numpy as np
from libcpp.string cimport string

from pytrt cimport TrtGooglenet
from pytrt cimport TrtMtcnnDet

cdef class PyTrtGooglenet:
    cdef TrtGooglenet *c_trtnet
    cdef tuple data_dims, prob_dims

    def __cinit__(PyTrtGooglenet self):
        self.c_trtnet = NULL

    def __init__(PyTrtGooglenet self,
                 str engine_path, tuple shape0, tuple shape1):
        assert len(shape0) == 3 and len(shape1) == 3
        self.c_trtnet = new TrtGooglenet()
        self.data_dims = shape0
        self.prob_dims = shape1
        cdef int[:] v0 = np.array(shape0, dtype=np.intc)
        cdef int[:] v1 = np.array(shape1, dtype=np.intc)
        cdef string c_str = engine_path.encode('UTF-8')
        self.c_trtnet.initEngine(c_str, &v0[0], &v1[0])

    def forward(PyTrtGooglenet self,
                np.ndarray[np.float32_t, ndim=4] np_imgs not None):
        """Do a forward() computation on the input batch of imgs."""
        assert np_imgs.shape[0] == 1  # only accept batch_size = 1
        if not np_imgs.flags['C_CONTIGUOUS']:
            np_imgs = np.ascontiguousarray(np_imgs)
        np_prob = np.ascontiguousarray(
            np.zeros((1,) + self.prob_dims, dtype=np.float32)
        )
        cdef float[:,:,:,::1] v_imgs = np_imgs
        cdef float[:,:,:,::1] v_prob = np_prob
        self.c_trtnet.forward(&v_imgs[0][0][0][0], &v_prob[0][0][0][0])
        return { 'prob': np_prob }

    def destroy(PyTrtGooglenet self):
        self.c_trtnet.destroy()


cdef class PyTrtMtcnn:
    cdef TrtMtcnnDet *c_trtnet
    cdef int batch_size
    cdef int num_bindings
    cdef tuple data_dims, prob1_dims, boxes_dims, marks_dims

    def __cinit__(PyTrtMtcnn self):
        self.c_trtnet = NULL

    def __init__(PyTrtMtcnn self,
                 str engine_path,
                 tuple shape0, tuple shape1, tuple shape2, tuple shape3=None):
        self.num_bindings = 4 if shape3 else 3
        assert len(shape0) == 3 and len(shape1) == 3 and len(shape2) == 3
        if shape3: assert len(shape3) == 3
        else: shape3 = (0, 0, 0)  # set to a dummy shape
        self.c_trtnet = new TrtMtcnnDet()
        self.batch_size = 0
        self.data_dims  = shape0
        self.prob1_dims = shape1
        self.boxes_dims = shape2
        self.marks_dims = shape3
        cdef int[:] v0 = np.array(shape0, dtype=np.intc)
        cdef int[:] v1 = np.array(shape1, dtype=np.intc)
        cdef int[:] v2 = np.array(shape2, dtype=np.intc)
        cdef int[:] v3 = np.array(shape3, dtype=np.intc)
        cdef string c_str = engine_path.encode('UTF-8')
        if 'det1' in engine_path:
            self.c_trtnet.initDet1(c_str, &v0[0], &v1[0], &v2[0])
        elif 'det2' in engine_path:
            self.c_trtnet.initDet2(c_str, &v0[0], &v1[0], &v2[0])
        elif 'det3' in engine_path:
            self.c_trtnet.initDet3(c_str, &v0[0], &v1[0], &v2[0], &v3[0])
        else:
            raise ValueError('engine is neither of det1, det2 or det3!')

    def set_batchsize(PyTrtMtcnn self, int batch_size):
        self.c_trtnet.setBatchSize(batch_size)
        self.batch_size = batch_size

    def _forward_3(PyTrtMtcnn self,
                   np.ndarray[np.float32_t, ndim=4] np_imgs  not None,
                   np.ndarray[np.float32_t, ndim=4] np_prob1 not None,
                   np.ndarray[np.float32_t, ndim=4] np_boxes not None):
        cdef float[:,:,:,::1] v_imgs  = np_imgs
        cdef float[:,:,:,::1] v_probs = np_prob1
        cdef float[:,:,:,::1] v_boxes = np_boxes
        self.c_trtnet.forward(&v_imgs[0][0][0][0],
                              &v_probs[0][0][0][0],
                              &v_boxes[0][0][0][0])
        return { 'prob1': np_prob1, 'boxes': np_boxes }

    def _forward_4(PyTrtMtcnn self,
                   np.ndarray[np.float32_t, ndim=4] np_imgs  not None,
                   np.ndarray[np.float32_t, ndim=4] np_prob1 not None,
                   np.ndarray[np.float32_t, ndim=4] np_boxes not None,
                   np.ndarray[np.float32_t, ndim=4] np_marks not None):
        cdef float[:,:,:,::1] v_imgs  = np_imgs
        cdef float[:,:,:,::1] v_probs = np_prob1
        cdef float[:,:,:,::1] v_boxes = np_boxes
        cdef float[:,:,:,::1] v_marks = np_marks
        self.c_trtnet.forward(&v_imgs[0][0][0][0],
                              &v_probs[0][0][0][0],
                              &v_boxes[0][0][0][0],
                              &v_marks[0][0][0][0])
        return { 'prob1': np_prob1, 'boxes': np_boxes, 'landmarks': np_marks }

    def forward(PyTrtMtcnn self,
                np.ndarray[np.float32_t, ndim=4] np_imgs not None):
        """Do a forward() computation on the input batch of imgs."""
        assert(np_imgs.shape[0] == self.batch_size)
        if not np_imgs.flags['C_CONTIGUOUS']:
            np_imgs = np.ascontiguousarray(np_imgs)
        np_prob1 = np.ascontiguousarray(
            np.zeros((self.batch_size,) + self.prob1_dims, dtype=np.float32)
        )
        np_boxes = np.ascontiguousarray(
            np.zeros((self.batch_size,) + self.boxes_dims, dtype=np.float32)
        )
        np_marks = np.ascontiguousarray(
            np.zeros((self.batch_size,) + self.marks_dims, dtype=np.float32)
        )
        if self.num_bindings == 3:
            return self._forward_3(np_imgs, np_prob1, np_boxes)
        else:  # self.num_bindings == 4
            return self._forward_4(np_imgs, np_prob1, np_boxes, np_marks)

    def destroy(PyTrtMtcnn self):
        self.c_trtnet.destroy()
