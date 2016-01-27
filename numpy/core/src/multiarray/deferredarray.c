

static int
PyDeferredArray_init(PyDeferredArrayObject *self, PyObject *args, PyObject *kwds)
{
    if (PyArray_Type.tp_init((PyObject *)self, args, kwds) < 0)
        return -1;
    self->materialized = 0;
    return 0;
}

NPY_NO_EXPORT PyTypeObject PyDeferredArray_Type = {
#if defined(NPY_PY3K)
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
#endif
    "numpy.deferredarray",         /* tp_name */
    sizeof(PyDeferredArrayObject),          /* tp_basicsize */
    0,                       /* tp_itemsize */
    0,                       /* tp_dealloc */
    0,                       /* tp_print */
    0,                       /* tp_getattr */
    0,                       /* tp_setattr */
    0,                       /* tp_compare */
    0,                       /* tp_repr */
    0,                       /* tp_as_number */
    0,                       /* tp_as_sequence */
    0,                       /* tp_as_mapping */
    0,                       /* tp_hash */
    0,                       /* tp_call */
    0,                       /* tp_str */
    0,                       /* tp_getattro */
    0,                       /* tp_setattro */
    0,                       /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT |
      Py_TPFLAGS_BASETYPE,   /* tp_flags */
    0,                       /* tp_doc */
    0,                       /* tp_traverse */
    0,                       /* tp_clear */
    0,                       /* tp_richcompare */
    0,                       /* tp_weaklistoffset */
    0,                       /* tp_iter */
    0,                       /* tp_iternext */
    0,          /* tp_methods */
    0,                       /* tp_members */
    0,                       /* tp_getset */
    0,                       /* tp_base */
    0,                       /* tp_dict */
    0,                       /* tp_descr_get */
    0,                       /* tp_descr_set */
    0,                       /* tp_dictoffset */
    (initproc)PyDeferredArray_init,   /* tp_init */
    0,                       /* tp_alloc */
    0,                       /* tp_new */
};

npy_bool PyDeferredArray_Materialize(PyObject* op) {
    if (!PyDeferredArray_Check(op)) {
        return 1;
    }
    if (((PyDeferredArrayObject*)op)->materialized) {
        return 1;
    }
    if (PyCompressedArray_CheckExact(op)) {
        PyCompressedArray_Materialize((PyCompressedArrayObject*)op);
    }
    return 0;
}


static int
PyCompressedArray_init(PyCompressedArrayObject *self, PyObject *args, PyObject *kwds)
{
    if (PyDeferredArray_Type.tp_init((PyObject *)self, args, kwds) < 0)
        return -1;
    return 0;
}

static PyObject *
compressedarray_str(PyCompressedArrayObject *self)
{
    return PyArray_Type.tp_str((PyObject*)self);
}

NPY_NO_EXPORT PyTypeObject PyCompressedArray_Type = {
#if defined(NPY_PY3K)
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
#endif
    "numpy.compressedarray",         /* tp_name */
    sizeof(PyCompressedArrayObject),          /* tp_basicsize */
    0,                       /* tp_itemsize */
    0,                       /* tp_dealloc */
    0,                       /* tp_print */
    0,                       /* tp_getattr */
    0,                       /* tp_setattr */
    0,                       /* tp_compare */
    0,                       /* tp_repr */
    0,                       /* tp_as_number */
    0,                       /* tp_as_sequence */
    0,                       /* tp_as_mapping */
    0,                       /* tp_hash */
    0,                       /* tp_call */
    (reprfunc)compressedarray_str,                       /* tp_str */
    0,                       /* tp_getattro */
    0,                       /* tp_setattro */
    0,                       /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT |
      Py_TPFLAGS_BASETYPE,   /* tp_flags */
    0,                       /* tp_doc */
    0,                       /* tp_traverse */
    0,                       /* tp_clear */
    0,                       /* tp_richcompare */
    0,                       /* tp_weaklistoffset */
    0,                       /* tp_iter */
    0,                       /* tp_iternext */
    0,          /* tp_methods */
    0,                       /* tp_members */
    0,                       /* tp_getset */
    0,                       /* tp_base */
    0,                       /* tp_dict */
    0,                       /* tp_descr_get */
    0,                       /* tp_descr_set */
    0,                       /* tp_dictoffset */
    (initproc)PyCompressedArray_init,   /* tp_init */
    0,                       /* tp_alloc */
    0,                       /* tp_new */
};


npy_bool PyCompressedArray_Materialize(PyCompressedArrayObject* op) {
    switch(op->compression_type) {
        case NPY_COMPRESSION_RUNLENGTHENCODING: {
            RunLengthEncoding *rle = (RunLengthEncoding*)op->compression_storage;
            switch(rle->type_num) {
                case NPY_INT32: {
                    int *ptr = malloc(rle->count * sizeof(int));
                    RunLengthType_int *current = (RunLengthType_int*)rle->elements;
                    size_t index = 0;
                    while(current != NULL) {
                        for(size_t j = 0; j < current->count; j++) {
                            ptr[index++] = current->value;
                        }
                        current = (RunLengthType_int*)current->next;
                    }
                    ((PyArrayObject_fields*)op)->data = (char*)ptr;
                    break;
                }
                default:
                    PyErr_SetString(PyExc_TypeError, "Unsupported NumPy array type.");
                    return 0;
            }
            ((PyDeferredArrayObject*)op)->materialized = 1;
            return 1;
        }
        default:
            PyErr_SetString(PyExc_TypeError, "Unsupported compression type.");
            return 0;
    }
    return 0;
}


PyObject *PyCompressedArray_CompressArray(PyObject* op, numpy_comptype type) {
    if (!PyArray_Check(op)) {
        PyErr_SetString(PyExc_TypeError, "Can only compress NumPy arrays for now.");
        return NULL;
    }
    PyCompressedArrayObject *res = malloc(sizeof(PyCompressedArrayObject));
    if (res == NULL) {
        return NULL;
    }
    PyObject_Init((PyObject *)res, &PyCompressedArray_Type);

    ((PyDeferredArrayObject*)res)->materialized = 0;
    ((PyArrayObject_fields*)res)->data = NULL;
    ((PyArrayObject_fields*)res)->nd = ((PyArrayObject_fields*)op)->nd;
    ((PyArrayObject_fields*)res)->dimensions = ((PyArrayObject_fields*)op)->dimensions;
    ((PyArrayObject_fields*)res)->strides = ((PyArrayObject_fields*)op)->strides;
    ((PyArrayObject_fields*)res)->base = ((PyArrayObject_fields*)op)->base;
    ((PyArrayObject_fields*)res)->descr = ((PyArrayObject_fields*)op)->descr;
    ((PyArrayObject_fields*)res)->flags = ((PyArrayObject_fields*)op)->flags;
    ((PyArrayObject_fields*)res)->weakreflist = ((PyArrayObject_fields*)op)->weakreflist;
    switch(type) {
        case NPY_COMPRESSION_RUNLENGTHENCODING:
            res->compression_type = NPY_COMPRESSION_RUNLENGTHENCODING;
            RunLengthEncoding *rle = (RunLengthEncoding*) malloc(sizeof(RunLengthEncoding));
            rle->type_num = ((PyArrayObject_fields*)op)->descr->type_num;
            rle->count = ((PyArrayObject_fields*)op)->dimensions[0];
            res->compression_storage = (void*) rle;
            switch(rle->type_num) {
                case NPY_INT32: {
                        int *ptr = (int*)PyArray_DATA((PyArrayObject*)op);
                        RunLengthType_int *current = (RunLengthType_int*)malloc(sizeof(RunLengthType_int));
                        current->value = ptr[0];
                        current->count = 1;
                        current->next = NULL;
                        rle->elements = (void*) current;
                        for(size_t j = 1; j < rle->count; j++) {
                            if (ptr[j] != current->value) {
                                // if there is a different element, we need to change to the next container
                                current->next = malloc(sizeof(RunLengthType_int));
                                current = (RunLengthType_int*)current->next;
                                current->value = ptr[j];
                                current->count = 1;
                                current->next = NULL;
                            } else {
                                current->count++;
                            }
                        }
                        break;
                    }
                default:
                    PyErr_SetString(PyExc_TypeError, "Unsupported NumPy array type.");
                    return NULL;
            }
            break;
        default:
            PyErr_SetString(PyExc_TypeError, "Unsupported compression type.");
            return NULL;
    }
    return (PyObject*)res;
}

PyObject *_compressarray_python(PyObject *NPY_UNUSED(ignored), PyObject *args) {
    return PyCompressedArray_CompressArray(args, NPY_COMPRESSION_RUNLENGTHENCODING);
}
