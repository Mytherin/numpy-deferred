typedef struct {   
    // Underlying numpy array
    PyArrayObject_fields underlying_array;
    // flag that indicates whether or not the array is materialized
    npy_bool materialized;
} PyDeferredArrayObject;

#define PyDeferredArray_Check(op) PyObject_TypeCheck(op, &PyDeferredArray_Type)
#define PyDeferredArray_CheckExact(op) (((PyObject*)(op))->ob_type == &PyDeferredArray_Type)

npy_bool PyDeferredArray_Materialize(PyObject*);


#define NPY_COMPRESSION_RUNLENGTHENCODING 1
#define NPY_COMPRESSION_DELTAENCODING 2
#define NPY_COMPRESSION_ANY 255

typedef int numpy_comptype;

#define GENERATE_RUNLENGTHTYPE(tpe)             \
typedef struct {                                \
    size_t count;                               \
    tpe value;                                  \
    void *next;                  \
} RunLengthType_##tpe;                          \

GENERATE_RUNLENGTHTYPE(int);


typedef struct {
    size_t count;
    int type_num;
    void *elements;
} RunLengthEncoding;


typedef struct {
    // Underlying deferred array
    PyDeferredArrayObject underlying_array;
    // Compression type
    numpy_comptype compression_type;
    // Compression storage
    void *compression_storage;
} PyCompressedArrayObject;


#define PyCompressedArray_Check(op) PyObject_TypeCheck(op, &PyCompressedArray_Type)
#define PyCompressedArray_CheckExact(op) (((PyObject*)(op))->ob_type == &PyCompressedArray_Type)

npy_bool PyCompressedArray_Materialize(PyCompressedArrayObject* op);

PyObject *PyCompressedArray_CompressArray(PyObject*, numpy_comptype);



PyObject *_compressarray_python(PyObject*, PyObject *);
