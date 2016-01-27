
typedef struct {   
    // Underlying numpy array
    PyArrayObject_fields underlying_array;
    // flag that indicates whether or not the array is materialized
    npy_bool materialized;
} PyDeferredArrayObject;

#define PyDeferredArray_Check(op) PyObject_TypeCheck(op, &PyDeferredArray_Type)
#define PyDeferredArray_CheckExact(op) (((PyObject*)(op))->ob_type == &PyDeferredArray_Type)

npy_bool PyDeferredArray_Materialize(PyObject*);