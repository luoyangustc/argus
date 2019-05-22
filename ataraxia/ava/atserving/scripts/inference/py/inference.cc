
#include <Python.h>
#include <iostream>

#include "inference.h"

char *pyString(PyObject *pyErr)
{
    char *err = PyString_AS_STRING(pyErr);
    char *ret = new char[strlen(err)];
    strcpy(ret, err);
    return ret;
}

char *pyError()
{
    PyObject *pyType, *pyValue, *pyTrace;
    PyErr_Fetch(&pyType, &pyValue, &pyTrace);
    PyErr_Clear();
    if (pyType != NULL)
        Py_DECREF(pyType);
    if (pyTrace != NULL)
        Py_DECREF(pyTrace);

    PyObject *pyResult = PyObject_Str(pyValue);
    PyErr_Clear();
    Py_DECREF(pyValue);
    if (pyResult != NULL)
    {
        char *result = pyString(pyResult);
        Py_DECREF(pyResult);
        return result;
    }
    return NULL;
}

void initEnv(
    void *params, const int params_size,
    int *code, char **err)
{
    try
    {
        Py_InitializeEx(0);
        PyEval_InitThreads();
        PySys_SetArgvEx(0, NULL, 0);

        return;
    }
    catch (std::exception *e)
    {
        std::cerr << e << std::endl;
        return;
    }
}

void *createNet(
    void *params, const int params_size,
    int *code, char **err)
{
    PyObject *pyModuler, *pyFunc;
    PyObject *pyNet, *pyCode, *pyErr;
    PyObject *pyParams = PyTuple_New(1);

    try
    {
        pyModuler = PyImport_Import(PyString_FromString("inference_py"));
        if (pyModuler == NULL)
        {
            *err = pyError();
            std::cerr << "createNet " << *err << std::endl;
            *code = 599;
            return NULL;
        }
        pyFunc = PyObject_GetAttrString(pyModuler, "create_net");
        if (pyFunc == NULL)
        {
            *err = pyError();
            Py_DECREF(pyModuler);
            std::cerr << "createNet " << *err << std::endl;
            *code = 599;
            return NULL;
        }

        PyObject *pyBuf = PyBuffer_FromMemory(params, params_size);
        PyTuple_SetItem(pyParams, 0, pyBuf);
        PyObject *pyRet = PyObject_CallObject(pyFunc, pyParams);
        if (pyRet == NULL)
        {
            *err = pyError();
            Py_DECREF(pyModuler);
            Py_DECREF(pyFunc);
            Py_DECREF(pyParams);
            std::cerr << "createNet " << *err << std::endl;
            *code = 599;
            return NULL;
        }

        pyNet = PyTuple_GET_ITEM(pyRet, 0);
        pyCode = PyTuple_GET_ITEM(pyRet, 1);
        pyErr = PyTuple_GET_ITEM(pyRet, 2);

        *code = PyInt_AS_LONG(pyCode);
        *err = PyString_AS_STRING(pyErr);

        Py_INCREF(pyNet);
        Py_DECREF(pyRet);
    }
    catch (std::exception *e)
    {
        std::cerr << e << std::endl;
        if (pyModuler != NULL)
        {
            Py_DECREF(pyModuler);
        }
        if (pyFunc != NULL)
        {
            Py_DECREF(pyFunc);
        }
        Py_DECREF(pyParams);
        return NULL;
    }

    Py_DECREF(pyModuler);
    Py_DECREF(pyFunc);
    Py_DECREF(pyParams);

    return pyNet;
}

void netPreprocess(
    const void *net, void *args, const int args_size,
    int *code, char **err, void *ret, int *ret_size)
{

    // PyGILState_STATE gilState = PyGILState_Ensure();
    PyObject *pyModuler, *pyFunc;
    PyObject *pyArgs = PyTuple_New(2);

    try
    {

        pyModuler = PyImport_Import(PyString_FromString("inference_py"));
        pyFunc = PyObject_GetAttrString(pyModuler, "net_preprocess");

        PyObject *pyBuf = PyBuffer_FromMemory(args, args_size);
        Py_INCREF((PyObject *)net);
        PyTuple_SetItem(pyArgs, 0, (PyObject *)net);
        PyTuple_SetItem(pyArgs, 1, pyBuf);
        PyObject *pyReturn = PyObject_CallObject(pyFunc, pyArgs);

        if (pyReturn == NULL)
        {
            *err = pyError();
            Py_DECREF(pyModuler);
            Py_DECREF(pyFunc);
            Py_DECREF(pyArgs);
            std::cerr << "preprocess " << *err << std::endl;
            *code = 599;
            return;
        }

        PyObject *pyRet, *pyCode, *pyErr;

        pyRet = PyTuple_GET_ITEM(pyReturn, 0);
        pyCode = PyTuple_GET_ITEM(pyReturn, 1);
        pyErr = PyTuple_GET_ITEM(pyReturn, 2);

        *code = PyInt_AS_LONG(pyCode);
        *err = pyString(pyErr);
        if (*code != 0 && *code != 200)
        {
            Py_DECREF(pyModuler);
            Py_DECREF(pyFunc);
            Py_DECREF(pyArgs);
            Py_DECREF(pyReturn);
            return;
        }

        char *_ret = NULL;
        long size = 0;
        PyString_AsStringAndSize(pyRet, &_ret, &size);
        memcpy(ret, _ret, size);

        // Py_INCREF(pyRet);
        Py_DECREF(pyReturn);
        *ret_size = (int)size;
    }
    catch (std::exception *e)
    {
        std::cerr << e << std::endl;
        // PyGILState_Release(gilState);
        if (pyModuler != NULL)
        {
            Py_DECREF(pyModuler);
        }
        if (pyFunc != NULL)
        {
            Py_DECREF(pyFunc);
        }
        Py_DECREF(pyArgs);
        return;
    }

    // PyGILState_Release(gilState);
    Py_DECREF(pyModuler);
    Py_DECREF(pyFunc);
    Py_DECREF(pyArgs);

    return;
}

void netInference(
    const void *net, void *args, const int args_size,
    int *code, char **err, void *ret, int *ret_size)
{

    PyObject *pyModuler, *pyFunc;
    PyObject *pyArgs = PyTuple_New(2);

    try
    {

        pyModuler = PyImport_Import(PyString_FromString("inference_py"));
        pyFunc = PyObject_GetAttrString(pyModuler, "net_inference");

        PyObject *pyBuf = PyBuffer_FromMemory(args, args_size);
        Py_INCREF((PyObject *)net);
        PyTuple_SetItem(pyArgs, 0, (PyObject *)net);
        PyTuple_SetItem(pyArgs, 1, pyBuf);
        PyObject *pyReturn = PyObject_CallObject(pyFunc, pyArgs);

        if (pyReturn == NULL)
        {
            *err = pyError();
            Py_DECREF(pyModuler);
            Py_DECREF(pyFunc);
            Py_DECREF(pyArgs);
            std::cerr << "inference " << *err << std::endl;
            *code = 599;
            return;
        }

        PyObject *pyRet, *pyCode, *pyErr;

        pyRet = PyTuple_GET_ITEM(pyReturn, 0);
        pyCode = PyTuple_GET_ITEM(pyReturn, 1);
        pyErr = PyTuple_GET_ITEM(pyReturn, 2);

        *code = PyInt_AS_LONG(pyCode);
        *err = pyString(pyErr);
        if (*code != 0 && *code != 200)
        {
            Py_DECREF(pyModuler);
            Py_DECREF(pyFunc);
            Py_DECREF(pyArgs);
            Py_DECREF(pyReturn);
            return;
        }

        char *_ret = NULL;
        long size = 0;
        PyString_AsStringAndSize(pyRet, &_ret, &size);
        memcpy(ret, _ret, size);

        // Py_INCREF(pyRet);
        Py_DECREF(pyReturn);
        *ret_size = (int)size;
    }
    catch (std::exception *e)
    {
        std::cerr << e << std::endl;
        if (pyModuler != NULL)
        {
            Py_DECREF(pyModuler);
        }
        if (pyFunc != NULL)
        {
            Py_DECREF(pyFunc);
        }
        Py_DECREF(pyArgs);

        return;
    }

    Py_DECREF(pyModuler);
    Py_DECREF(pyFunc);
    Py_DECREF(pyArgs);

    return;
}
