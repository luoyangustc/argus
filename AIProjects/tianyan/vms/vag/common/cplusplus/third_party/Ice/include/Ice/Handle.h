// **********************************************************************
//
// Copyright (c) 2003-2011 ZeroC, Inc. All rights reserved.
//
// This copy of Ice is licensed to you under the terms described in the
// ICE_LICENSE file included in this distribution.
//
// **********************************************************************

#ifndef ICE_HANDLE_H
#define ICE_HANDLE_H

#include <IceUtil/Handle.h>
#include <Ice/Config.h>

//
// We include ProxyHandle.h here to make sure that the Ice::ProxyHandle
// template is defined before any definition of upCast().
//
// See http://gcc.gnu.org/bugzilla/show_bug.cgi?id=25495 for information
// on why this is necessary.
//
#include <Ice/ProxyHandle.h>

//
// "Handle" or "smart pointer" template for classes derived from
// IceInternal::GCShared, IceUtil::Shared, or IceUtil::SimpleShared.
//
// In constrast to IceUtil::Handle, IceInternal::Handle<T> can be used
// for a type T that has been declared but not defined. The only
// requirement is a declaration of the following function:
//
// namespace IceInternal
// {
//     X* upCast(T*);
// }
//
// Where X is (or derives from) IceUtil::Shared or IceUtil::SimpleShared.
//

namespace IceInternal
{

template<typename T>
class Handle : public ::IceUtil::HandleBase<T>
{
public:
    
#if defined(__BCPLUSPLUS__) && (__BCPLUSPLUS__ >= 0x0600)
    //
    // C++Builder 2009 does not allow setting Ptr to 0.
    //
    Handle(int p)
    {
        assert(p == 0);
        this->_ptr = 0;
    }
#endif

    Handle(T* p = 0)
    {
        this->_ptr = p;

        if(this->_ptr)
        {
            upCast(this->_ptr)->__incRef();
        }
    }
    
    template<typename Y>
    Handle(const Handle<Y>& r)
    {
        this->_ptr = r._ptr;

        if(this->_ptr)
        {
            upCast(this->_ptr)->__incRef();
        }
    }

    template<typename Y>
    Handle(const ::IceUtil::Handle<Y>& r)
    {
        this->_ptr = r._ptr;

        if(this->_ptr)
        {
            upCast(this->_ptr)->__incRef();
        }
    }

    Handle(const Handle& r)
    {
        this->_ptr = r._ptr;

        if(this->_ptr)
        {
            upCast(this->_ptr)->__incRef();
        }
    }
    
    ~Handle()
    {
        if(this->_ptr)
        {
            upCast(this->_ptr)->__decRef();
        }
    }
    
    Handle& operator=(T* p)
    {
        if(this->_ptr != p)
        {
            if(p)
            {
                upCast(p)->__incRef();
            }

            T* ptr = this->_ptr;
            this->_ptr = p;

            if(ptr)
            {
                upCast(ptr)->__decRef();
            }
        }
        return *this;
    }
        
    template<typename Y>
    Handle& operator=(const Handle<Y>& r)
    {
        if(this->_ptr != r._ptr)
        {
            if(r._ptr)
            {
                upCast(r._ptr)->__incRef();
            }

            T* ptr = this->_ptr;
            this->_ptr = r._ptr;

            if(ptr)
            {
                upCast(ptr)->__decRef();
            }
        }
        return *this;
    }

    template<typename Y>
    Handle& operator=(const ::IceUtil::Handle<Y>& r)
    {
        if(this->_ptr != r._ptr)
        {
            if(r._ptr)
            {
                upCast(r._ptr)->__incRef();
            }

            T* ptr = this->_ptr;
            this->_ptr = r._ptr;

            if(ptr)
            {
                upCast(ptr)->__decRef();
            }
        }
        return *this;
    }

    Handle& operator=(const Handle& r)
    {
        if(this->_ptr != r._ptr)
        {
            if(r._ptr)
            {
                upCast(r._ptr)->__incRef();
            }

            T* ptr = this->_ptr;
            this->_ptr = r._ptr;

            if(ptr)
            {
                upCast(ptr)->__decRef();
            }
        }
        return *this;
    }
        
    template<class Y>
    static Handle dynamicCast(const ::IceUtil::HandleBase<Y>& r)
    {
#ifdef __BCPLUSPLUS__
        return Handle<T>(dynamic_cast<T*>(r._ptr));
#else
        return Handle(dynamic_cast<T*>(r._ptr));
#endif
    }

    template<class Y>
    static Handle dynamicCast(Y* p)
    {
#ifdef __BCPLUSPLUS__
        return Handle<T>(dynamic_cast<T*>(p));
#else
        return Handle(dynamic_cast<T*>(p));
#endif
    }

    void __clearHandleUnsafe()
    {
        this->_ptr = 0;
    }
};

}

#endif
