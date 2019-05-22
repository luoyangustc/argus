
#ifndef __BUFFER_INFO__
#define __BUFFER_INFO__

#include <stdio.h>
#include <string.h>
#include <boost/thread.hpp>
#include <boost/shared_array.hpp>


struct SDataBuff
{
    unsigned int data_size_;
    unsigned int buff_size_;
    boost::shared_array<char> pbuff_;

    SDataBuff()
    {
        data_size_ = 0;
        buff_size_ = 0;
    }

    SDataBuff(unsigned int size)
    {
        pbuff_.reset(new char[size]);
        if( !pbuff_.get() )
        {
            return;
        }
        buff_size_ = size;
        data_size_ = 0;
    }

    SDataBuff(const SDataBuff& r)
    {
        data_size_ = r.data_size_;
        buff_size_ = r.buff_size_;
        pbuff_ = r.pbuff_;
    }

    const SDataBuff& operator = (const SDataBuff& r)
    {
        data_size_ = r.data_size_;
        buff_size_ = r.buff_size_;
        pbuff_ = r.pbuff_;

        return *this;
    }
    
    bool resize( unsigned int size )
    {
        pbuff_.reset(new char[size]);
        if( !pbuff_.get() )
        {
            return false;
        }

        buff_size_ = size;
        data_size_ = 0;

        return true;
    }

    void clear()
    {
        data_size_ = 0;
    }

    char* get_buffer() const
    {
        return pbuff_.get();
    }

    unsigned int buffer_size() const
    {
        return buff_size_;
    }

    unsigned int data_size() const
    {
        return data_size_;
    }

    bool copy_data( const void* data, unsigned int data_size)
    {
#if 1
        if( buff_size_ < data_size)
        {
            if( !resize(data_size) )
            {
                return false;
            }
        }
#else //test
        if( !resize(data_size) )
        {
            return false;
        }
#endif
        memcpy(pbuff_.get(), data, data_size);
        data_size_ = data_size;

        return true;
    }

    bool push_back( const void* data, unsigned int data_size)
    {
        if( buff_size_ < (data_size_+data_size) )
        {
            return false;
        }

        char* dest = pbuff_.get()+data_size_;
        char* src = (char*)data;
        unsigned int idx = 0;
        while(idx < data_size)
        {
            dest[idx] = src[idx];
            ++idx;
        }

        data_size_ += data_size;

        return true;
    }

    bool push_back( const char* str)
    {
        unsigned int str_len = strlen(str);
        if( buff_size_ < (data_size_+str_len+1) )
        {
            return false;
        }

        char *dest = pbuff_.get()+data_size_;
        strcpy(dest, str);
        dest[str_len] = '\0';

        data_size_ += str_len+1;

        return true;
    }

    bool pop_front( unsigned int size )
    {
        if( data_size_ <= size )
        {
            data_size_ = 0;
        }
        else
        {
            data_size_ -= size;
            char* src = pbuff_.get()+size;
            char* dest = pbuff_.get();
            memmove(dest, src, data_size_);
        }

        return true;
    }
};

#endif /* defined(__BUFFER_INFO__) */

