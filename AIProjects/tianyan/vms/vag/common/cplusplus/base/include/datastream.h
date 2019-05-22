#ifndef __DATASTREAM_H__
#define __DATASTREAM_H__

#include "typedefine.h"

#ifdef _WINDOWS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <stdarg.h>
#include <string.h>
#endif

#include <stdio.h> 
#include <assert.h> 

#ifndef ENABLE_EMBED_OS

#include <string>
#include <vector>
#include <list>
using namespace std;

#endif //ENABLE_EMBED_OS

class CDataStream  
{
public :
	CDataStream(signed char * szBuf,int isize)
	{
		good_bit_=true;
		size_ = isize;
		buffer_ = szBuf;
		current_ = szBuf;
	}
	CDataStream(char * szBuf,int isize)
	{
		good_bit_=true;
		size_ = isize;
		buffer_ = (signed char*)szBuf;
		current_ = (signed char*)szBuf;
	}
	CDataStream(unsigned char * szBuf,int isize)
	{
		good_bit_=true;
		size_ = isize;
		buffer_ = (signed char*)szBuf;
		current_ = (signed char*)szBuf;
	}
	~CDataStream()
	{
	}
    
	void clear()
	{
		current_ = buffer_;
		current_[0]=0;
	}
	signed char * getcurrent_pos()
	{
		return current_;
	}
	void move(int ilen)//当前指针向后移动ilen
	{
		assert((current_ + ilen) <= (buffer_ + size_));
		if(good_bit_ && (current_ + ilen) <= (buffer_ + size_))
		{
			current_ += ilen;
		}else{
			good_bit_	= false;
		}
	}
	void reset()
	{
		current_ = buffer_;
	}

	uint8 readuint8()
	{
		assert((current_ + 1) <= (buffer_ + size_));
		if(good_bit_ && (current_ + 1) <= (buffer_ + size_))
		{
			current_ ++;
			return *(current_-1);
		}
		good_bit_ = false;
		return (uint8)-1;
	}

	void writeuint8(uint8 btValue)
	{
		assert((current_ + 1) <= (buffer_ + size_));
		if(good_bit_ && (current_ + 1) <= (buffer_ + size_))
		{
			*current_ = btValue;
			current_ ++; 	
		}
		else
			good_bit_ = false;

	}

	uint16 readuint16()
	{
		assert((current_ + 2) <= (buffer_ + size_));
		if(good_bit_ && (current_ + 2) <= (buffer_ + size_))
		{
			current_ +=2;
			return *((uint16*)(current_-2));
		}

		good_bit_ = false;
		return (uint16)-1;
	}
	void writeuint16(uint16 wValue)
	{		
		assert((current_ + 2) <= (buffer_ + size_));
		if(good_bit_ && (current_ + 2) <= (buffer_ + size_))
		{
			*((uint16*)current_) = wValue;
			current_ +=2;
		}
		else
			good_bit_ = false;
	}
	int readint32()
	{
		assert((current_ + sizeof(int32)) <= (buffer_ + size_));
		if((current_ + sizeof(int32)) <= (buffer_ + size_))
		{
			current_ +=sizeof(int32);
			return *((int32*)(current_-sizeof(int32)));
		}
		good_bit_ = false;
		return 0;

	}
	void writeint32(int32 iValue)
	{
		assert((current_ + sizeof(int32)) <= (buffer_ + size_));
		if((current_ + sizeof(int32)) <= (buffer_ + size_))
		{
			*((int32*)current_) = iValue;
			current_ +=sizeof(int32);
		}
		else
			good_bit_ = false;
	}
	uint32 readuint32()
	{
		assert((current_ + 4) <= (buffer_ + size_));
		if(good_bit_ && (current_ + 4) <= (buffer_ + size_))
		{
			current_ +=4;
			return *((uint32*)(current_-4));
		}
		good_bit_ = false;
		return 0;
	}
	void writeuint32(uint32 dwValue)
	{
		assert((current_ + 4) <= (buffer_ + size_));
		if((current_ + 4) <= (buffer_ + size_))
		{
			*((uint32*)current_) = dwValue;
			current_ +=4;
		}
		else
			good_bit_ = false;
	}
	int64 readint64()
	{
		assert((current_ + 8) <= (buffer_ + size_));
		if(good_bit_ && (current_ + 8) <= (buffer_ + size_))
		{
			current_ +=8;
			return *((int64*)(current_-8));
		}
		
		good_bit_ = false;
		return (int64)-1;
	}
	void writeint64(int64 iValue)
	{
		assert((current_ + 8) <= (buffer_ + size_));
		if((current_ + 8) <= (buffer_ + size_))
		{
			*((int64*)current_) = iValue;
			current_ +=8;
		}
		else
			good_bit_ = false;
	}

	uint64 readuint64()
	{
		assert((current_ + 8) <= (buffer_ + size_));
		if(good_bit_ && (current_ + 8) <= (buffer_ + size_))
		{
			current_ +=8;
			return *((uint64*)(current_-8));
		}
		
		good_bit_ = false;
		return (uint64)-1;
	}
	void writeuint64(uint64 iValue)
	{
		assert((current_ + 8) <= (buffer_ + size_));
		if((current_ + 8) <= (buffer_ + size_))
		{
			*((uint64*)current_) = iValue;
			current_ +=8;
		}
		else
			good_bit_ = false;
	}

	bool readdata(uint32 dwLen,void * pbyData)
	{
		if(good_bit_ && (current_ + dwLen) <= (buffer_ + size_))
		{
			memcpy(pbyData,current_,dwLen);
			current_ +=dwLen;
			return true;
		}
		good_bit_ = false; 
		return false;
	}
	void writedata(void * pData,uint32 dwLen)
	{
		assert((current_ + dwLen) <= (buffer_ + size_));
		if((current_ + dwLen) <= (buffer_ + size_))
		{
			memcpy(current_,pData,dwLen);		
			current_ +=dwLen;
		}
		else
			good_bit_ = false;
	}

	char * readstring()
	{
		int ilen = 0;
		int buf_left = leavedata();
		bool good = false;
		for(ilen=0; good_bit_ && ilen<buf_left; ++ilen)
		{
			if(0==current_[ilen])
			{
				good	= true;
				break;
			}
		}

		static char szNull[256] = {'\0'};
		if(!good)
		{
			good_bit_	= false;
			
			return szNull;
		}
		char * szRes = (char*)current_;
		if(good_bit_ && (current_ + ilen) <= (buffer_ + size_))
		{
			current_ +=(ilen+1);
			return szRes;
		}
		good_bit_ = false;
		return szNull;
	}

	bool writestring(const char * szStr)
	{
		if(current_&&szStr)
		{
			int ilen = strlen((const char*)szStr);
			if((size_-(current_ - buffer_)) < (ilen +1))	
				return false;
			memcpy(current_,szStr,ilen+1);
			current_ += (ilen+1);				
			return true;
		}
		return false;
	}
	
	bool writestring(const signed char * szStr)
	{
		if(current_&&szStr)
		{
			int ilen = strlen((const char*)szStr);
			if((size_-(current_ - buffer_)) < (ilen +1))	
				return false;
			memcpy(current_,szStr,ilen+1);
			current_ += (ilen+1);				
			return true;
		}
		return false;
	}
    
    /*bool writestring(const std::string& str)
	{
		if(current_)
		{
			int ilen = str.length();
			if((size_-(current_ - buffer_)) < (ilen +1))	
				return false;
			memcpy(current_,str.c_str(),ilen+1);
			current_ += (ilen+1);				
			return true;
		}
		return false;
	}*/
    
	void good_bit(bool flag){good_bit_=flag;}
	bool good_bit()
	{
		return good_bit_;
	}

	int size()
	{
		return (int)(current_-buffer_);
	}
	int leavedata()
	{
		return size_-size();
	}
	const signed char * getbuffer(){return buffer_;}
	int getbuffer_length(void)const{return size_;}
protected :
	bool 	good_bit_;
	signed char* buffer_;
	signed char* current_;
	int size_;
};


//输入流
inline CDataStream & operator >> (CDataStream &is, unsigned long & x)
{
    x = is.readuint32();
    return  is;
}

inline CDataStream & operator >> (CDataStream &is, uint32 & x)
{
	x = is.readuint32();
	return  is;
}
inline CDataStream & operator >> (CDataStream &is, uint16 & x)
{
	x = is.readuint16();
	return  is;
}
inline CDataStream & operator >> (CDataStream &is, uint8 & x)
{
	x = is.readuint8();
	return  is;
}

inline CDataStream & operator >> (CDataStream &is, int64 & x)
{
	x = is.readint64();
	return  is;
}

inline CDataStream & operator >> (CDataStream &is, int32 & x)
{
	x = is.readint32();
	return  is;
}

inline CDataStream & operator >> (CDataStream &is, uint64 & x)
{
	x = is.readuint64();
	return  is;
}
#ifndef ENABLE_EMBED_OS
inline CDataStream & operator >> (CDataStream &is, string & x)
{
	char * pstr = is.readstring();
	if(pstr)//如果是空指针,赋值给string会崩溃
		x = (const char*)pstr;
	return  is;
}
#endif //ENABLE_EMBED_OS


//输出流
inline CDataStream & operator << (CDataStream &os, unsigned long & x)
{
    os.writeuint32(x);
    return  os;
}

inline CDataStream & operator << (CDataStream &os, uint32 & x)
{
	os.writeuint32(x);
	return  os;
}
inline CDataStream & operator << (CDataStream &os, uint16 & x)
{
	os.writeuint16(x);
	return  os;
}
inline CDataStream & operator << (CDataStream &os, uint8 & x)
{
	os.writeuint8(x );
	return  os;
}

inline CDataStream & operator << (CDataStream &os, int64 & x)
{
	os.writeint64(x);
	return  os;
}
inline CDataStream & operator << (CDataStream &os, int32 & x)
{
	os.writeint32(x);
	return  os;
}

inline CDataStream & operator << (CDataStream &os, uint64 & x)
{
	os.writeuint64(x);
	return  os;
}
#ifndef ENABLE_EMBED_OS
inline CDataStream & operator << (CDataStream &os, string & x)
{
	os.writestring((const signed char*)x.c_str());
	return  os;
}
#endif //ENABLE_EMBED_OS

class CNetworkByteOrder
{
public:
	static uint16 convert(uint16 iValue)
	{
		uint16 iData;
		((uint8*)&iData)[0] = ((uint8*)&iValue)[1];
		((uint8*)&iData)[1] = ((uint8*)&iValue)[0];
		return iData;
	}
	static uint32 convert(uint32 iValue)
	{
		uint32 iData;
		((uint8*)&iData)[0] = ((uint8*)&iValue)[3];
		((uint8*)&iData)[1] = ((uint8*)&iValue)[2];
		((uint8*)&iData)[2] = ((uint8*)&iValue)[1];
		((uint8*)&iData)[3] = ((uint8*)&iValue)[0];
		return iData;
	}
	static uint64 convert(uint64 iValue)
	{
		uint64 iData;
		((uint8*)&iData)[0] = ((uint8*)&iValue)[7];
		((uint8*)&iData)[1] = ((uint8*)&iValue)[6];
		((uint8*)&iData)[2] = ((uint8*)&iValue)[5];
		((uint8*)&iData)[3] = ((uint8*)&iValue)[4];
		((uint8*)&iData)[4] = ((uint8*)&iValue)[3];
		((uint8*)&iData)[5] = ((uint8*)&iValue)[2];
		((uint8*)&iData)[6] = ((uint8*)&iValue)[1];
		((uint8*)&iData)[7] = ((uint8*)&iValue)[0];
		return iData;
	}

	static int16 convert(int16 iValue)
	{
		int16 iData;
		((uint8*)&iData)[0] = ((uint8*)&iValue)[1];
		((uint8*)&iData)[1] = ((uint8*)&iValue)[0];
		return iData;
	}
	static int32 convert(int32 iValue)
	{
		int32 iData;
		((uint8*)&iData)[0] = ((uint8*)&iValue)[3];
		((uint8*)&iData)[1] = ((uint8*)&iValue)[2];
		((uint8*)&iData)[2] = ((uint8*)&iValue)[1];
		((uint8*)&iData)[3] = ((uint8*)&iValue)[0];
		return iData;
	}
	static int64 convert(int64 iValue)
	{
		int64 iData;
		((uint8*)&iData)[0] = ((uint8*)&iValue)[7];
		((uint8*)&iData)[1] = ((uint8*)&iValue)[6];
		((uint8*)&iData)[2] = ((uint8*)&iValue)[5];
		((uint8*)&iData)[3] = ((uint8*)&iValue)[4];
		((uint8*)&iData)[4] = ((uint8*)&iValue)[3];
		((uint8*)&iData)[5] = ((uint8*)&iValue)[2];
		((uint8*)&iData)[6] = ((uint8*)&iValue)[1];
		((uint8*)&iData)[7] = ((uint8*)&iValue)[0];
		return iData;
	}
};

#endif //__DATASTREAM_H__
