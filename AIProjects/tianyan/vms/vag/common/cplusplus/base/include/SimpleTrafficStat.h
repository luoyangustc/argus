#ifndef __SIMPLE_TRAFFIC_STAT_H__
#define __SIMPLE_TRAFFIC_STAT_H__

#include <boost/thread/recursive_mutex.hpp>
#include <boost/thread.hpp>
#include <map>

#include <fstream>
#include <sstream>
#include <iomanip>
using namespace std;

#include "tick.h"
#include "typedef_win.h"
#include "typedefine.h"
#include "GetTickCount.h"
class CSimpleTrafficStat
{
private:
	class CTrafficRecord
	{
	public:
		CTrafficRecord()
		{
			unSendBytes = 0;
			unRecvBytes = 0;
			unSendSpeed = 0;
			unRecvSpeed = 0;
		}
		unsigned int unSendBytes;//本秒投递请求的字节数
		unsigned int unRecvBytes;//本秒收到的字节数
		unsigned int unSendSpeed;
		unsigned int unRecvSpeed;
	};

	CSimpleTrafficStat(const CSimpleTrafficStat&){}
	CSimpleTrafficStat& operator=(const CSimpleTrafficStat&){return *this;}

public:
	CSimpleTrafficStat(void);
	~CSimpleTrafficStat(void);
	void Init(unsigned int unSecond = 60);
    void Reset();
	bool Send(unsigned int unBytes);
	bool Recv(unsigned int unBytes);
	unsigned int GetSendSpeed(unsigned int unSecond);
	unsigned int GetRecvSpeed(unsigned int unSecond);	
	inline long long GetTotalRecvBytes()
	{
		return m_nTotalRecvBytes;
	}
	inline long long GetTotalSendBytes()
	{
		return m_nTotalSendBytes;
	}

	inline unsigned int GetTotalSendTimes()
	{
		return m_unTotalSendTimes;
	}
	inline unsigned int GetTotalRecvTimes()
	{
		return m_unTotalRecvTimes;
	}

	inline unsigned int GetLastSendTick()
	{
		return m_unLastSendTick;
	}
	inline unsigned int GetLastRecvTick()
	{
		return m_unLastRecvTick;
	}

	inline unsigned int GetStartTick()
	{
		return m_unStartTick;
	}

	inline unsigned int GetStartExpired()
	{
		return GetTickCount() - m_unStartTick;
	}

	unsigned int GetSendExpired()
	{
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
		return GetTickCount() - m_unLastSendTick;
	}

	unsigned int GetRecvExpired()
	{
		boost::lock_guard<boost::recursive_mutex> lock(lock_);
		return GetTickCount() - m_unLastRecvTick;
	}

	unsigned int GetTotalSendSpeed();
	unsigned int GetTotalRecvSpeed();

	unsigned int GetPeakSendSpeed(){return m_unPeakUploadSpeed;}
	unsigned int GetPeakRecvSpeed(){return m_unPeakDownloadSpeed;}
    
	std::ostringstream& DumpInfo(std::ostringstream& oss);
    
public:
    static string ConvertSpeed2String(uint32 speed);
    static string ConvertFlux2String(uint64 flux);
    
private:
	void ClearTimeoutRecord(unsigned int unSec);
    
private:
	boost::recursive_mutex lock_;	
	unsigned int m_unSecond;//计算速率的时间
	long long m_nTotalSendBytes;
	long long m_nTotalRecvBytes;
	unsigned int m_unTotalSendTimes;
	unsigned int m_unTotalRecvTimes;
	unsigned int m_unLastSendTick;
	unsigned int m_unLastRecvTick;
	unsigned int m_unStartTick;
	std::map<unsigned int,CTrafficRecord> m_mapTrafficRecord;

	unsigned int m_unPeakDownloadSpeed;
	unsigned int m_unPeakUploadSpeed;

};

#endif //__SIMPLE_TRAFFIC_STAT_H__

