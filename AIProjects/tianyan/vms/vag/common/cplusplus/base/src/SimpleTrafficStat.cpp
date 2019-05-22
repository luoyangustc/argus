#include "SimpleTrafficStat.h"
#include "to_string_util.h"

CSimpleTrafficStat::CSimpleTrafficStat(void)
{
    m_unSecond = 60; //计算速率的时间

    m_nTotalSendBytes = 0;
    m_nTotalRecvBytes = 0;
    m_unTotalSendTimes = 0;
    m_unTotalRecvTimes = 0;
    m_unLastSendTick = 0;
    m_unLastRecvTick = 0;
    m_unStartTick = 0;
    
    m_unPeakDownloadSpeed = 0;
    m_unPeakUploadSpeed = 0;

    m_mapTrafficRecord.clear();
}

CSimpleTrafficStat::~CSimpleTrafficStat(void)
{
}

void CSimpleTrafficStat::Init(DWORD unSecond)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    m_unStartTick = get_current_tick();
    m_unSecond = unSecond;
}

void CSimpleTrafficStat::ClearTimeoutRecord(unsigned int unSec)
{
	map<unsigned int,CTrafficRecord>::iterator it = m_mapTrafficRecord.begin();

	while( it != m_mapTrafficRecord.end() )
	{
		if( unSec < it->first )//清除掉异常情况
		{
			m_mapTrafficRecord.erase(it++);
		}
		else if( (unSec - it->first) > m_unSecond )
		{
			m_mapTrafficRecord.erase(it++);
		}
		else
		{
			break;
		}
	}
}

bool CSimpleTrafficStat::Send(unsigned int unBytes)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);

    m_unLastSendTick = get_current_tick();
    m_unTotalSendTimes++;

    unsigned int unSec = m_unLastSendTick/1000;
    map<unsigned int,CTrafficRecord>::iterator it = m_mapTrafficRecord.find(unSec);
    if(it != m_mapTrafficRecord.end())
    {
        it->second.unSendBytes +=unBytes;
    }
    else
    {
        CTrafficRecord tr;
        tr.unSendBytes = unBytes;
        it = m_mapTrafficRecord.insert(pair<unsigned int,CTrafficRecord>(unSec,tr)).first;
    }
    m_nTotalSendBytes+=unBytes;

    unsigned int nSpeed = GetSendSpeed(1);
    if(nSpeed > m_unPeakUploadSpeed)
    {
        m_unPeakUploadSpeed = nSpeed;
    }

    ClearTimeoutRecord(unSec);

    return true;
}

bool CSimpleTrafficStat::Recv(unsigned int unBytes)
{
	boost::lock_guard<boost::recursive_mutex> lock(lock_);
    
    m_unLastRecvTick = get_current_tick();
    m_unTotalRecvTimes++;

	unsigned int unSec = m_unLastRecvTick/1000;
	map<unsigned int,CTrafficRecord>::iterator it = m_mapTrafficRecord.find(unSec);
	if(it != m_mapTrafficRecord.end())
	{
		it->second.unRecvBytes +=unBytes;
	}
	else
	{
		CTrafficRecord tr;
		tr.unRecvBytes = unBytes;
		it = m_mapTrafficRecord.insert(pair<unsigned int,CTrafficRecord>(unSec,tr)).first;
	}

	m_nTotalRecvBytes+=unBytes;
    
    unsigned int nSpeed = GetRecvSpeed(1);
    if(nSpeed > m_unPeakUploadSpeed)
    {
        m_unPeakDownloadSpeed = nSpeed;
    }

	ClearTimeoutRecord(unSec);

	return true;
}

unsigned int CSimpleTrafficStat::GetSendSpeed(unsigned int unSecond)
{
	boost::lock_guard<boost::recursive_mutex> lock(lock_);
    unsigned int unSec = get_current_tick()/1000;
	ClearTimeoutRecord(unSec);

	unsigned int unStartTick = 0;
	unsigned int unTotalSendBytes = 0;
	for(map<unsigned int,CTrafficRecord>::reverse_iterator it=m_mapTrafficRecord.rbegin(); it!=m_mapTrafficRecord.rend(); ++it)
	{
		if((unSec - it->first) > unSecond)
		{			
			break;
		}
		unStartTick	= it->first;
		unTotalSendBytes += it->second.unSendBytes;
	}

	if(unSec>unStartTick && unStartTick)
	{
		return unTotalSendBytes * 8 / (unSec - unStartTick);
	}

	return 0;
}

unsigned int CSimpleTrafficStat::GetRecvSpeed(unsigned int unSecond)
{
	boost::lock_guard<boost::recursive_mutex> lock(lock_);

	unsigned int unSec = get_current_tick()/1000;
	ClearTimeoutRecord(unSec);

	unsigned int unStartTick = 0;
	unsigned int unTotalRecvBytes = 0;
	for(map<unsigned int,CTrafficRecord>::reverse_iterator it=m_mapTrafficRecord.rbegin(); it!=m_mapTrafficRecord.rend(); ++it)
	{
		if((unSec - it->first) > unSecond)
		{			
			break;
		}
		unStartTick	= it->first;
		unTotalRecvBytes += it->second.unRecvBytes;
	}

	if(unSec>unStartTick && unStartTick)
	{
		return unTotalRecvBytes * 8 / (unSec - unStartTick);
	}
	return 0;
}

unsigned int CSimpleTrafficStat::GetTotalSendSpeed()
{
    if(m_unStartTick)
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        return m_nTotalSendBytes * 8 / (GetStartExpired()/1000);
    }
    return 0;
}

unsigned int CSimpleTrafficStat::GetTotalRecvSpeed()
{
    if(m_unStartTick)
    {
        boost::lock_guard<boost::recursive_mutex> lock(lock_);
        return m_nTotalRecvBytes * 8 / (GetStartExpired()/1000);
    }
    return 0;
}

string CSimpleTrafficStat::ConvertSpeed2String(uint32 speed)
{
    char buff[128];

    memset(buff, 0x0, sizeof(buff));

    float BSpeed = (float)(speed/8.0);
    if ( BSpeed < 1024 )
    {
        sprintf(buff,"%.1fB/s",BSpeed);
    }
    else if ( BSpeed < 1024*1024 )
    {
        sprintf(buff,"%.1fKB/s",BSpeed/1024);
    }
    else if ( BSpeed < 1024*1024*1024 )
    {
        sprintf(buff,"%.1fMB/s",BSpeed/1024/1024);
    }
    else
    {
        sprintf(buff,"%.1fGB/s",BSpeed/1024/1024/1024);
    }

    return buff;
}

string CSimpleTrafficStat::ConvertFlux2String(uint64 flux)
{
    char buff[128];

    memset(buff, 0x0, sizeof(buff));

    if ( flux < 1024 )
    {
        sprintf(buff,"%uB",(int)flux);
    }
    else if ( flux < 1024*1024 )
    {
        sprintf(buff,"%.1fKB",flux/1024.0);
    }
    else if ( flux < 1024*1024*1024 )
    {
        sprintf(buff,"%.1fMB",flux/1024.0/1024);
    }
    else
    {
        sprintf(buff,"%.1fGB",flux/1024.0/1024/1024);
    }

    return buff;
}

void CSimpleTrafficStat::Reset()
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);
    m_unSecond = 60; //计算速率的时间

    m_nTotalSendBytes = 0;
    m_nTotalRecvBytes = 0;
    m_unTotalSendTimes = 0;
    m_unTotalRecvTimes = 0;
    m_unLastSendTick = 0;
    m_unLastRecvTick = 0;
    m_unStartTick = 0;

    m_unPeakDownloadSpeed = 0;
    m_unPeakUploadSpeed = 0;

    m_mapTrafficRecord.clear();
}

std::ostringstream& CSimpleTrafficStat::DumpInfo(std::ostringstream& oss)
{
    boost::lock_guard<boost::recursive_mutex> lock(lock_);

    oss << "{";
    {
        oss << "\"duration\":\"";
        oss << calc_time_unit(GetStartExpired()) << "\"";
        
        oss << ",";
        oss << "\"recv\":{";
        {
            oss << "\"flux\":\"";
            oss << calc_flux_unit(m_nTotalRecvBytes) << "\"";

            oss << ","; 
            oss << "\"count\":";
            oss << m_unTotalRecvTimes;

            oss << ","; 
            oss << "\"expire\":\"";
            oss << calc_time_unit(GetRecvExpired());

            oss << ","; 
            oss << "\"recent_speed\":\"";
            oss << calc_time_unit(GetRecvSpeed(m_unSecond));

            oss << ","; 
            oss << "\"avg_speed\":\"";
            oss << calc_time_unit(GetTotalRecvSpeed());

            oss << ","; 
            oss << "\"peak_speed\":\"";
            oss << calc_time_unit(GetPeakRecvSpeed());
        }

        oss << ",";
        oss << "\"send\":{";
        {
            oss << "\"flux\":\"";
            oss << calc_flux_unit(m_nTotalSendBytes) << "\"";

            oss << ","; 
            oss << "\"count\":";
            oss << m_unTotalSendTimes;

            oss << ","; 
            oss << "\"expire\":\"";
            oss << calc_time_unit(GetSendExpired());

            oss << ","; 
            oss << "\"recent_speed\":\"";
            oss << calc_time_unit(GetSendSpeed(m_unSecond));

            oss << ","; 
            oss << "\"avg_speed\":\"";
            oss << calc_time_unit(GetTotalSendSpeed());

            oss << ","; 
            oss << "\"peak_speed\":\"";
            oss << calc_time_unit(GetPeakSendSpeed());
        }
    }
    
    return oss;
}