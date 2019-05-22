#ifndef __CRITICALSECTION_MGR_H__
#define __CRITICALSECTION_MGR_H__

#pragma once

#ifdef _WINDOWS
#define CCriticalSection CCriticalSection_win
#else

#include <pthread.h>
#ifndef PTHREAD_MUTEX_RECURSIVE_NP
#define PTHREAD_MUTEX_RECURSIVE_NP PTHREAD_MUTEX_RECURSIVE
#endif

#endif //

#include "typedef_win.h"

class CCriticalSectionMgr
{
public:
	CCriticalSectionMgr(DWORD dwSpinCount = 4000)
	{
	    m_iRef = 0;
#ifdef _WINDOWS
		InitializeCriticalSection(&m_cs);
#else
	    pthread_mutexattr_t attr;
	    pthread_mutexattr_init(&attr);
	    pthread_mutexattr_settype(&attr,PTHREAD_MUTEX_RECURSIVE_NP);//pthread_mutexattr_settype(&ma,PTHREAD_MUTEX_ERRORCHECK); 
	    pthread_mutexattr_setpshared(&attr,PTHREAD_PROCESS_SHARED);

	    pthread_mutex_init(&m_pmt,&attr);
	    pthread_mutexattr_destroy(&attr);
#endif //_WINDOWS

	}


	~CCriticalSectionMgr()
	{
#ifdef _WINDOWS
		DeleteCriticalSection(&m_cs);
#else
	    pthread_mutex_destroy(&m_pmt);
#endif //_WINDOWS
	}
	
	void lock()
	{
		m_iRef++;
#ifdef _WINDOWS
		EnterCriticalSection(&m_cs);
#else
		pthread_mutex_lock(&m_pmt);
#endif //_WINDOWS
	}
	void unlock()
	{
#ifdef _WINDOWS
		LeaveCriticalSection(&m_cs);
#else
		pthread_mutex_unlock(&m_pmt);
#endif //_WINDOWS
		m_iRef--;
	}

	BOOL canlock(){return m_iRef?FALSE:TRUE;}
	void SetName(const char* szName){}

	void lock(const char* szFunc, int nLine)
	{
		lock();
	}
#ifndef _WINDOWS	
	bool trylock()
	{
		return pthread_mutex_trylock(&m_pmt)==0;
	}
#endif //_WINDOWS

private:
#ifdef _WINDOWS
	CRITICAL_SECTION m_cs;
#else
	pthread_mutex_t m_pmt;
#endif //_WINDOWS
	int m_iRef;
};

class CCriticalSection
{
public:
	explicit CCriticalSection(CCriticalSectionMgr * pCS)
	{
	    if(pCS)pCS->lock();m_pCS = pCS;
	}

	CCriticalSection(CCriticalSectionMgr * pCS, const char* szFunc, int nLine)
	{
	    if(pCS)
		pCS->lock(szFunc, nLine);
	    m_pCS = pCS;
	}

	~CCriticalSection()	
	{
	    if(m_pCS) 
		m_pCS->unlock();
	}
private:
	CCriticalSectionMgr * m_pCS;
};

#endif

