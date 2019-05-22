#ifndef __GETTICKCOUNT_H__
#define __GETTICKCOUNT_H__

#ifndef _WINDOWS

#include <time.h>
#include <string>
using namespace std;   
#include <vector>	
#include <errno.h>

#include <sys/types.h>
#include <sys/syscall.h>

#include "typedef_win.h"
#include "CriticalSectionMgr.h"

unsigned int GetTickCount();
void Sleep(DWORD dwMilliseconds);

class GetTickCountUtil
{
public:
	GetTickCountUtil(){}
	~GetTickCountUtil(){}
	unsigned int GetTickCount();
private:
	GetTickCountUtil(const GetTickCountUtil& obj);
	GetTickCountUtil& operator=(const GetTickCountUtil& obj);
private:
	CCriticalSectionMgr m_cs;
};

unsigned int getTickCount(void);

#endif//NOT _WINDOWS
#endif

