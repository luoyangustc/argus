#ifndef _WINDOWS
#include "GetTickCount.h"
#ifdef __APPLE__
#include <mach/mach_time.h>
#else
#include <sys/vfs.h>
#endif

#include <errno.h>
#include <stdio.h>
#include <string.h>

// ANSC C/C++
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <errno.h>

// linux
#include <unistd.h>
#include <string.h>
#include <fcntl.h>

#include <sys/types.h>
#include <netinet/in.h>
#include <sys/socket.h>  // socket
#include <arpa/inet.h>
#include <sys/times.h>  // time
#include <sys/select.h> 
#include <sys/ioctl.h>
#include <net/if.h>
//#include <net/if_arp.h>

#include <sys/wait.h>
#include <sys/stat.h>
unsigned int GetTickCount()
{
    static GetTickCountUtil s_tc;
    return s_tc.GetTickCount();
}

unsigned int GetTickCountUtil::GetTickCount()
{
    //CCriticalSection lock(&m_cs);	
#ifdef __APPLE__
    static uint64_t tick_begin = mach_absolute_time();
    uint64_t endTime = mach_absolute_time();
    
    // Elapsed time in mach time units
    uint64_t elapsedTime = endTime - tick_begin;
    
    // The first time we get here, ask the system
    // how to convert mach time units to nanoseconds
    static double g_ticksToNanoseconds = 0.0;
    if (0.0 == g_ticksToNanoseconds)
    {
        mach_timebase_info_data_t timebase;
        
        // to be completely pedantic, check the return code of this next call.
        mach_timebase_info(&timebase);
        g_ticksToNanoseconds = (double)timebase.numer / timebase.denom;
    }
    double elapsedTimeInNanoseconds = (elapsedTime * g_ticksToNanoseconds) / NSEC_PER_SEC*1000;//1000000000.0;
    return ((int)elapsedTimeInNanoseconds);//&0xFFFF0;
#else
    struct timespec time_tmp={0,0};
    clock_gettime(CLOCK_MONOTONIC,&time_tmp);
    unsigned int ret = time_tmp.tv_sec;
    ret *= 1000;
    ret = ret + ((time_tmp.tv_nsec/1000000));//&0xFFFF0
    return ret;
#endif
}
#else
#include "GetTickCount.h"
#endif