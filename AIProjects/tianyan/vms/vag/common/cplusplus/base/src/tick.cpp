#include "tick.h"
#ifdef __APPLE__
#include <mach/mach_time.h>
#else
#ifndef _WINDOWS
#include <time.h>
#else
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#endif
tick_t get_current_tick()
{
    tick_t timeTick;
    
#ifdef __APPLE__
    /*
    mach_timebase_info_data_t info;
    uint64_t machineTime;
    mach_timebase_info(&info);
    machineTime = mach_absolute_time();
    timeTick = (tick_t)(machineTime * info.numer /info.denom / 1000000);*/
    
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

    //return ((int)elapsedTimeInNanoseconds)&0xFFFF0;
    timeTick = (tick_t)elapsedTimeInNanoseconds;
#else

#ifndef _WINDOWS
    struct timespec time_tmp={0,0};
    clock_gettime(CLOCK_MONOTONIC,&time_tmp);
    
    timeTick = time_tmp.tv_sec;
    
    timeTick *= 1000;
    timeTick = timeTick + (time_tmp.tv_nsec/1000000);
#else
	timeTick = ::GetTickCount();
#endif 
    
#endif
    
    return timeTick;
}




