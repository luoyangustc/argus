#ifndef __SIGNAL_OBJ__
#define __SIGNAL_OBJ__

#include <stdio.h>
#include <boost/shared_ptr.hpp>
#ifdef _WINDOWS
#include <windows.h>
#else
#include <signal.h>
#define  WINAPI
#endif

class CSignalObject 
{
public:
    CSignalObject();
    ~CSignalObject();
    void wait();
private:
    #ifdef _WINDOWS
    void set_event();
    HANDLE m_Event;
    #endif
};

typedef boost::shared_ptr<CSignalObject> CSignalObject_ptr;

#endif /* defined(__SIGNAL_OBJ__) */
