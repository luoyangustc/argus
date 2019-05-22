#include <boost/bind.hpp>
#include <boost/function.hpp>
#include "SignalObject.h"

#ifdef _WINDOWS
boost::function0<void> console_ctrl_function;
BOOL WINAPI console_ctrl_handler(DWORD ctrl_type)
{
    switch (ctrl_type)
    {
    case CTRL_C_EVENT:
    case CTRL_BREAK_EVENT:
        console_ctrl_function();
        return TRUE;
    case CTRL_CLOSE_EVENT:
    case CTRL_SHUTDOWN_EVENT:
        console_ctrl_function();
        return TRUE;
    default:
        return FALSE;
    }
}

void CSignalObject::set_event()
{
    if (! SetEvent(m_Event) ) 
    {
        printf("SetEvent failed (%d)\n", GetLastError());
        return;
    }
}
#endif

CSignalObject::CSignalObject()
{
#ifdef _WINDOWS
    m_Event = CreateEvent( 
        NULL,               // default security attributes
        TRUE,               // manual-reset event
        FALSE,              // initial state is nonsignaled
        TEXT("WaitEvent")  // object name
        );
    if (m_Event == NULL) 
    { 
        printf("CreateEvent failed (%d)\n", GetLastError());
        return;
    }
    console_ctrl_function = boost::bind(&CSignalObject::set_event, this);
    SetConsoleCtrlHandler(console_ctrl_handler, TRUE);
#endif
}

void CSignalObject::wait()
{
#ifdef _WINDOWS
    WaitForSingleObject(m_Event, INFINITE);
#else
    sigset_t wait_mask;
    sigemptyset(&wait_mask);
    sigaddset(&wait_mask, SIGINT);
    sigaddset(&wait_mask, SIGQUIT);
    sigaddset(&wait_mask, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &wait_mask, 0);
    int sig = 0;
    sigwait(&wait_mask, &sig);
#endif
}

CSignalObject::~CSignalObject()
{
#ifdef _WINDOWS
    CloseHandle(m_Event);
#endif
}