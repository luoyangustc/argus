#ifndef __LOG__
#define __LOG__

#include <stdio.h>
#include <string>
#include <stdarg.h>
#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/thread/recursive_mutex.hpp>
#include "typedefine.h"
#include "typedef_win.h"
#include "CriticalSectionMgr.h"


using namespace std;

enum EN_LOG_LEVEL
{
    EN_LOG_LEVEL_FATAL = 0,
    EN_LOG_LEVEL_ERROR = 1,
    EN_LOG_LEVEL_CURRENT = 2,
    EN_LOG_LEVEL_WARNING = 3,
    EN_LOG_LEVEL_DEBUG = 4,
    EN_LOG_LEVEL_TRACE = 5,
    EN_LOG_LEVEL_MAX
};

#ifndef _WINDOWS
typedef int FILE_FD;
const FILE_FD INVALIDE_FILE_FD = -1;
#else
typedef HANDLE FILE_FD;
const FILE_FD INVALIDE_FILE_FD = INVALID_HANDLE_VALUE;
#endif

class CLog 
{
public:
	#define Log_Debug WriteLog
    static CLog* GetLLog();
private:
    static CLog instance_;
public:
    CLog();
    ~CLog();

    BOOL Reset();
    BOOL IsOpen();
    BOOL PrepareLog(const CHAR* szPrefix, int iLevel, DWORD dwLogSize=10*1024*1024, const CHAR* szPath = NULL);
    void SetLevel(int iLevel);
    void SetMaxFileSize(DWORD max_size);
    void WriteLog(int ilevel,const char * format, ...);
    void WriteLog(int ilevel,const char * format, va_list args);
    void DumpDataBlock (int ilevel,PBYTE buffer, DWORD datalen);
    void WriteSysMsg(const char* szmsg);
private:
    string GetFilePostfix();
    ULONGLONG GetFileLen(const char *szfilename);
    BOOL PathFileExists(const char * szFilename);
    BOOL RotateLogFile();

    FILE_FD OpenFile(const char *szfilename);
    BOOL WriteFile(const char* buf, size_t len);
    BOOL DeleteFile(const char * szFilename);
    BOOL RenameFile(const char * szNewname,const char * szOldname);
    void CloseFile(FILE_FD fd);
    
    //static BOOL IsPathValid(const char * szPathName);
private:
    FILE_FD file_fd_;
    string file_name_;
    DWORD max_file_size_;

    int log_level_;
    char log_buff_[8096];
    DWORD curr_size_;
    CCriticalSectionMgr log_guard_;
};

class CLogTest
{
public:
    CLogTest();
    ~CLogTest();
    void Start();
    void Stop();
    int RunProc(int id);
private:
    boost::recursive_mutex lock_;
    volatile bool running_;
    boost::thread_group threads_;
};

void aynet_log_prepare( int level, uint32 log_size, const char* log_path );
void aynet_log_level_set( int level );
void aynet_log_size_set( uint32 max_size ); //Unit:Mb
void aynet_log_write( int level, const char* file, const char* func, int line, const char* fmt, va_list ap);

#ifndef _WINDOWS
#define __AY_FUNCTION__     __PRETTY_FUNCTION__
#else
#define __AY_FUNCTION__     __FUNCTION__
#endif

inline void AYNET_LOG_PRINTF( int level, const char* file, const char* func, int line, const char* fmt, ... )
{
    va_list ap; 
    va_start(ap, fmt);
    aynet_log_write(level, file, func, line, fmt, ap);
    va_end(ap);
}

#ifndef _WINDOWS
#define FATAL_LOG(fmt, args...)    AYNET_LOG_PRINTF(EN_LOG_LEVEL_FATAL, __FILE__, __PRETTY_FUNCTION__, __LINE__, fmt, ##args)
#define ERROR_LOG(fmt, args...)    AYNET_LOG_PRINTF(EN_LOG_LEVEL_ERROR, __FILE__, __PRETTY_FUNCTION__, __LINE__, fmt, ##args)
#define WARN_LOG(fmt, args...)     AYNET_LOG_PRINTF(EN_LOG_LEVEL_WARNING, __FILE__, __PRETTY_FUNCTION__, __LINE__, fmt, ##args)
#define INFO_LOG(fmt, args...)     AYNET_LOG_PRINTF(EN_LOG_LEVEL_CURRENT, __FILE__, __PRETTY_FUNCTION__, __LINE__, fmt, ##args)
#define DEBUG_LOG(fmt, args...)    AYNET_LOG_PRINTF(EN_LOG_LEVEL_DEBUG, __FILE__, __PRETTY_FUNCTION__, __LINE__, fmt, ##args)
#define TRACE_LOG(fmt, args...)    AYNET_LOG_PRINTF(EN_LOG_LEVEL_TRACE, __FILE__, __PRETTY_FUNCTION__, __LINE__, fmt, ##args)
#else
#define FATAL_LOG(fmt, ...)        AYNET_LOG_PRINTF(EN_LOG_LEVEL_FATAL, __FILE__, __FUNCTION__, __LINE__, fmt, __VA_ARGS__)
#define ERROR_LOG(fmt, ...)        AYNET_LOG_PRINTF(EN_LOG_LEVEL_ERROR, __FILE__, __FUNCTION__, __LINE__, fmt, __VA_ARGS__)
#define WARN_LOG(fmt, ...)         AYNET_LOG_PRINTF(EN_LOG_LEVEL_WARNING, __FILE__, __FUNCTION__, __LINE__, fmt, __VA_ARGS__)
#define INFO_LOG(fmt, ...)         AYNET_LOG_PRINTF(EN_LOG_LEVEL_CURRENT, __FILE__, __FUNCTION__, __LINE__, fmt, __VA_ARGS__)
#define DEBUG_LOG(fmt, ...)        AYNET_LOG_PRINTF(EN_LOG_LEVEL_DEBUG, __FILE__, __FUNCTION__, __LINE__, fmt, __VA_ARGS__)
#define TRACE_LOG(fmt, ...)        AYNET_LOG_PRINTF(EN_LOG_LEVEL_TRACE, __FILE__, __FUNCTION__, __LINE__, fmt, __VA_ARGS__)
#endif

#endif /* defined(__LOG__) */
