#include <string.h>
#ifndef _WINDOWS
#include <pthread.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/syscall.h>
#include <fcntl.h>
#include <errno.h>
#include <syslog.h>
#else
//#include <windows.h>
#endif //_WINDOWS

#include "Log.h"

const char* LOG_LEVEL_TEXT[EN_LOG_LEVEL_MAX] = { "FAT", "ERR", "CUR","WAR","DEB","TRA"};
CLog CLog::instance_;

int utoa_pad(unsigned long val, char *buf, int min_dig, int pad);
string get_local_time();
unsigned long int get_cur_thread_id();
std::string trim_class_func(const char* func);

CLog gAyNetLog;

void aynet_log_prepare( int level, uint32 log_size, const char* log_path )
{
    do 
    {
        if( gAyNetLog.IsOpen())
        {
            WARN_LOG("ay net log has open!");
            break;
        }

        if( !gAyNetLog.PrepareLog("ay_net_lib", level, log_size, log_path) )
        {
            printf("ay net log init failed, level(%d), max_log_file_size(%u), log_path(%s)!",
                level, log_size, log_path);
            break;

        }

        WARN_LOG("aynet log init, level(%d), max_log_file_size(%u), log_path(%s)!",
            level, log_size, log_path);

    } while (0);

}

void aynet_log_level_set( int level )
{
    if( !gAyNetLog.IsOpen())
    {
        return;
    }

    gAyNetLog.SetLevel(level);

    WARN_LOG("aynet log level(%d)!", level);
}

void aynet_log_size_set( uint32 max_size )
{
    if( !gAyNetLog.IsOpen())
    {
        return;
    }

    gAyNetLog.SetMaxFileSize(max_size*1024*1024);

    WARN_LOG("aynet log max size(%dMB)!", max_size);
}

void aynet_log_write( int level, const char* file, const char* func, int line, const char* fmt, va_list ap)
{
    if( !gAyNetLog.IsOpen())
    {
        return;
    }

    if (level > EN_LOG_LEVEL_MAX)
    {
        return;
    }

    int ret = 0;
    char new_fmt[256]={0};
    const char* file_name = strrchr(file, '\\');
    if (!file_name)
    {
        file_name = strrchr(file, '/');
        if ( !file_name )
        {
            file_name = file;
        }
        else
        {
            ++file_name;
        }
    }
    else
    {
        ++file_name;
    }

#ifndef _WINDOWS
    ret = snprintf(new_fmt, sizeof(new_fmt)-1, "%s %s %d >>> %s", file_name, trim_class_func(func).c_str(), line, fmt);
#else
    ret = _snprintf(new_fmt, sizeof(new_fmt)-1, "%s %s %d >>> %s", file_name, trim_class_func(func).c_str(), line, fmt);
#endif
    if ( ret < 0 )
    {
        return;
    }
    new_fmt[ret] = '\0';

    gAyNetLog.WriteLog(level, new_fmt, ap);
}

unsigned long int get_cur_thread_id()
{
#ifndef _WINDOWS
    return (unsigned long int)syscall(SYS_gettid);//gettid();
#else
    return (unsigned long int)GetCurrentThreadId();
#endif
}

string get_local_time()
{
    char   strTime[128];
    memset(strTime, 0x0, sizeof(strTime));

#ifndef _WINDOWS
    struct timeval cur_tv;
    struct timezone tz;
    gettimeofday(&cur_tv,&tz);
    struct tm* local_time = localtime((time_t*)&cur_tv.tv_sec);
    sprintf(strTime, "%04d-%02d-%02d %02d:%02d:%02d:%05d", 
        local_time->tm_year + 1900, 
        local_time->tm_mon+1, 
        local_time->tm_mday,
        local_time->tm_hour,
        local_time->tm_min, 
        local_time->tm_sec, 
        (int)cur_tv.tv_usec );
#else    
    SYSTEMTIME sys;
    GetLocalTime( &sys );
    sprintf(strTime, "%04d-%02d-%02d %02d:%02d:%02d:%06d", sys.wYear, sys.wMonth, sys.wDay, sys.wHour,sys.wMinute,sys.wSecond, sys.wMilliseconds);
#endif
    return strTime;
}

int utoa_pad(unsigned long val, char *buf, int min_dig, int pad)
{
    char *p;
    int len;

    p = buf;
    do
    {
        unsigned long digval = (unsigned long)(val % 10);
        val /= 10;
        *p++ = (char)(digval + '0');
    } while (val > 0);

    len = p - buf;
    while (len < min_dig)
    {
        *p++ = (char)pad;
        ++len;
    }
    *p-- = '\0';

    do
    {
        char temp = *p;
        *p = *buf;
        *buf = temp;
        --p;
        ++buf;
    } while (buf < p);

    return len;
}

int utoa(unsigned long val, char *buf)
{
    return utoa_pad(val, buf, 0, 0);
}

std::string trim_class_func(const char* func)
{
    char buf[512];
    strncpy(buf,func,sizeof(buf)-1);
    char* pos = strchr(buf,'(');
    if (pos)
    {
        *pos = '\0';
    }

    bool blank = false;
    char* p = buf;
    while(*p != '\0')
    {
        if (*p == ' ')
        {
            blank = true;
        }
        else
        {
            if (blank)
            {
                return p;
            }
        }

        p++;
    }

    return buf;
}

CLog::CLog():log_level_(EN_LOG_LEVEL_WARNING),file_fd_(INVALIDE_FILE_FD)
{
}

CLog::~CLog()
{
    Reset();
}

CLog* CLog::GetLLog()
{
    return &instance_;
}

BOOL CLog::Reset()
{
    CCriticalSection lock( &log_guard_ );
    if(file_fd_)
    {
        CloseFile(file_fd_);
        file_fd_ = INVALIDE_FILE_FD;
    }
    
    file_name_.clear();

    return TRUE;

}

BOOL CLog::IsOpen()
{
    CCriticalSection lock( &log_guard_ );
    if( file_fd_ == INVALIDE_FILE_FD )
    {
        return FALSE;
    }
    return TRUE; 
}

BOOL CLog::PrepareLog(const CHAR* szPrefix, int iLevel, DWORD dwLogSize, const CHAR* szPath)
{
    CCriticalSection lock( &log_guard_ );
    if ( iLevel >= EN_LOG_LEVEL_MAX )
    {
        return FALSE;
    }

    if ( szPath && strlen(szPath) )
    {
        file_name_ = szPath;
        file_name_ += "/";
        file_name_ += szPrefix;
        file_name_ += ".log";
    }
    else
    {
        file_name_ = szPrefix;
        file_name_ += ".log";
    }

    if ( file_name_.length() > 255 )
    {
        file_name_ = "";
        return FALSE;
    }

    log_level_ = iLevel;
    max_file_size_ = dwLogSize;

    if ( PathFileExists(file_name_.c_str()) )
    {
        string bak_file_name = file_name_ + "." + GetFilePostfix();
        RenameFile(bak_file_name.c_str(), file_name_.c_str());
    }

    file_fd_ = OpenFile(file_name_.c_str());
    if ( file_fd_ == INVALIDE_FILE_FD )
    {
        WriteSysMsg("open log file failed!");
        return FALSE;
    }

    return TRUE;
}

void CLog::SetLevel(int iLevel)
{
    CCriticalSection lock( &log_guard_ );
    if ( iLevel >= EN_LOG_LEVEL_MAX )
    {
        return;
    }
    log_level_ = iLevel;
}

void CLog::SetMaxFileSize(DWORD max_size)
{
    CCriticalSection lock( &log_guard_ );
    max_file_size_ = max_size;
}

void CLog::WriteLog(int ilevel,const char * format, ...)
{
	va_list args;
    va_start(args, format);
    WriteLog(ilevel, format, args);
    va_end(args);
}

void CLog::WriteLog(int ilevel,const char * format, va_list args)
{
    CCriticalSection lock( &log_guard_ );
    if( ilevel > log_level_ )
    {
        return;
    }
    
    if( !RotateLogFile() )
    {
        return;
    }

    string strTime = get_local_time();
    if(strTime.empty())
    {
        return;
    }
    
    unsigned int i = 0;
    char* pos = log_buff_;

    i = 0;
    while( i < strTime.length() )
    {
        *pos++ = strTime[i++];
    }

    *pos++ = ' ';
    unsigned long int thread_id = get_cur_thread_id();
    pos += utoa_pad(thread_id, pos, 10, '0');

    *pos++ = ' ';
    strcpy(pos, LOG_LEVEL_TEXT[ilevel]);
    pos += strlen(LOG_LEVEL_TEXT[ilevel]);

    *pos++ = ' ';

    int max_size = sizeof(log_buff_) - (pos - log_buff_);
    int print_len = vsnprintf(pos, max_size, format, args);
    if( print_len < 0 )
    {
        print_len = sprintf(pos, "<logging error: msg too long>");
    }
    pos += print_len;

    *pos++ = '\n';

    curr_size_ = pos - log_buff_;

    if( WriteFile(log_buff_, curr_size_) < 0 )
    {
        log_buff_[curr_size_] = '\0';
        printf("write log failed-->%s", log_buff_);
        WriteSysMsg("write log failed!");
    }
}

BOOL CLog::RotateLogFile()
{
    ULONGLONG file_size = GetFileLen(file_name_.c_str());
    if( file_size >= (ULONGLONG)max_file_size_)
    {
        string bak_file_name = "";
        {
            bak_file_name = file_name_ + "." + GetFilePostfix();
        }
        RenameFile(bak_file_name.c_str(), file_name_.c_str());

        CloseFile(file_fd_);
        file_fd_ = INVALIDE_FILE_FD;
    }

    if(file_fd_ == INVALIDE_FILE_FD)
    {
        file_fd_ = OpenFile(file_name_.c_str());
        if(file_fd_ == INVALIDE_FILE_FD)
        {
            WriteSysMsg("rotate log file failed!");
            return false;
        }
    }
    return true;
}

void CLog::DumpDataBlock (int ilevel,PBYTE buffer, DWORD datalen)
{
    //
}

void CLog::WriteSysMsg(const char* szmsg)
{
#ifndef _WINDOWS
    char szname[64] = {0};
    int rslt = readlink("/proc/self/exe", szname, sizeof(szname)); 
    char* p = strrchr(szname, '/');
    openlog(p, LOG_CONS | LOG_PID, 0);
    syslog(LOG_ERR, "%s/n", szmsg);  
    closelog();
#else
    
#endif
}

FILE_FD CLog::OpenFile(const char *szfilename)
{
#ifndef _WINDOWS
    FILE_FD fd = open(szfilename, O_APPEND | O_CREAT | O_WRONLY | O_CLOEXEC, DEFFILEMODE);
#else
    FILE_FD fd = CreateFileA(szfilename, GENERIC_WRITE, FILE_SHARE_READ, NULL, CREATE_ALWAYS, FILE_FLAG_OVERLAPPED|FILE_FLAG_NO_BUFFERING, NULL);
    if (fd != INVALIDE_FILE_FD && GetLastError() == ERROR_ALREADY_EXISTS)
    {
        SetEndOfFile(fd);    // truncate the existing file
    }
#endif
    return fd;
}

BOOL CLog::WriteFile(const char* buf, size_t len)
{
#ifndef _WINDOWS
    while (len > 0) 
    {
        ssize_t r;
        do {} while ( (r = write(file_fd_, buf, len)) < 0 && errno == EINTR );
        if (r <= 0)
        {
            printf("write log failed, errno=%d", errno);
            return FALSE;
        }
        buf += r;
        len -= r;
    }
#else
    while (len > 0) 
    {
        DWORD r;
        BOOL ok = ::WriteFile(file_fd_, buf, len, &r, NULL);
        // We do not use an asynchronous file handle, so ok==false means an error
        if (!ok) return FALSE;;
        buf += r;
        len -= r;
    }
#endif

    return TRUE;
}

ULONGLONG CLog::GetFileLen(const char *szFilename)
{
    if ( !szFilename )
    {
       return 0; 
    }

#ifndef _WINDOWS
    struct stat buf;
    if (stat(szFilename, &buf) != 0)
    {
        return 0;
    }
    return buf.st_size;
#else  

    HANDLE hFile;
    DWORD sizeLo, sizeHi;
    ULONGLONG size;

    hFile = CreateFile(szFilename, 
        READ_CONTROL, 
        FILE_SHARE_READ | FILE_SHARE_WRITE, NULL,
        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE)
    {
        return -1;
    }

    sizeLo = GetFileSize(hFile, &sizeHi);
    if (sizeLo == INVALID_FILE_SIZE) 
    {
        DWORD dwStatus = GetLastError();
        if (dwStatus != NO_ERROR) 
        {
            CloseHandle(hFile);
            return -1;
        }
    }

    size = sizeHi;
    size = (size << 32) + sizeLo;

    CloseHandle(hFile);
    return size;

#endif
}

BOOL CLog::PathFileExists(const char * szFilename)
{
    if ( !szFilename )
    {
        return FALSE; 
    }

#ifndef _WINDOWS
    struct stat buf;
    if (stat(szFilename, &buf) != 0)
    {
        return FALSE;
    }
    return TRUE;
#else
    HANDLE hFile;

    hFile = CreateFile(szFilename, 
        FILE_LIST_DIRECTORY, 
        FILE_SHARE_READ, NULL,
        OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, NULL);

    if (hFile == INVALID_HANDLE_VALUE)
    {
        return FALSE;
    }
    CloseHandle(hFile);
    return TRUE;
#endif
}

BOOL CLog::DeleteFile(const char * szFilename)
{
    if ( !szFilename )
    {
        return FALSE; 
    }

#ifndef _WINDOWS
    if (unlink(szFilename)!=0) 
    {
        return FALSE;
    }
    return TRUE;
#else
    if (DeleteFile( szFilename ) == FALSE)
    {
        DWORD error = GetLastError();
        return FALSE;
    }
    return TRUE;
#endif
}
BOOL CLog::RenameFile(const char * szNewname,const char * szOldname)
{
    if ( !szNewname || !szOldname )
    {
        return FALSE; 
    }

#ifndef _WINDOWS
    if (rename(szOldname, szNewname) != 0) 
    {
        return FALSE;
    }
    return TRUE;
#else
    BOOL rc = MoveFileEx(szOldname, szNewname, MOVEFILE_COPY_ALLOWED|MOVEFILE_REPLACE_EXISTING);
    if( !rc )
    {
        DWORD error = GetLastError();
        return FALSE;
    }
    return TRUE;
#endif
}

void CLog::CloseFile(FILE_FD fd)
{
#ifndef _WINDOWS
    do {} while ((close(fd)) < 0 && errno == EINTR);
#else
    CloseHandle(fd);
#endif
}

string CLog::GetFilePostfix()
{
    char   strTime[128];
    memset(strTime, 0x0, sizeof(strTime));
#ifndef _WINDOWS
    struct timeval cur_tv;
    struct timezone tz;
    gettimeofday(&cur_tv,&tz);
    struct tm* local_time = localtime((time_t*)&cur_tv.tv_sec);
    sprintf(strTime, "%04d-%02d-%02d-%02d-%02d-%02d-%04d", 
        local_time->tm_year + 1900, 
        local_time->tm_mon+1, 
        local_time->tm_mday,
        local_time->tm_hour,
        local_time->tm_min, 
        local_time->tm_sec, 
        (int)cur_tv.tv_usec );
#else    
    SYSTEMTIME sys;
    GetLocalTime( &sys );
    sprintf(strTime, "%04d-%02d-%02d-%02d-%02d-%02d-%04d", sys.wYear, sys.wMonth, sys.wDay, sys.wHour,sys.wMinute,sys.wSecond, sys.wMilliseconds);
#endif
    return strTime;
}

CLogTest::CLogTest()
{
}

void CLogTest::Start()
{
    gAyNetLog.WriteSysMsg("start log test!");
    gAyNetLog.SetMaxFileSize(5*1024*1024);
    running_ = true;
    for(int i=0; i<10; ++i)
    {
        threads_.create_thread(boost::bind(&CLogTest::RunProc, this, i));
    }
}

void CLogTest::Stop()
{
    DEBUG_LOG("Enter!");
    running_ = false;
    threads_.join_all();
    DEBUG_LOG("Exit!");
}

int CLogTest::RunProc(int id)
{
    uint32 count = 0;
    while(running_)
    {
        string strTime = get_local_time();
        DEBUG_LOG("(%d,%u)-->cur_time=%s", id, count, strTime.c_str());

        count++;
        boost::this_thread::sleep(boost::posix_time::millisec(100));
    }

    DEBUG_LOG("tid(%d) exit!", id);
    
    return 0;
}

CLogTest::~CLogTest()
{
    if(running_)
    {
        Stop();
    }
}