#pragma  once

#include <string>
#include <stdio.h>
#include <string.h>
//#include <atomic>

#ifdef NDEBUG
#define hlog(level, ...) \
    do { \
        if (level<=Logger::getLogger().getLogLevel()) { \
            Logger::getLogger().logv(level, __FILE__, __LINE__, __PRETTY_FUNCTION__, __VA_ARGS__); \
        } \
    } while(0)
#else
#define hlog(level, ...) \
    do { \
        if (level<=Logger::getLogger().getLogLevel()) { \
            snprintf(0, 0, __VA_ARGS__); \
            Logger::getLogger().logv(level, __FILE__, __LINE__, __PRETTY_FUNCTION__ , __VA_ARGS__); \
        } \
    } while(0)

#endif

/******************************日志输出函数************************************/
//  日志级别[ FATAL  ERROR WARN INFO DEBUG TRACE ]

#define Trace(...) hlog(Logger::LLTRACE, __VA_ARGS__)
#define Debug(...) hlog(Logger::LLDEBUG, __VA_ARGS__)
#define Info(...) hlog(Logger::LLINFO, __VA_ARGS__)
#define Warn(...) hlog(Logger::LLWARN, __VA_ARGS__)
#define Error(...) hlog(Logger::LLERROR, __VA_ARGS__)
#define Fatal(...) hlog(Logger::LLFATAL, __VA_ARGS__)
/*******************************************************************************/	
	
#define fatalif(b, ...) do { if((b)) { hlog(Logger::LLFATAL, __VA_ARGS__); } } while (0)
#define check(b, ...) do { if((b)) { hlog(Logger::LLFATAL, __VA_ARGS__); } } while (0)
#define exitif(b, ...) do { if ((b)) { hlog(Logger::LLERROR, __VA_ARGS__); _exit(1); }} while(0)

#define getloglevelstr()  Logger::getLogger().getLogLevelStr()

#define setloglevel(l) Logger::getLogger().setLogLevel(l)
#define setlogfile(n) Logger::getLogger().setFileName(n)

struct Logger/*: private noncopyable*/ {
    enum LogLevel{LLFATAL=0, LLERROR, LLWARN, LLINFO, LLDEBUG, LLTRACE, LLALL};
    Logger();
    ~Logger();
    void logv(int level, const char* file, int line, const char* func, const char* fmt ...);

    void setFileName(const std::string& filename);
    void setLogLevel(const std::string& level);
    void setLogLevel(LogLevel level) { level_ = std::min(LLALL, std::max(LLFATAL, level)); }

    LogLevel getLogLevel() { return level_; }
    const char* getLogLevelStr() { return levelStrs_[level_]; }
    int getFd() { return fd_; }

    void adjustLogLevel(int adjust) { setLogLevel(LogLevel(level_+adjust)); }
    void setRotateInterval(long rotateInterval) { rotateInterval_ = rotateInterval; }
    static Logger& getLogger();
private:
    void maybeRotate();
    static const char* levelStrs_[LLALL+1];
    int fd_;
    volatile LogLevel level_;
    long lastRotate_;
    //std::atomic<int64_t> realRotate_;
    long realRotate_;
    pthread_mutex_t mutex_;
    long rotateInterval_;
    std::string filename_;
};
