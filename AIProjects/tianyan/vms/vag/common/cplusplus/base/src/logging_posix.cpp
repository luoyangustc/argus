#include "logging_posix.h"
#include <assert.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <fcntl.h>
#include <time.h>
#include <utility>
#include <stdarg.h>
#include <unistd.h>
#include <errno.h>
#include <syslog.h>
#include <string>

using namespace std;

Logger logger;

static void create_log_prefix_dir(const char* filepath);
static const char* trim_file_path(const char* file);
static std::string trim_class_func(const char* func);

Logger::Logger(): level_(LLINFO), lastRotate_(time(NULL)), rotateInterval_(86400) {
		pthread_mutex_init(&mutex_, NULL);
    tzset();
    fd_ = -1;
    realRotate_ = lastRotate_;
}

Logger::~Logger() {
    if (fd_ != -1) {
        close(fd_);
    }
    pthread_mutex_destroy(&mutex_);
}

const char* Logger::levelStrs_[LLALL+1] = {
    "FATAL",
    "ERROR",
    "WARN",
    "INFO",
    "DEBUG",
    "TRACE",
    "ALL",
};

Logger& Logger::getLogger() {
    //static Logger logger;
    return logger;
}

void Logger::setLogLevel(const string& level) {
    LogLevel ilevel = LLDEBUG;
    for (size_t i = 0; i < sizeof(levelStrs_)/sizeof(const char*); i++) {
        if (strcasecmp(levelStrs_[i], level.c_str()) == 0) {
            ilevel = (LogLevel)i;
            break;
        }
    }
    setLogLevel(ilevel);
}

void Logger::setFileName(const string& filename) {
		create_log_prefix_dir(filename.c_str());
		
    int fd = open(filename.c_str(), O_APPEND|O_CREAT|O_WRONLY|O_CLOEXEC, DEFFILEMODE);
    if (fd < 0) {
        fprintf(stderr, "open log file %s failed. msg: %s ignored\n",
                filename.c_str(), strerror(errno));
        return;
    }
    filename_ = filename;
    if (fd_ == -1) {
        fd_ = fd;
    } else {
        int r = dup2(fd, fd_);
        fatalif(r<0, "dup2 failed");
        close(fd);
    }
}

void Logger::maybeRotate() {
    time_t now = time(NULL);
    if (filename_.empty() || (now - timezone) / rotateInterval_ == (lastRotate_ - timezone)/ rotateInterval_) {
        return;
    }
    lastRotate_ = now;
    //long old = realRotate_.exchange(now);
    
    pthread_mutex_lock(&mutex_);
    int64_t pre_realRotate = realRotate_;
    realRotate_ = now;
    long old = pre_realRotate;
    pthread_mutex_unlock(&mutex_);
    
    //如果realRotate的值是新的，那么返回，否则，获得了旧值，进行rotate
    if ((old - timezone) / rotateInterval_ == (lastRotate_ - timezone) / rotateInterval_) {
        return;
    }

    struct tm ntm;
	time_t now2 = now - rotateInterval_;
    localtime_r(&now2, &ntm);
    char newname[4096];
    snprintf(newname, sizeof(newname), "%s.%d%02d%02d",
        filename_.c_str(), ntm.tm_year + 1900, ntm.tm_mon + 1, ntm.tm_mday);
    const char* oldname = filename_.c_str();
    int err = rename(oldname, newname);
    if (err != 0) {
        fprintf(stderr, "rename logfile %s -> %s failed msg: %s\n",
            oldname, newname, strerror(errno));
        return;
    }
    int fd = open(filename_.c_str(), O_APPEND | O_CREAT | O_WRONLY | O_CLOEXEC, DEFFILEMODE);
    if (fd < 0) {
        fprintf(stderr, "open log file %s failed. msg: %s ignored\n",
            newname, strerror(errno));
        return;
    }
    dup2(fd, fd_);
    close(fd);
}

void Logger::logv(int level, const char* file, int line, const char* func, const char* fmt ...) {
    if (level > level_) {
        return;
    }
    maybeRotate();
    char buffer[4*1024];
    char* p = buffer;
    char* limit = buffer + sizeof(buffer);

    struct timeval now_tv;
    gettimeofday(&now_tv, NULL);
    const time_t seconds = now_tv.tv_sec;
    struct tm t;
    localtime_r(&seconds, &t);
    /*
    p += snprintf(p, limit - p,
        "%04d-%02d-%02d %02d:%02d:%02d.%06d %lx %s %s:%d:%s ",
        t.tm_year + 1900,
        t.tm_mon + 1,
        t.tm_mday,
        t.tm_hour,
        t.tm_min,
        t.tm_sec,
        static_cast<int>(now_tv.tv_usec),
        (long)syscall(SYS_gettid),
        levelStrs_[level],
        trim_file_path(file),
        line,
		trim_class_func(func).c_str());*/
    p += snprintf(p, limit - p,
        "%04d-%02d-%02d %02d:%02d:%02d.%06d %lx %s %s %d >>> ",
        t.tm_year + 1900,
        t.tm_mon + 1,
        t.tm_mday,
        t.tm_hour,
        t.tm_min,
        t.tm_sec,
        static_cast<int>(now_tv.tv_usec),
        (long)syscall(SYS_gettid),
        levelStrs_[level],
        trim_class_func(func).c_str(),
        line);
    va_list args;
    va_start(args, fmt);
    p += vsnprintf(p, limit-p, fmt, args);
    va_end(args);
    p = std::min(p, limit - 2);
    //trim the ending \n
    while (*--p == '\n') {
    }
    *++p = '\n';
    *++p = '\0';
    int fd = fd_ == -1 ? 1 : fd_;
    int err = ::write(fd, buffer, p - buffer);
    if (err != p-buffer) {
        fprintf(stderr, "write log file %s failed. written %d errmsg: %s\n",
            filename_.c_str(), err, strerror(errno));
    }
    if (level <= LLERROR) {
        syslog(LOG_ERR, "%s", buffer+27);
    }
    if (level == LLFATAL) {
        fprintf(stderr, "%s", buffer);
        //assert(0);
    }
}

//
static void create_log_prefix_dir(const char* filepath) 
{
    char tmpPath[260] = { 0 };
    const char* p = strrchr(filepath, '/');
    if (!p) {
        return;
    }

    strncpy(tmpPath, filepath, p-filepath);
    if(access(tmpPath,0) == 0) {
        return;
    }

    memset(tmpPath, 0, sizeof(tmpPath));

    const char* pCur = filepath;
    int pos = 0;
    while(1) {
        if (pCur >= p) {
            break;
        }

        pCur++;
        tmpPath[pos++] = *(pCur-1);

        if(*pCur == '/' || *pCur == '\0') {
            if(access(tmpPath,0) != 0 && strlen(tmpPath) > 0) {
				mkdir(tmpPath, 0755);
            }
        }
    }
}

static const char* trim_file_path(const char* file) 
{
	const char* pos = strrchr(file,'/');
	if (pos)
		return pos+1;
	else
		return file;
}

static std::string trim_class_func(const char* func)
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
    char* q = p;
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
				q = p;
                blank = false;
			}
		}

		p++;
	}

	return q;
}
