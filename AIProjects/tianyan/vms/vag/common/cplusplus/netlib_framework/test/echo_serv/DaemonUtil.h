/*************************************************************************
 Author: Lirunhua
 Created Time: 2011年12月23日 星期五 01时17分29秒
 File Name: DaemonUtil.h
 Description: 
 ************************************************************************/
#ifndef __VD_DAEMON_UTIL_H__
#define __VD_DAEMON_UTIL_H__
#ifndef _WINDOWS
#include <unistd.h>
#include <sys/param.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <signal.h>
#include <unistd.h>
#include <stdlib.h>
#include <fcntl.h>
#include <syslog.h>
#include <string.h>
#include <errno.h>
#include <stdio.h>
#include <sys/resource.h>
#define LOCKMODE (S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH)
#endif

void init_daemon(void);
int lockfile(int);
int already_running(const char szLockFile[]);
int test_running(const char szLockFile[]);
int test_local_process_running(const char szLockFile[]);
#endif

