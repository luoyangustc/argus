#ifndef __VD_DAEMON_UTIL_H__
#define __VD_DAEMON_UTIL_H__

#include <signal.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <sys/resource.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdio.h>

void init_daemon(void);

int already_running(const char* szLockFile);

int test_running(const char* szLockFile);

#endif

