#include "base/include/DaemonUtil.h"

static int read_pid (const char *pidfile);
static int check_pid (const char *pidfile);
static int write_pid (const char *pidfile);

void init_daemon() 
{
	signal(SIGTTOU, SIG_IGN);

	signal(SIGTTIN, SIG_IGN);

	signal(SIGTSTP, SIG_IGN);

	if (0 != fork()) exit(0);

	if (-1 == setsid()) exit(0);

	signal(SIGHUP, SIG_IGN);

	if (0 != fork()) exit(0);

	if (0 != chdir("/")) exit(0);

	int fd;
	fd = open("/dev/null", O_RDWR, 0);
	if (fd != -1) {
		dup2(fd, STDIN_FILENO);
		dup2(fd, STDOUT_FILENO);
		dup2(fd, STDERR_FILENO);
		if (fd > 2) {
			close(fd);
		}
	}
}

int test_running(const char* szLockFile)
{
    return check_pid(szLockFile);
}

int already_running(const char* szLockFile)
{
    int ret = check_pid(szLockFile);
    if (ret)
        return ret;

    write_pid(szLockFile);

	return 0;
}

static int read_pid (const char *pidfile)
{
  FILE *f;
  int pid;

  if (!(f=fopen(pidfile,"r")))
    return 0;

  fscanf(f,"%d", &pid);
  fclose(f);
  return pid;
}

static int check_pid (const char *pidfile)
{
  int pid = read_pid(pidfile);

  if ((!pid) || (pid == getpid ()))
    return 0;

  if (kill(pid, 0) && errno == ESRCH)
	  return(0);

  return pid;
}

static int write_pid (const char *pidfile)
{
    FILE *f;
    int fd;
    int pid;

    if ( ((fd = open(pidfile, O_RDWR|O_CREAT, 0644)) == -1)
        || ((f = fdopen(fd, "r+")) == NULL) ) {
            fprintf(stderr, "Can't open or create %s.\n", pidfile);
            return 0;
    }

    if (flock(fd, LOCK_EX|LOCK_NB) == -1) {
        fscanf(f, "%d", &pid);
        fclose(f);
        printf("Can't lock, lock is held by pid %d.\n", pid);
        return 0;
    }

    pid = getpid();
    if (!fprintf(f,"%d\n", pid)) {
        printf("Can't write pid , %s.\n", strerror(errno));
        close(fd);
        return 0;
    }
    fflush(f);

    if (flock(fd, LOCK_UN) == -1) {
        printf("Can't unlock pidfile %s, %s.\n", pidfile, strerror(errno));
        close(fd);
        return 0;
    }

    close(fd);

    return pid;
}