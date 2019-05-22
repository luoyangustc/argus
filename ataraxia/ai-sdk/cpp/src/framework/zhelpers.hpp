#ifndef __ZHELPERS_HPP_INCLUDED__
#define __ZHELPERS_HPP_INCLUDED__

#include <time.h>

#if (!defined(WIN32))
#include <sys/time.h>
#include <unistd.h>
#endif

//  Provide random number from 0..(num-1)
#define within(num) (int)((float)(num)*random() / (RAND_MAX + 1.0))

//  Convert string to 0MQ string and send to socket
static bool
s_send(zmq::socket_t &socket, const std::string &string) {
  zmq::message_t message(string.size());
  memcpy(message.data(), string.data(), string.size());

  bool rc = socket.send(message);
  return (rc);
}

#if (!defined(WIN32))
//  Set simple random printable identity on socket
//  Caution:
//    DO NOT call this version of s_set_id from multiple threads on MS Windows
//    since s_set_id will call rand() on MS Windows. rand(), however, is not
//    reentrant or thread-safe. See issue #521.
inline std::string
s_set_id(zmq::socket_t &socket) {
  std::stringstream ss;
  ss << std::hex << std::uppercase
     << std::setw(4) << std::setfill('0') << within(0x10000) << "-"
     << std::setw(4) << std::setfill('0') << within(0x10000);
  socket.setsockopt(ZMQ_IDENTITY, ss.str().c_str(), ss.str().length());
  return ss.str();
}
#else
// Fix #521
inline std::string
s_set_id(zmq::socket_t &socket, intptr_t id) {
  std::stringstream ss;
  ss << std::hex << std::uppercase
     << std::setw(4) << std::setfill('0') << id;
  socket.setsockopt(ZMQ_IDENTITY, ss.str().c_str(), ss.str().length());
  return ss.str();
}
#endif

static void
s_version_assert(int want_major, int want_minor) {
  int major, minor, patch;
  zmq_version(&major, &minor, &patch);
  if (major < want_major || (major == want_major && minor < want_minor)) {
    std::cout << "Current 0MQ version is " << major << "." << minor << std::endl;
    std::cout << "Application needs at least " << want_major << "." << want_minor
              << " - cannot continue" << std::endl;
    exit(EXIT_FAILURE);
  }
}

//  Return current system clock as milliseconds
static int64_t
s_clock(void) {
#if (defined(WIN32))
  FILETIME fileTime;
  GetSystemTimeAsFileTime(&fileTime);
  unsigned __int64 largeInt = fileTime.dwHighDateTime;
  largeInt <<= 32;
  largeInt |= fileTime.dwLowDateTime;
  largeInt /= 10000;  // FILETIME is in units of 100 nanoseconds
  return (int64_t)largeInt;
#else
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (int64_t)(tv.tv_sec * 1000 + tv.tv_usec / 1000);
#endif
}

//  Sleep for a number of milliseconds
static void
s_sleep(int msecs) {
#if (defined(WIN32))
  Sleep(msecs);
#else
  struct timespec t;
  t.tv_sec = msecs / 1000;
  t.tv_nsec = (msecs % 1000) * 1000000;
  nanosleep(&t, NULL);
#endif
}

#endif