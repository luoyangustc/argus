#ifndef _COMMON_STREAM_SERV_H__
#define _COMMON_STREAM_SERV_H__

#include <stdio.h>
#include <time.h>
#include <cstdlib>
#include <pthread.h>
#include <arpa/inet.h>
#include <sys/time.h>
#include <sys/resource.h>

#include <set>
#include <map>
#include <list>
#include <stack>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <signal.h>
#include <iomanip>
using namespace std;

#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>
#include <boost/thread.hpp>
#include <boost/scoped_array.hpp>
#include <boost/thread/lock_guard.hpp>
#include <boost/thread/recursive_mutex.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>

#include "base/include/tick.h"
#include "base/include/typedefine.h"
#include "base/include/typedef_win.h"
#include "base/include/HostInfo.h"
#include "base/include/datastream.h"
#include "base/include/common_thread_base.h"
#include "base/include/common_thread_group.h"
#include "base/include/ConfigHelper.h"
#include "base/include/DeviceID.h"
#include "base/include/LFile.h"
#include "base/include/logging_posix.h"
#include "base/include/ParamParser.h"
#include "base/include/DaemonUtil.h"
#include "base/include/TokenMgr.h"
#include "base/include/DeviceChannel.h"
#include "base/include/LBitField.h"
#include "base/include/encry/crc32.h"

#include "protocol/include/protocol_header.h"
#include "protocol/include/protocol_client.h"
#include "protocol/include/protocol_device.h"
#include "protocol/include/protocol_stream.h"
#include "protocol/include/protocol_status.h"

#include "netlib_framework/include/AYServerApi.h"
#include "netlib_framework/include/IServerLogical.h"
#include "netlib_framework/include/AYClientApi.h"

using namespace protocol;

//#include "CommonStruct.h"

#endif