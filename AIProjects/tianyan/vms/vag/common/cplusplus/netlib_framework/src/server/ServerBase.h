#ifndef __SERVER_BASE_H_
#define __SERVER_BASE_H_

#include <stdio.h>
#include <list>
#include <map>
#include <fstream>
#include <sstream>
#include <boost/shared_ptr.hpp>   
#include "HostInfo.h"
#include "IServerLogical.h"

enum EN_SERV_TYPE
{
    en_serv_type_udp = 1,
    en_serv_type_tcp = 2,
    en_serv_type_http = 3,
};

class IServerBase
{
public:
    virtual int Init(IServerLogical* pServerLogical) = 0;
    virtual int Start() = 0;
    virtual int Stop() = 0;
    virtual void Update() = 0;
    virtual string GetServerName() = 0;
    virtual std::ostringstream& DumpInfo(std::ostringstream& oss) = 0;
};

typedef boost::shared_ptr<IServerBase>                  IServerBase_ptr;
typedef std::list<IServerBase_ptr>                      IServerBaseList;
typedef std::list<IServerBase_ptr>::iterator            IServerBaseListIter;

#endif