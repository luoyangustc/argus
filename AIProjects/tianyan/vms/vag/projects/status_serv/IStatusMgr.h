#ifndef  __ISTATUS_MGR_H__
#define __ISTATUS_MGR_H__

#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>

class CDataStream;
class CHostInfo;

class IStatusMgr : boost::noncopyable
{
public:
	virtual ~IStatusMgr() {}
	virtual int OnStatusReport(CDataStream& recvds, CDataStream& sendds) = 0;
	virtual int OnSessionOffline(const CHostInfo& hi_remote) = 0;
};

typedef boost::shared_ptr<IStatusMgr> IStatusMgrPtr;

#endif  //__ISTATUS_MGR_H__