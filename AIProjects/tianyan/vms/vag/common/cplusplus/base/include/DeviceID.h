#ifndef __DEVICE_ID_H__
#define __DEVICE_ID_H__

#include "typedef_win.h"
#include "typedefine.h"
#include <string.h>
#include <list>
#include <map>
#include <algorithm>
#include <string>
#include <boost/shared_ptr.hpp>
using namespace std;

#define DEVICE_MAX_LEN (21)

#pragma pack (push, 1)

class CDeviceID
{
public:	
	CDeviceID(){clear();}
	CDeviceID(const device_id_t& did);
	CDeviceID(const char * szValue);
	CDeviceID(const CDeviceID& right);

	bool is_valid();
	int get_group(uint32 group_num);
	static bool generate_device_id(const string& orig_did,string& out_new_did);
 	bool get_raw_device_id(device_id_t& raw_did);
	void clear(){memset(this,0,sizeof(CDeviceID));}
	bool isempty() const {CDeviceID address;address.clear();return address == * this;}

	void getidstring(string & strSha1)const;
	bool operator == (const CDeviceID & first) const;
	bool operator != (const CDeviceID & first) const;
	bool operator >= (const CDeviceID & first) const;
	bool operator <= (const CDeviceID & first) const;
	bool operator > (const CDeviceID & first) const;
	bool operator < (const CDeviceID & first) const;	
	const CDeviceID & operator=(const char * szValue);
	CDeviceID&	operator=(const CDeviceID& right);
	inline int getIndex(uint8 group_num);

	void dump_info();
private:
	uint8 data_[DEVICE_MAX_LEN];
	uint8 data_len_;
};

#pragma pack (pop)	

typedef boost::shared_ptr<CDeviceID> CDeviceID_ptr;

#endif //__DEVICE_ID_H__

