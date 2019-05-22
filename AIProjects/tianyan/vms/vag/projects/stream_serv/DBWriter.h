#ifndef __DB_WRITER_H__
#define __DB_WRITER_H__

#include "CommonInc.h"
#include "third_party/sqlite3/include/SqliteDB.h"

class CDBWriter
{ 
#define  MAX_DB_FILE_SIZE 600*1024*1024

public:
	CDBWriter(string file_name);
	~CDBWriter();
	void update();

	//user interface
	/*void recordUserEvent(const MsgClientLoginRequest& msg, const char* pub_ip, const char* did, uint32 token_flag, int eventResult);
	void recordUserEvent(const MsgClientActionNofity& msg, const char* user_name, const char* pub_ip, const char* did, int eventResult);
	void recordUserEvent(const char *user_name, const char *pub_ip, const char* did, const char* action, int eventResult = 0);
	void recordUserEvent(const MsgC2SUserDataNotify& msg, const char *pub_ip, const char* did, int eventResult = 0);
	void recordUserEvent(const MsgC2SUserDataNotify& msg, const char* other_user, const char *pub_ip, const char* did, int eventResult = 0);

	//device interface
	void recordDeviceEvent(const MsgDeviceLoginRequest& msg, const char* pub_ip, const char* did, uint32 token_flag, int eventResult);
	void recordDeviceEvent(const char* pub_ip, const char* did, const char* action, int eventResult = 0);
	void recordDeviceEvent(const MsgDeviceActionNofity& msg, const char* pri_ip, const char* did, uint32 token_flag, int eventResult);

	//cloud storage interface
	void recordStorageEvent(const string& sql);
	void recordStorageEvent(const string& sql, const uint8* blobData, const uint32& blobDataLen);*/
private:
	bool checkDB();
	void cleanDBRecords();
	void getCurrentTime(char *timeBuf, int32 timeBufLen);
	void create_tables_for_stream_serv();
	void tryCommitSql();

private:
	boost::recursive_mutex lock_;
    string db_file_name_;
	CSqliteDB_ptr pSqliteDB_;
	CLFile db_file_;
	uint32 db_time_;
	enum { max_sql_list_count = 100};
	list<string> sqlBufferList_;
	tick_t last_clean_tick_;  
	tick_t last_exe_sql_tick_;  
};

typedef boost::shared_ptr<CDBWriter> CDBWriter_ptr;
#endif
