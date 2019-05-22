#include "DBWriter.h"
#include "ServerLogical.h"

CDBWriter::CDBWriter(string file_name)
    : db_file_name_(file_name)
{
	db_time_ = 0;
	
	last_clean_tick_ = 0;
    last_exe_sql_tick_  = 0;
}

CDBWriter::~CDBWriter()
{
}

void CDBWriter::create_tables_for_stream_serv()
{
	do
	{
		bool executeSqlResult = false;

		executeSqlResult = pSqliteDB_->executeSql("CREATE TABLE IF NOT EXISTS USER(\
			ts TIMESTAMP,\
			name VCHAR(32),\
			pub_ip VCHAR(32),\
			pri_ip VCHAR(32),\
			action VCHAR(64),\
			mask INTEGER,\
			flag INTEGER,\
			did VCHAR(64),\
			token_flag VCHAR(64),\
			version VCHAR(32),\
			channel INTEGER,\
			rate INTEGER,\
			ter_id VCHAR(64),\
			result INTEGER)");
		if(!executeSqlResult)
		{
			Debug( "create USER table failed");
			break;
		}
		Debug( "create USER table success");

		executeSqlResult = pSqliteDB_->executeSql("CREATE TABLE IF NOT EXISTS DEVICE(\
												ts TIMESTAMP,\
												did VCHAR(64),\
												pub_ip VCHAR(32),\
												pri_ip VCHAR(32),\
												action VCHAR(64),\
												mask INTEGER,\
												flag INTEGER,\
												token_flag VCHAR(64),\
												version VCHAR(32),\
												status INTEGER,\
												dev_type INTEGER,\
												media_type INTEGER,\
												channel_num INTEGER,\
												min_rate INTEGER,\
												max_rate INTEGER,\
												time INT8,\
												tick INT8,\
												frm_rate INTEGER,\
												achannelcount INTEGER,\
												aSamplerate INTEGER,\
												abitLength INTEGER,\
												result INTEGER)");
		if(!executeSqlResult)
		{
			Debug( "create DEVICE table failed");
			break;
		}
		Debug( "create DEVICE table success");


		executeSqlResult = pSqliteDB_->executeSql("CREATE TABLE IF NOT EXISTS CLOUD_SEGMENTS("
												"ts TIMESTAMP,"
												"did VCHAR(32),"
												"channel_index INTEGER,"
												"upload_rate INTEGER,"
												"starttime INTEGER,"
												"endtime INTEGER,"
												"uniquename VCHAR(64))");
		if(!executeSqlResult)
		{
			Debug( "create CLOUD_SEGMENTS table failed");
			break;
		}
		Debug( "create CLOUD_SEGMENTS table success");


		executeSqlResult = pSqliteDB_->executeSql("CREATE TABLE IF NOT EXISTS CLOUD_BITMAPS("
												"ts TIMESTAMP,"
												"did VCHAR(32),"
												"cloud_name VCHAR(32),"
												"channel_index INTEGER,"
												"upload_rate INTEGER,"
												"starttime INTEGER,"
												"endtime INTEGER,"
												"uploadinfo TEXT)");
		if(!executeSqlResult)
		{
			Debug( "create CLOUD_BITMAPS table failed");
			break;
		}
		Debug( "create CLOUD_BITMAPS table success");

	}while(false);
}


bool CDBWriter::checkDB()
{
	do
	{
		if ( ( !db_time_ ) || ( !db_file_.IsOpen()) )
                {
                    Debug("CDBWriter::check_db-->need reset, db_time(%u), db_file(name=%s, open_flag=%u)\r\n", 
                        db_time_, db_file_.GetCurrFilename(), (uint32)db_file_.IsOpen());

                    tryCommitSql();
                    
                    db_time_ = (uint32)time(NULL);
                    //db_time_ = db_time_ / 3600 * 3600;
                    pSqliteDB_.reset();
                }

		if (!pSqliteDB_)
		{
			string file_name = db_file_name_;
			file_name += "/";
			file_name += boost::lexical_cast<string>(db_time_);
			file_name += ".db";

			pSqliteDB_.reset(new CSqliteDB(file_name.c_str()));
			if (!pSqliteDB_)
                        {
                            Debug("CDBWriter::check_db-->create db(%s) failed!\r\n", file_name.c_str());
                            break;
                        }

                        if( !db_file_.OpenFile(file_name.c_str()) )
                        {
                            Debug("CDBWriter::check_db-->open db_file(%s) failed!\r\n", file_name.c_str());
                            break;
                        }
                        
			create_tables_for_stream_serv();

		}

		return true;
		
	}while(false);
	return false;
}

void CDBWriter::cleanDBRecords()
{
#if 0
	//删除过期的数据库记录，最多保存30天
	if(db_time_)
	{
		uint32 minEffectiveTime = db_time_ - 30 * 24 * 3600; 
		char bashString[1024] = {'\0'};
		snprintf(bashString, 1024, "ls  %s/ | sed 's/.db$//g' | awk '$1<%u{print \"rm -rf %s/\"$1\".db\"|\"bash\"}'",
				db_file_name_.c_str(), minEffectiveTime, CServerLogical::GetLogical()->GetServCfg()->GetRecordPath().c_str());

	//		Debug( "checkDB:bashString: %s", bashString);
		std::system(bashString);
	}
#endif
}

void CDBWriter::tryCommitSql()
{
    tick_t current_tick = get_current_tick();
    
    boost::lock_guard<boost::recursive_mutex> lock(lock_);

    uint32 sqlListSize = sqlBufferList_.size();
    if( pSqliteDB_ && ( (sqlListSize >= max_sql_list_count) || ( ( current_tick-last_exe_sql_tick_)>5*1000) ) )
    {
        list<string> sqlBufferList;
        sqlBufferList_.swap(sqlBufferList);
        pSqliteDB_->executeSqlList(sqlBufferList);

        last_exe_sql_tick_ = current_tick;
    }
}

void CDBWriter::update()
{
	tryCommitSql();
    /*
	tick_t current_tick = get_current_tick();
    if( (current_tick-last_clean_tick_) >60*60*1000)
    {
        cleanDBRecords();
        last_clean_tick_ = current_tick;
    }*/
}

void CDBWriter::getCurrentTime(char *timeBuf, int32 timeBufLen)
{
	time_t nowtime = time(NULL);
	struct tm *local= localtime(&nowtime);
	strftime(timeBuf, timeBufLen, "%Y-%m-%d %H:%M:%S", local);  
}