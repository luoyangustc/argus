#include "SqliteDB.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

CSqliteDB::CSqliteDB(const char* pcszPath)
{
	do
	{
		int retVal = SQLITE_OK;
		retVal =  sqlite3_open(pcszPath,&m_DB);
		if(retVal != SQLITE_OK)
		{
			Debug( "create database: %s failed", pcszPath);
			break;
		}
		Debug( "create database: %s success", pcszPath);

	}while(false);
}

CSqliteDB::~CSqliteDB(void)
{
	if (m_DB)
	{
		sqlite3_close(m_DB);
		m_DB = NULL;
	}
}

bool CSqliteDB::executeSql(const char* pcszSql, ...)
{
    do
    {
		if (!m_DB)
		{
			break;
		}

		int retVal = SQLITE_OK;
		char *err, *tmp;

		va_list ap;
		va_start(ap, pcszSql);
		tmp = sqlite3_vmprintf(pcszSql, ap);
		va_end(ap);

		retVal = sqlite3_exec(m_DB, tmp, NULL, NULL, &err);
		if(retVal != SQLITE_OK)
		{
			if(err != NULL)
			{
				Debug( "Error %d: %s; caused by SQL: %s",
						retVal, err, tmp ? tmp : "<NULL>");
				sqlite3_free(err);
			}
		}
		if(tmp)
		{
			sqlite3_free(tmp);
		}

		if(retVal != SQLITE_OK)
		{
			break;
		}
		return true;
    }while(false);
    return false;
}

bool CSqliteDB::insertSqlToList(list<string>& sqlBufferList, const char* pcszSql, ...)
{
    do
    {
		if (!m_DB)
		{
			break;
		}

		char *tmp;

		va_list ap;
		va_start(ap, pcszSql);
		tmp = sqlite3_vmprintf(pcszSql, ap);
		va_end(ap);

		string sqlStr(tmp);
		sqlBufferList.push_back(sqlStr);

		if(tmp)
		{
			sqlite3_free(tmp);
		}

		return true;
    }while(false);
    return false;
}

bool CSqliteDB::executeSqlWithBlob(const char* pcszSql, const uint8* blobData, const uint32& blobDataLen)
{
	sqlite3_stmt *stmt = NULL;
	do
	{
		if (!m_DB)
		{
			break;
		}

		if(sqlite3_prepare_v2(m_DB, pcszSql, -1, &stmt, NULL) != SQLITE_OK)
		{
			Debug( "Error: Can't prepare statement: %s", sqlite3_errmsg(m_DB));
			break;
		}

		int result = sqlite3_bind_blob(stmt, 1, blobData, blobDataLen, SQLITE_STATIC);
		if(result != SQLITE_OK)
		{
			Debug( "sqlite3_bind_blob() ret: %u. sql:%s, blobDataLen:%u. error: %s", 
			result, pcszSql, blobDataLen, sqlite3_errmsg(m_DB));
			break;
		}

		if(sqlite3_step(stmt) != SQLITE_DONE)
	 	{
			Debug( "sqlite3_step() error: %s", sqlite3_errmsg(m_DB));
			break;
		}
		sqlite3_finalize(stmt);
		return true;
	}while(false);
	if(stmt)
	{
		sqlite3_finalize(stmt);
	}
	return false;
}

bool CSqliteDB::executeSqlList(list<string>& sql_list)
{
    do
    {
        if (!m_DB)
        {
            break;
        }

        if( sql_list.empty() )
        {
            break;
        }

        int retVal = SQLITE_OK;
        char *err;

        if(sqlite3_exec(m_DB, "BEGIN", NULL, NULL, &err) != SQLITE_OK)
        {
            if(err != NULL)
            {
                Debug( "Error %d: %s; caused by SQL: BEGIN", retVal, err);
                sqlite3_free(err);
                break;
            }
        }

        list<string>::iterator it = sql_list.begin();
        for(; it != sql_list.end(); ++it)
        {
            retVal = sqlite3_exec(m_DB, (*it).c_str(), NULL, NULL, &err);
            if(retVal != SQLITE_OK)
            {
                if(err != NULL)
                {
                    Debug( "Error %d: %s; caused by SQL: %s", retVal, err, (*it).c_str());
                    sqlite3_free(err);
                    break;
                }
            }
        }

        if( sqlite3_exec(m_DB, "COMMIT", NULL, NULL, &err) != SQLITE_OK )
        {
            Debug( "Error %d: %s; caused by SQL: BEGIN", retVal, err);
            sqlite3_free(err);
            break;
        }

        if(retVal != SQLITE_OK)
        {
            break;
        }
        
        return true;
    }while(false);
    return false;
}

