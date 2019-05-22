#ifndef __SQLITE_DB_CXXWRAPE__
#define __SQLITE_DB_CXXWRAPE__

#pragma once

#include <string>
#include <list>
using namespace std;

#include "sqlite3.h"
#include "typedef_win.h"
#include "typedefine.h"
#include "logging_posix.h"
#include <boost/shared_ptr.hpp>

class CSqliteDB
{
public:
	CSqliteDB(const char* pcszPath);
	~CSqliteDB(void);
	bool executeSql(const char* pcszSql, ...);
	bool insertSqlToList(list<string>& sqlBufferList, const char* pcszSql, ...);
	bool executeSqlWithBlob(const char* pcszSql, const uint8* blobData, const uint32& blobDataLen); 
	bool executeSqlList(list<string>& sql_list);
private:
	sqlite3 *m_DB; 
};

typedef boost::shared_ptr<CSqliteDB> CSqliteDB_ptr;

#endif
