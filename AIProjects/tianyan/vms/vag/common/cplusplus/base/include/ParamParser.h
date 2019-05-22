#ifndef __PARAMPARSER_H__
#define __PARAMPARSER_H__

#include "typedef_win.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <list>
#include <map>
#include <set>
#include <vector>
#include <queue>
#include <string>
using namespace std;

class CParamParser
{
public:
	CParamParser(const char * szSeparate = NULL);
	~CParamParser();
public:
	string GetBehindString(int iID);
	string GetBeforeString(int iID);
	int GetIntParam(int iID);
	__int64 GetInt64Param(int iID);
	string GetParam(int iID);
	void SetParam(int iID, int iParam);
	void SetParam(int iID,const char * szParam);
	void SetParam(const char * szParam);
	void SetParam(BYTE * pParam,DWORD dwLen);
	void InitParamCount(int iCount);
public:
	string	m_strSeparate;
	list<string> m_listString;
};

#endif // __PARAMPARSER_H__

