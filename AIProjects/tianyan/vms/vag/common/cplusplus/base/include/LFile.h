#ifndef __LFILE__H__
#define __LFILE__H__

#pragma once

#include "typedef_win.h"

#ifdef _WINDOWS
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#endif


#include <list>
#include <map>
#include <set>
#include <vector>
#include <queue>
#include <string>
using namespace std;

enum SEEK_FLAG {SEEK_SET_FLAG=0,SEEK_CUR_FLAG,SEEK_END_FLAG};

class CLFile  
{
public:				
	CLFile();
	~CLFile();
	
	BOOL OpenFile(const char * szFilename,BOOL bWrite = FALSE);
	BOOL OpenFileEx(const char * szFilename,BOOL bWrite = FALSE,BOOL bOverwrite = FALSE);
	void CloseFile();
	bool IsOpen(){return m_fd!=NULL;}
	BOOL SeekFile(ULONGLONG offset,SEEK_FLAG fromwhere);
	const char * GetCurrFilename();
	void GetCurrFilename(string& sFilename);
	BOOL AddData(BYTE *pData, DWORD dwLen);
public:
	static BOOL PathFileExists(const char * szFilename);	
	static BOOL DeleteFile(const char * szFilename);

	static BOOL RenameFile(const char * szNewname,const char * szOldname);
	
	//递归创建一个指定目录路径的所有目录
	static BOOL CreateDir(const char * szDir);
	
	//		test.exe
	static string GetModuleFileName(void);
	
	//    /home/abc/test.exe
	static string GetModulePath(void);
	
	//    /home/abc
	static string GetModuleDirectory(void);
private:
	string		m_sFilename;
	FILE* m_fd;
};

#endif //__LFILE__H__
