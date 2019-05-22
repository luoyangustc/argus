#include "LFile.h"
#include <unistd.h>
#include <string.h>
#include <stdio.h>

CLFile::CLFile()
{
	m_fd = NULL;
}

CLFile::~CLFile()
{
	CloseFile();
}
	
BOOL CLFile::OpenFile(const char * szFilename,BOOL bWrite)
{
	if (!szFilename || strlen(szFilename)==0)
		return FALSE;
		
	if (bWrite)
		m_fd = fopen(szFilename, "ab+");
	else
		m_fd = fopen(szFilename, "rb");
	
	if (!m_fd)
		return FALSE;
		
	m_sFilename = szFilename;
	
	return TRUE;		
}

BOOL CLFile::OpenFileEx(const char * szFilename,BOOL bWrite,BOOL bOverwrite)
{
	if (!szFilename || strlen(szFilename)==0)
		return FALSE;
		
	if (bWrite && bOverwrite)
	{
		DeleteFile(szFilename);
	}
	
	return OpenFile(szFilename,bWrite);
}

void CLFile::CloseFile()
{
	if (m_fd)
	{
		fclose(m_fd);
		m_fd = NULL;
	}
}

BOOL CLFile::SeekFile(ULONGLONG offset,SEEK_FLAG fromwhere)
{
	if (!m_fd)
		return FALSE;
	
	return fseek(m_fd, offset, fromwhere) == 0;	
}
	
const char * CLFile::GetCurrFilename()
{
	return m_sFilename.c_str();
}

void CLFile::GetCurrFilename(string& sFilename)
{
	sFilename = m_sFilename;
}

BOOL CLFile::AddData(BYTE *pData, DWORD dwLen)
{
	if (!m_fd)
		return FALSE;
		
	return fwrite(pData,1,dwLen,m_fd) == dwLen;
}

BOOL CLFile::PathFileExists(const char * szFilename)
{
	if (!szFilename || strlen(szFilename)==0)
		return FALSE;
		
	return access(szFilename, F_OK) == 0;
}

BOOL CLFile::DeleteFile(const char * szFilename)
{
	if (!szFilename || strlen(szFilename)==0)
		return FALSE;
		
	remove(szFilename);
	return TRUE;
}

BOOL CLFile::RenameFile(const char * szNewname,const char * szOldname)
{
	if (!szNewname || strlen(szNewname)==0 || !szOldname || strlen(szOldname)==0)
		return FALSE;
		
	rename(szOldname,szNewname);
	return TRUE;
}
	
BOOL CLFile::CreateDir(const char * szDir)
{
	if (!szDir || strlen(szDir)==0)
		return FALSE;
	
	char c = '/';

	string sPath = szDir;
	if (sPath.empty())
		return FALSE;

	if (sPath[sPath.size()-1] != c)
		sPath += c;

	for(std::string::iterator it=sPath.begin(); it!=sPath.end(); it++)
	{
		if ((*it) == c)
		{
			string tmp;
			tmp.assign(sPath.begin(), it);
			if (access(tmp.c_str(), 0) != 0)
			{
				mkdir(tmp.c_str(), 0755);
			}
		}
	}
	
	return TRUE;
}
	
string CLFile::GetModuleFileName()
{
	char c_filename[512] = { 0 };
	int rslt = readlink("/proc/self/exe", c_filename, sizeof(c_filename)); 
	char* position = strrchr(c_filename, '/');
	return position+1;
}

string CLFile::GetModulePath()
{
	char c_filename[512] = { 0 };
	int rslt = readlink("/proc/self/exe", c_filename, sizeof(c_filename)); 
	return c_filename;
}

string CLFile::GetModuleDirectory()
{
	char c_filename[512] = { 0 };
	int rslt = readlink("/proc/self/exe", c_filename, sizeof(c_filename)); 
	char* position = strrchr(c_filename, '/');
	*position = 0;

	return c_filename;
}