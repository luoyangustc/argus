#include <stdlib.h>
#include <string.h>
#include <sstream>
#ifdef _WINDOWS
#include <Windows.h>
#else
#include <unistd.h>
#endif
#include "ConfigHelper.h"

CConfigHelper::CConfigHelper()
{

}

CConfigHelper::~CConfigHelper()
{

}

int CConfigHelper::read_config_file(const string& filename)
{
    string fn;
    if (filename.size() == 0)
        fn = CConfigHelper::get_default_config_filename();
    else
        fn = filename;
	
	m_strFilePath = fn;
	ifstream is(fn.c_str());
	if(is.fail())
	{
		return EN_CFG_ERR_OPENFILE_FAIL;
	}
	
	string strLine, strSection, strLeftValue, strRightValue;
	string::size_type pos;
	
	while( !is.eof() )
	{
		(void)std::getline(is, strLine);
		
		//去注释
		pos = strLine.find("#", 0);
		if(pos != string::npos)
		{
			strLine.erase(pos, strLine.length()-pos);
		}
		
		//去两端空格
        pos = strLine.find_first_not_of("\t ");
        if((pos != string::npos)&&pos)
        {
            strLine.erase(0, pos );
        }
		
        pos = strLine.find_last_not_of("\t\r\n ");
        if((pos != string::npos)&&(pos!=(strLine.length()-1)))
        {
            strLine.erase(pos+1, strLine.length()-pos-1 );
        }

        //空字符串检查
        if(strLine.empty())
        {
            continue;
        }
		
		//解析Section
		if( (strLine[0] == '[') &&
			(strLine[strLine.length() -1] == ']'))
		{
			strSection = strLine.substr(1, strLine.length() -2);
			continue;
		}
		
		//解析Item
		pos = strLine.find("=", 0);
		if(pos != string::npos)
		{
			strLeftValue = strLine.substr(0, pos);
			strRightValue = strLine.substr(pos+1, strLine.length() - pos - 1);
			
			m_SectionMap[strSection][strLeftValue] = strRightValue;
		}
	}
	
	is.close();
	
	return EN_CFG_SUCCESS;
}

int CConfigHelper::save_config_file(const string& filename)
{
	string strFileNameBak = m_strFilePath + ".bak";
	
	//先删除之前的备份文件
	(void)remove(strFileNameBak.c_str());
	
	//重命名
	(void)rename(m_strFilePath.c_str(), strFileNameBak.c_str());
	
	//创建输出文件流
	m_strFilePath = filename;
	ofstream os(m_strFilePath.c_str());
	if(os.fail())
	{
		return EN_CFG_ERR_OPENFILE_FAIL;
	}
	
	//创建输入文件流
	ifstream is(strFileNameBak.c_str());
	if(is.fail())
	{
		return EN_CFG_ERR_OPENFILE_FAIL;
	}
	
	string strLine, strSection, strKey, strValue;
	string::size_type pos;
	
	while( !is.eof() )
	{
		std::getline(is, strLine);
		
		//复制注释
		pos = strLine.find("#", 0);
		if(pos != string::npos)
		{
			os << strLine << endl;
			continue;
		}
		
		//去两端空格
		strLine.erase( strLine.find_first_not_of("\t ") );
		strLine.erase( strLine.find_last_not_of("\t\r\n ") + 1);
		
		if(strLine.empty())
		{
			//如果文件已到尾，不再输出新行
			if( !is.eof() )
			{
				os << strLine << endl;
			}
			continue;
		}
		
		//解析Section， 并复制
		if( (strLine[0] == '[') &&
			(strLine[strLine.length() -1] == ']'))
		{
			strSection = strLine.substr(1, strLine.length() -2);
			os << strLine << endl;
			continue;
		}
		
		//解析Item， 并复制
		pos = strLine.find("=", 0);
		if(pos != string::npos)
		{
			strKey = strLine.substr(0, pos);
			strValue = strLine.substr(pos+1, strLine.length() - pos - 1);
			
			//查找map
			string strNewValue = "";
			int nResult = get_value(strNewValue, strSection, strKey);
			if( nResult == EN_CFG_SUCCESS )
			{
				strValue = strNewValue;
			}
			os << strKey << "=" << strValue << endl;
		}
	}
	
	//关闭输入文件流
	is.close();
	
	//删除备份文件
	(void)remove(strFileNameBak.c_str());
	
	return EN_CFG_SUCCESS;
}

void CConfigHelper::set_value(const string& section, const string& key, const string& value)
{
	m_SectionMap[section][key] = value;
}

void CConfigHelper::set_value(const string& section, const string& key, unsigned int value)
{	
    std::ostringstream oss;
    oss << value;
	m_SectionMap[section][key] = oss.str();
}


int CConfigHelper::get_section(const string& section, ITEM_MAP &keyValueMap)
{
	SECTION_MAP::iterator itorSection = m_SectionMap.find(section);
	if( itorSection == m_SectionMap.end() )
	{
		return EN_CFG_ERR_NOSECTION;
	}
	
	ITEM_MAP itemMap = itorSection->second;
	ITEM_MAP::iterator itorItem = itemMap.begin();
	
	while( itorItem != itemMap.end() )
	{
		keyValueMap[itorItem->first] = itorItem->second;
		++itorItem;
	}
	
	return EN_CFG_SUCCESS;
}


int CConfigHelper::get_value(string& value, const string& section, const string& key, const string& default_value )
{
	if(m_SectionMap.find(section) == m_SectionMap.end())
	{
        value = default_value;
		return EN_CFG_ERR_NOSECTION;
	}
	
	if(m_SectionMap[section].find(key) == m_SectionMap[section].end())
	{
        value = default_value;
		return EN_CFG_ERR_NOITEM;
	}
	
	value = m_SectionMap[section][key];
	
	return EN_CFG_SUCCESS;
}

int CConfigHelper::get_value(bool& value, const string& section, const string& key, bool default_value)
{
    string strValue;
    int nResult = get_value(strValue, section, key);
    if( EN_CFG_SUCCESS == nResult )
    {
        unsigned int temp = strtoul(strValue.c_str(), NULL, 0);
        if (0 != temp)
        {
            value = true;
        }
        else
        {
            value = false;
        }
    }
    else
    {
        value = default_value;
    }
	return nResult;
}

int CConfigHelper::get_value(unsigned int& value, const string& section, const string& key, unsigned int default_value)
{
	string strValue;
	int nResult = get_value(strValue, section, key);
	if( EN_CFG_SUCCESS == nResult )
	{
		value = strtoul(strValue.c_str(), NULL, 0);
	}
	else
	{
		value = default_value;
	}
	
	return nResult;
}

string CConfigHelper::get_module_short_name()
{
    char* pos;
    char c_filename[512];
    memset(c_filename, 0x0, sizeof(c_filename));

#ifdef _WINDOWS
    ::GetModuleFileName(NULL, c_filename, sizeof(c_filename));
    pos = strrchr(c_filename, '\\');
#else
    (void)readlink("/proc/self/exe", c_filename, sizeof(c_filename));
    pos = strrchr(c_filename, '/');
#endif
    if(pos)
    {
        return pos;
    }

    return "";
}

string CConfigHelper::get_module_full_name()
{
    char c_filename[512];
    memset(c_filename, 0x0, sizeof(c_filename));

#ifdef _WINDOWS
    ::GetModuleFileName(NULL, c_filename, sizeof(c_filename));
#else
    (void)readlink("/proc/self/exe", c_filename, sizeof(c_filename)); 
#endif
    return c_filename;
}

string CConfigHelper::get_default_config_filename()
{
    string file_name = get_module_full_name();
    if( file_name.length() > 0 )
    {
        return file_name + ".conf";
    }
    return "";
}

string CConfigHelper::get_default_config_dir()
{
    char c_filename[512];
    memset(c_filename, 0x0, sizeof(c_filename));

#ifdef _WINDOWS
    ::GetModuleFileName(NULL, c_filename, sizeof(c_filename));
    char* pos = strrchr(c_filename, '\\');
    if(pos) {
        *pos-- = 0;
        if(*pos == '\\')
        {
            *pos = 0;
        }
    }
#else
    (void)readlink("/proc/self/exe", c_filename, sizeof(c_filename));
    char* pos = strrchr(c_filename, '/');
    if(pos) *pos = 0;
#endif
    return c_filename;
}

unsigned int GetPrivateProfileInt(const char* szAppName, const char* szKeyName, int nDefault, const char* szFileName)
{
    unsigned int nValue = nDefault;
    string fn;
    if (!szFileName || strlen(szFileName)==0)
    	fn = CConfigHelper::get_default_config_filename();
    else
    	fn = szFileName;
    
    do
    {
        CConfigHelper cfg;
        if( cfg.read_config_file( fn ) != EN_CFG_SUCCESS)
        {
            break;
        }
        
        (void)cfg.get_value(nValue, szAppName, szKeyName, nDefault);
        
    }while(0);
    
    return nValue;
}

unsigned int GetPrivateProfileString(const char* szAppName, const char* szKeyName, const char* szDefault, char* szReturnedString, unsigned int nSize, const char* szFileName)
{
    int ret = -1;
    string fn;
    if (!szFileName || strlen(szFileName)==0)
    	fn = CConfigHelper::get_default_config_filename();
    else
    	fn = szFileName;
    	
    do
    {
        CConfigHelper cfg;
        if( cfg.read_config_file( fn ) != EN_CFG_SUCCESS)
        {
            break;
        }
        
        string strValue;
        (void)cfg.get_value(strValue, szAppName, szKeyName, szDefault);
        
        size_t len = strValue.length();
        if(len>nSize-1)
        {
            break;
        }
        
        memcpy(szReturnedString, strValue.c_str(), len);
        szReturnedString[len] = '\0';
        
        ret = 0;
        
    }while(0);
    
    return ret;
}

bool WritePrivateProfileString(const char* szAppName, const char* szKeyName, const char* szString, const char* szFileName)
{
    bool ret = false;
    string fn;
    if (!szFileName || strlen(szFileName)==0)
    	fn = CConfigHelper::get_default_config_filename();
    else
    	fn = szFileName;
    	
    do
    {
        CConfigHelper cfg;
        if( cfg.read_config_file( fn ) != EN_CFG_SUCCESS)
        {
            break;
        }
        
        cfg.set_value(szAppName, szKeyName, szString);
        
        ret = true;
        
    }while(0);
    
    return ret;
}