#include <string.h>
#include "ParamParser.h"

CParamParser::CParamParser(const char* szSeparate)
{
    m_strSeparate = szSeparate;
}

CParamParser::~CParamParser()
{

}

void CParamParser::SetParam(const char * szParam)
{
    //m_listString.swap(vector<string>());    //clear vector
    list<string>().swap(m_listString);
    
    size_t total_len = strlen(szParam);
    size_t seprate_len = m_strSeparate.length();

    size_t start_idx = 0;
    size_t cur_idx = 0;
    while( cur_idx < total_len )
    {
        if( strncmp(szParam+cur_idx, m_strSeparate.c_str(), seprate_len) == 0 )
        //if( m_strSeparate.compare(cur_idx, seprate_len, szParam) == 0)
        {
            size_t param_size = cur_idx - start_idx;
            string strParam(&szParam[start_idx], param_size);
            m_listString.push_back(strParam);

            cur_idx += seprate_len;
            start_idx = cur_idx;
        }
        else
        {
            cur_idx++;
        }
    }

    if(cur_idx != start_idx)
    {
        size_t param_size = cur_idx - start_idx;
        string strParam(&szParam[start_idx], param_size);
        m_listString.push_back(strParam);
    }
    
}

string CParamParser::GetParam(int id)
{
    if( id >= (int)m_listString.size() )
    {
        return string();
    }
	
	list<string>::iterator itor = m_listString.begin();
	
	while(--id)
	{
		++itor;
	}
	
    return *itor;
}

#if 0
string CParamParser::GetBehindString(int iID)
{

}

string CParamParser::GetBeforeString(int iID)
{

}

int CParamParser::GetIntParam(int iID)
{

}

__int64 CParamParser::GetInt64Param(int iID)
{

}

void CParamParser::SetParam(int iID, int iParam)
{

}

void CParamParser::SetParam(int iID,const char * szParam)
{

}
void CParamParser::SetParam(BYTE * pParam,DWORD dwLen)
{
    
}

void CParamParser::InitParamCount(int iCount)
{

}

#endif



