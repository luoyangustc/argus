#include <stdlib.h>
#include "HTTPResponse.h"
#include <boost/lexical_cast.hpp>
#include "url_helper.h"

CHttpResponse::CHttpResponse(): m_bIsGood(false)
{
}

CHttpResponse::~CHttpResponse()
{
}

bool CHttpResponse::Set(SHttpResponsePara_ptr& resp)
{
    Reset();

    if( !resp.get() )
    {
        m_bIsGood = false;
        return false;
    }

    m_strStatus = "HTTP/1.1 " + resp->ret_code;

    SHeader header;
    header.name = "Server";
    header.value = "AnYan";
    m_Headers.push_back(header);

    if( !resp->location.empty() )
    {
        header.name = "Location";
        header.value = url_encode(resp->location);
        m_Headers.push_back(header);
    }

    if( !resp->content_type.empty() )
    {
        header.name = "Content-Type";
        header.value = resp->content_type;
        m_Headers.push_back(header);
    }

    header.name = "Content-Length";
    header.value = boost::lexical_cast<string>(resp->content_len);
    m_Headers.push_back(header);

    header.name = "Connection";
    if( resp->keep_alive )
    {
        header.value = "keep-alive";
    }
    else
    {
        header.value = "close";
    }
    m_Headers.push_back(header);

    header.name = "Cache-Control";
    header.value = "no-cache, must-revalidate";
    m_Headers.push_back(header);

    header.name = "Accept-Ranges";
    header.value = "bytes";
    m_Headers.push_back(header);

    m_spContent = resp->pContent;
    m_unContentLen = resp->content_len;

    m_bIsGood = true;

    return true;
}

SDataBuff CHttpResponse::ToBuffers()
{
    SDataBuff buffer(10*1024+m_unContentLen);
    do 
    {
        if ( !m_bIsGood )
        {
            break;
        }
        buffer.push_back(m_strStatus.c_str(), m_strStatus.length());
        buffer.push_back(CRLF, 2);
        for (std::size_t i = 0; i < m_Headers.size(); ++i)
        {
            SHeader& h = m_Headers[i];
            buffer.push_back(h.name.c_str(), h.name.length());
            buffer.push_back(NAME_VAL_SEPARATOR, 2);
            buffer.push_back(h.value.c_str(), h.value.length());
            buffer.push_back(CRLF, 2);
        }
        buffer.push_back(CRLF, 2);
        if( m_unContentLen )
        {
            buffer.push_back(m_spContent.get(), m_unContentLen);
        }

    } while (0);
    
    return buffer;
}

void CHttpResponse::Reset()
{
    if( !m_spContent.get() )
    {
        boost::shared_array<uint8> dummy_arry;
        m_spContent.swap(dummy_arry);
    }

    m_Headers.clear();
    m_bIsGood = false;
    m_strStatus.clear();
    m_unContentLen = 0;

}
