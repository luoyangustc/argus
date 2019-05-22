#include <stdlib.h>
#include <boost/lexical_cast.hpp>
#include "url_helper.h"
#include "Log.h"
#include "HTTPParser.h"

http_parser_settings CHTTPParser::m_ParserSettings =
{
     CHTTPParser::message_begin_cb       //.on_message_begin
    ,CHTTPParser::request_url_cb         //.on_url
    ,CHTTPParser::header_field_cb        //.on_header_field
    ,CHTTPParser::header_value_cb        //.on_header_value
    ,CHTTPParser::headers_complete_cb    //.on_headers_complete
    ,CHTTPParser::body_cb                //.on_body
    ,CHTTPParser::message_complete_cb    //.on_message_complete
    ,CHTTPParser::response_reason_cb     //.on_reason
    ,CHTTPParser::chunk_header_cb        //.on_chunk_header
    ,CHTTPParser::chunk_complete_cb      //.on_chunk_complete
};

CHTTPParser::CHTTPParser()
{
    http_parser_init(&m_Parser, HTTP_BOTH);
    m_Parser.data = this;
}

CHTTPParser::~CHTTPParser()
{
}

void CHTTPParser::Reset()
{
    if( !m_spHeaderDetail.get() )
    {
        CHttpHeader_ptr dummy_header;
        m_spHeaderDetail.swap(dummy_header);
    }

    if( !m_spHeaderData.get() )
    {
        boost::shared_array<uint8> dummy_arry;
        m_spHeaderData.swap(dummy_arry);
    }

    if( !m_spBodyData.get() )
    {
        boost::shared_array<uint8> dummy_arry;
        m_spBodyData.swap(dummy_arry);
    }
    m_strReasonCode = "";
    m_unHeaderDataSize = 0;
    m_unBodyDataSize = 0;
    m_unRecvedBodySize = 0;
    m_strLastHeaderField = "";
    m_bParseCompleteFlag = false;
}

EN_PARSER_RESULT CHTTPParser::ParserExe(
    IN const char* raw_msg, IN size_t raw_msg_size, 
    OUT CHttpHeader_ptr& pHeaderDetail, 
    OUT boost::shared_array<uint8>& pHeaderData, OUT uint32& header_data_len,
    OUT boost::shared_array<uint8>& pContentData, OUT uint32& content_data_len)
{
    TRACE_LOG("%p,http msg parser start, msg=\n%s", &m_Parser, raw_msg);

    EN_PARSER_RESULT ret = EN_PARSER_RST_OK;
    do 
    {
        Reset();

        http_parser_init(&m_Parser, HTTP_BOTH);

        size_t nParsed = http_parser_execute( &m_Parser, &m_ParserSettings, raw_msg, raw_msg_size );
        if( nParsed != raw_msg_size )
        {
            ret = EN_PARSER_RST_FAIL;
            break;
        }

        if( !m_bParseCompleteFlag )
        {
            ret = EN_PARSER_RST_NO_COMPLETE;
            break;
        }

        if( m_unRecvedBodySize < m_unBodyDataSize )
        {
            ret = EN_PARSER_RST_FAIL;
            break;
        }

        pHeaderDetail = m_spHeaderDetail;

        pHeaderData = m_spHeaderData;
        header_data_len = m_unHeaderDataSize;

        pContentData = m_spBodyData;
        content_data_len = m_unBodyDataSize;

    } while (0);

    TRACE_LOG("%p,http msg parser end, rst=%d", &m_Parser, (int)ret);
    
    return ret;
}

EN_PARSER_RESULT CHTTPParser::URLParser(IN const char* url, IN size_t url_len, OUT string& page, OUT map<string,string>& params)
{
    struct http_parser_url u;
    if( http_parser_parse_url(url, url_len, 0, &u) != 0 ) 
    {
        return EN_PARSER_RST_FAIL;
    }

    if( u.field_set & (1<<UF_PATH) )
    {
        page.assign(url+u.field_data[UF_PATH].off, u.field_data[UF_PATH].len);
    }

    if( u.field_set & (1<<UF_QUERY) )
    {
        char* p = (char*)url+u.field_data[UF_QUERY].off;
        char* url_end_pos = p + u.field_data[UF_QUERY].len ;

        char *start = p;
        param_state state = s_param_key_start;
        string key, value;
        while(p != url_end_pos)
        {
            switch(state)
            {
            case s_param_key_start:
                if(*p == '=')
                {
                    key.assign(start, p-start);
                    state = s_param_value_start;
                    start = p+1;
                }
                else if(*p == '&')
                {
                    start = p+1;
                }
                break;
            case s_param_value_start:
                if(*p == '&')
                {
                    value.assign(start, p-start);
                    //params[key] = value;
                    params[key] = url_decode(value);
                    state = s_param_key_start;
                    start = p+1;
                }
                break;
            default:
                start = p;
                state = s_param_key_start;
                break;
            }

            ++p;    //next char
        }

        if( state == s_param_value_start)
        {
            value.assign(start, p-start);
            params[key] = url_decode(value);
        }

    }

    return EN_PARSER_RST_OK;
}

int CHTTPParser::message_begin_cb (http_parser *p)
{
    int ret = 0;
    CHTTPParser* parser = NULL;
    do 
    {
        parser = (CHTTPParser*)p->data;
        if( !parser )
        {
            ret = -1;
            break;
        }

        parser->m_spHeaderDetail = CHttpHeader_ptr(new CHttpHeader());
        if( !parser->m_spHeaderDetail.get() )
        {
            ret = -2;
            break;
        }

    } while (0);    
    
    TRACE_LOG("%p, ret=%d", parser, ret);

    return ret;
}

int CHTTPParser::request_url_cb (http_parser *p, const char *buf, size_t len)
{
    int ret = 0;
    CHTTPParser* parser = NULL;
    do 
    {
        parser = (CHTTPParser*)p->data;
        if( !parser )
        {
            ret = -1;
            break;
        }

        string org_url(buf, len);
        parser->m_spHeaderDetail->url_ = org_url;
        //parser->m_spHeaderDetail->url_.assign(buf, len);

        EN_PARSER_RESULT rst = URLParser( parser->m_spHeaderDetail->url_.c_str(), 
            parser->m_spHeaderDetail->url_.length(), 
            parser->m_spHeaderDetail->url_detail_.page_, 
            parser->m_spHeaderDetail->url_detail_.params_);
        if( rst != EN_PARSER_RST_OK )
        {
            ret = -2;
            break;
        }

    } while (0);

    TRACE_LOG("%p, url=%s, ret=%d", parser, parser->m_spHeaderDetail->url_.c_str(), ret);

    return ret;
}

int CHTTPParser::header_field_cb (http_parser *p, const char *buf, size_t len)
{
    CHTTPParser* parser = (CHTTPParser*)p->data;
    if( !parser )
    {
        ERROR_LOG("get parser failed, %p", p);
        return -1;
    }

    parser->m_strLastHeaderField.assign(buf, len);

    TRACE_LOG("%p, head_key=%s", parser, parser->m_strLastHeaderField.c_str());

    return 0;
}
int CHTTPParser::header_value_cb (http_parser *p, const char *buf, size_t len)
{
    CHTTPParser* parser = (CHTTPParser*)p->data;
    if( !parser )
    {
        ERROR_LOG("get parser failed, %p", p);
        return -1;
    }

    if( parser->m_strLastHeaderField.empty() )
    {
        ERROR_LOG("%p, get head key failed", parser);
        return -2;
    }

    string strValue(buf, len);
    parser->m_spHeaderDetail->map_key_values_[parser->m_strLastHeaderField] = strValue;

    parser->m_unHeaderDataSize += parser->m_strLastHeaderField.length() + len + 3; //3--> ":"(1byte) + "\r\n"(2byte)

    TRACE_LOG("%p, head_key_value=%s:%s", parser, parser->m_strLastHeaderField.c_str(), strValue.c_str());

    parser->m_strLastHeaderField = ""; //clear

    return 0;
}

int CHTTPParser::headers_complete_cb (http_parser *p, const char *buf, size_t len)
{
    CHTTPParser* parser = (CHTTPParser*)p->data;
    if( !parser )
    {
        ERROR_LOG("get parser failed, %p", p);
        return -1;
    }

    parser->m_spHeaderDetail->method_ = http_method_str((enum http_method)p->method);
    parser->m_spHeaderDetail->code_ = p->status_code;
    if(p->type == HTTP_REQUEST)
    {
        parser->m_spHeaderDetail->is_request = true;
    }
    else
    {
        parser->m_spHeaderDetail->is_request = false;
    }

    std::map<string,string>::iterator itor = parser->m_spHeaderDetail->map_key_values_.find("Content-Length");
    if( itor != parser->m_spHeaderDetail->map_key_values_.end() )
    {
        parser->m_unRecvedBodySize = 0;
        parser->m_unBodyDataSize = boost::lexical_cast<size_t>( itor->second );
        parser->m_spBodyData = boost::shared_array<uint8>(new uint8[parser->m_unBodyDataSize]);
        if( !parser->m_spBodyData.get() )
        {
            ERROR_LOG("%p, malloc body buffer failed, buff_size=%d", parser, parser->m_unBodyDataSize);
            return -2;
        }
    }

    parser->m_spHeaderData = boost::shared_array<uint8>(new uint8[parser->m_unHeaderDataSize+1]);
    if( !parser->m_spHeaderData )
    {
        ERROR_LOG("%p, malloc header buffer failed, buff_size=%d", parser, parser->m_unHeaderDataSize+1);
        return -3;
    }

    char* pos = (char*)parser->m_spHeaderData.get();
    itor = parser->m_spHeaderDetail->map_key_values_.begin();
    std::map<string,string>::iterator itor_end = parser->m_spHeaderDetail->map_key_values_.end();
    for( ; itor != itor_end; ++itor )
    {
        memcpy(pos, itor->first.c_str(), itor->first.size());
        pos += itor->first.size();

        *pos++ = ':';

        memcpy(pos, itor->second.c_str(), itor->second.size());
        pos += itor->second.size();

        *pos++ = '\r';
        *pos++ = '\n';
    }
    *pos++ = '\0';

    TRACE_LOG("%p, head_msg=\n%s", parser, (char*)parser->m_spHeaderData.get());

    return 0;
}

int CHTTPParser::body_cb (http_parser *p, const char *buf, size_t len)
{
    CHTTPParser* parser = (CHTTPParser*)p->data;
    if( !parser )
    {
        ERROR_LOG("get parser failed, %p", p);
        return -1;
    }

    if( (parser->m_unRecvedBodySize + len) > parser->m_unBodyDataSize )
    {
        ERROR_LOG("%p, body incorret, body_size=%u, recved_size=%u, len=%u", 
            parser, parser->m_unBodyDataSize, parser->m_unRecvedBodySize, len);
        return -2;
    }

    memcpy( parser->m_spBodyData.get()+parser->m_unRecvedBodySize, buf, len);
    parser->m_unRecvedBodySize += len;
    
    TRACE_LOG("%p, body_size=%u, recved_size=%u", parser, parser->m_unBodyDataSize, parser->m_unRecvedBodySize);

    return 0;
}
int CHTTPParser::count_body_cb (http_parser *p, const char *buf, size_t len)
{
    CHTTPParser* parser = (CHTTPParser*)p->data;
    if( !parser )
    {
        ERROR_LOG("get parser failed, %p", p);
        return -1;
    }

    TRACE_LOG("%p", parser);

    return 0;
}

int CHTTPParser::message_complete_cb (http_parser *p)
{
    CHTTPParser* parser = (CHTTPParser*)p->data;
    if( !parser )
    {
        ERROR_LOG("get parser failed, %p", p);
        return -1;
    }

    parser->m_bParseCompleteFlag = true;

    if( parser->m_unRecvedBodySize < parser->m_unBodyDataSize )
    {
        TRACE_LOG("%p, recv body failed, body_size=%u, recved_size=%u", parser, parser->m_unBodyDataSize, parser->m_unRecvedBodySize);
        return -2;
    }

    TRACE_LOG("%p, header_size=%u,body_size=%u", parser, parser->m_unHeaderDataSize, parser->m_unRecvedBodySize);

    return 0;
}

int CHTTPParser::response_reason_cb (http_parser *p, const char *buf, size_t len)
{
    CHTTPParser* parser = (CHTTPParser*)p->data;
    if( !parser )
    {
        ERROR_LOG("get parser failed, %p", p);
        return -1;
    }

    parser->m_strReasonCode.assign(buf, len);
    
    TRACE_LOG("%p, reason_code=%s", parser, parser->m_strReasonCode.c_str());

    return 0;
}

int CHTTPParser::chunk_header_cb (http_parser *p)
{
    CHTTPParser* parser = (CHTTPParser*)p->data;
    if( !parser )
    {
        return -1;
    }

    return 0;
}

int CHTTPParser::chunk_complete_cb (http_parser *p)
{
    CHTTPParser* parser = (CHTTPParser*)p->data;
    if( !parser )
    {
        return -1;
    }

    return 0;
}
