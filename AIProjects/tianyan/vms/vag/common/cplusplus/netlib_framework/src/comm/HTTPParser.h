#ifndef __HTTP_PARSER_H__
#define __HTTP_PARSER_H__

#pragma once

#include <map>
#include <string>
using namespace std;

#include "typedef_win.h"
#include "typedefine.h"
#include <map>
#include <boost/shared_ptr.hpp>
#include "http_parser.h"

using namespace proxygen;

enum EN_PARSER_RESULT
{
    EN_PARSER_RST_NONE = 0,
    EN_PARSER_RST_OK,
    EN_PARSER_RST_FAIL,
    EN_PARSER_RST_NO_COMPLETE,
    EN_PARSER_RST_MAX
};

class CHTTPParser
{
public:
    enum param_state
    {
        s_param_key_start = 0,
        s_param_key_end,
        s_param_value_start,
        s_param_value_end
    };
public:
    CHTTPParser();
    ~CHTTPParser();
    void Reset();
    EN_PARSER_RESULT ParserExe(
        IN const char* raw_msg, IN size_t raw_msg_size, 
        OUT CHttpHeader_ptr& pHeaderDetail, 
        OUT boost::shared_array<uint8>& pHeaderData, OUT uint32& header_data_len,
        OUT boost::shared_array<uint8>& pContentData, OUT uint32& content_data_len
        );
    static EN_PARSER_RESULT URLParser(IN const char* url, IN size_t url_len, OUT string& page, OUT map<string,string>& params);
private:
    static int message_begin_cb (http_parser *p);
    static int request_url_cb (http_parser *p, const char *buf, size_t len);
    static int header_field_cb (http_parser *p, const char *buf, size_t len);
    static int header_value_cb (http_parser *p, const char *buf, size_t len);
    static int headers_complete_cb (http_parser *p, const char *buf, size_t len);
    static int body_cb (http_parser *p, const char *buf, size_t len);
    static int message_complete_cb (http_parser *p);
    static int count_body_cb (http_parser *p, const char *buf, size_t len);
    static int response_reason_cb (http_parser *p, const char *buf, size_t len);
    static int chunk_header_cb (http_parser *p);
    static int chunk_complete_cb (http_parser *p);
private:
    static http_parser_settings m_ParserSettings;
    http_parser                 m_Parser;

    CHttpHeader_ptr             m_spHeaderDetail;
    string                      m_strReasonCode;

    boost::shared_array<uint8>  m_spHeaderData;
    size_t                      m_unHeaderDataSize;

    boost::shared_array<uint8>  m_spBodyData;
    size_t                      m_unBodyDataSize;

    size_t                      m_unRecvedBodySize;
    string                      m_strLastHeaderField;
    bool                        m_bParseCompleteFlag;
};

typedef boost::shared_ptr<CHTTPParser> CHTTPParser_ptr;

#endif //__HTTP_PARSER_H__

