#ifndef __HTTP_RESPONSE_H__
#define __HTTP_RESPONSE_H__

#include <string>
#include <vector>
#include "Common.h"
#include "typedef_win.h"
#include "typedefine.h"
#include "BufferInfo.h"

using namespace std;

const char CRLF[2] = {'\r', '\n'};
const char NAME_VAL_SEPARATOR[2] = { ':', ' ' };

class CHttpResponse
{
public:
    struct SHeader
    {
        std::string name;
        std::string value;
    };
public:
    CHttpResponse();
    ~CHttpResponse();

    bool IsGood(){return m_bIsGood;}
    bool Set(SHttpResponsePara_ptr& resp);
    SDataBuff ToBuffers();
    void Reset();

private:
    bool                        m_bIsGood;
    string                      m_strStatus;
    std::vector<SHeader>        m_Headers;
    boost::shared_array<uint8>  m_spContent;
    uint32                      m_unContentLen;
};

#endif //__HTTP_RESPONSE_H__

