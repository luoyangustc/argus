#ifndef __AY_MSG_READER_H__
#define __AY_MSG_READER_H__
#include <queue>
#include "Common.h"
#include "BufferInfo.h"
#include "typedefine.h"

enum en_reader_seq
{
    en_reader_seq_byte01,
    en_reader_seq_byte02,
    en_reader_seq_data
};

class CAYMsgReader
{
public:
    CAYMsgReader();
    virtual ~CAYMsgReader();
public:
    int Push(const void* data, size_t data_len);
    int PopMsg(SDataBuff& msg);
    int PopMsgQueue(std::queue<SDataBuff>& msg_que);
    int GetMsgQueueSize();
    void Reset();
private:
    en_reader_seq m_eReaderSeq;
    uint8 m_nByte01;
    SDataBuff m_ReadingMsg;
    std::queue<SDataBuff> m_MsgQueue;
    //boost::recursive_mutex m_MsgQueueLock;
};

#endif //__TCPCLIENTSOCKET_H__

