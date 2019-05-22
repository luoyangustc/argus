#include "AYMsgReader.h"
#include "bit_t.h"
#include "Log.h"
#include "protocol_header.h"

using namespace protocol;

#define  MIN_AY_MSG_LEN (sizeof(protocol::MSG_HEADER))
#define  MAX_AY_MSG_LEN (16*1024+4*1024)

CAYMsgReader::CAYMsgReader()
    :m_ReadingMsg(MAX_AY_MSG_LEN)
    ,m_eReaderSeq(en_reader_seq_byte01)
    ,m_nByte01(0)
{
}

CAYMsgReader::~CAYMsgReader()
{
}

void CAYMsgReader::Reset()
{
    m_eReaderSeq = en_reader_seq_byte01;
    m_nByte01 = 0;
    m_ReadingMsg.clear();
    
    //boost::lock_guard<boost::recursive_mutex> lock(m_MsgQueueLock);
    while(!m_MsgQueue.empty())
    {
        m_MsgQueue.pop();
    }
}

int CAYMsgReader::Push(const void* data, size_t data_len)
{
    if (!data||!data_len)
    {
        return -1;
    }

    uint8* cur_pos = (uint8*)data;
    uint8* end_pos = cur_pos + data_len;

    while( cur_pos < end_pos )
    {
        switch(m_eReaderSeq)
        {
        case en_reader_seq_byte01:
            {
                if(end_pos-cur_pos>=2)
                {
                    uint16 msg_size = *(uint16*)cur_pos;
                    if(msg_size<MIN_AY_MSG_LEN || msg_size>MAX_AY_MSG_LEN)
                    {
                        return -1;
                    }
                    if( !m_ReadingMsg.resize(msg_size) )
                    {
                        return -2;
                    }

                    m_ReadingMsg.push_back(cur_pos, 2);
                    cur_pos += 2;

                    m_eReaderSeq = en_reader_seq_data;
                }
                else if(end_pos-cur_pos==1)
                {
                    m_nByte01 = *cur_pos++;
                    m_eReaderSeq = en_reader_seq_byte02;
                }
            }
            break;
        case en_reader_seq_byte02:
            {
                uint8 tmp[2];
                tmp[0] = m_nByte01;
                tmp[1] = *cur_pos++;

                uint16 msg_size = *(uint16*)tmp;
                if(msg_size<MIN_AY_MSG_LEN || msg_size>MAX_AY_MSG_LEN)
                {
                    return -1;
                }
                if( !m_ReadingMsg.resize(msg_size) )
                {
                    return -2;
                }

                m_ReadingMsg.push_back(tmp, 2);

                m_eReaderSeq = en_reader_seq_data;
            }
            break;
        case en_reader_seq_data:
            {
                uint16 msg_size = *(uint16*)m_ReadingMsg.get_buffer();
                uint32 unread_msg_size = msg_size - m_ReadingMsg.data_size();
                if( unread_msg_size <= end_pos-cur_pos )
                {
                    m_ReadingMsg.push_back(cur_pos, unread_msg_size);
                    cur_pos += unread_msg_size;
                    {
                        //boost::lock_guard<boost::recursive_mutex> lock(m_MsgQueueLock);
                        m_MsgQueue.push(m_ReadingMsg);
                    }
                    m_eReaderSeq = en_reader_seq_byte01;
                }
                else
                {
                    m_ReadingMsg.push_back(cur_pos, end_pos-cur_pos);
                    cur_pos += end_pos-cur_pos;
                }
            }
            break;
        default:
            {
                return -2;
            }
            break;
        }
    }

    return 0;
}

int CAYMsgReader::PopMsg(SDataBuff& msg)
{
    //boost::lock_guard<boost::recursive_mutex> lock(m_MsgQueueLock);
    if(m_MsgQueue.empty())
    {
        return 0;
    }

    msg = m_MsgQueue.front();
    m_MsgQueue.pop();

    return 1;
}

int CAYMsgReader::PopMsgQueue(std::queue<SDataBuff>& msg_que)
{
    //boost::lock_guard<boost::recursive_mutex> lock(m_MsgQueueLock);
    int ret = m_MsgQueue.size();
    while(!m_MsgQueue.empty())
    {
        msg_que.push(m_MsgQueue.front());
        m_MsgQueue.pop();
    }
    return ret;
}

int CAYMsgReader::GetMsgQueueSize()
{
    //boost::lock_guard<boost::recursive_mutex> lock(m_MsgQueueLock);
    return m_MsgQueue.size();
}
