
#include "typedef_win.h"
#include "typedefine.h"
#include "AYCrypt.h"

CAYCrypt::CAYCrypt()
{
}

CAYCrypt::~CAYCrypt() 
{
}

/*
 * C=A^B
 * A=B^C
 */
int CAYCrypt::EncryUdpMsg(char * buff, int buff_len)
{
    int i = 0;
    int start_key_pos = 0;
    unsigned char key = 0;
    unsigned char algo = 1;
    do
    {
        //generate rand key
        //srand((int)time(NULL));
        key = rand()%256;
        start_key_pos = 2+4+8-1;

        if(buff_len < start_key_pos)
        {
            break;
        }

        //保存在flag的最高字节
        buff[start_key_pos] = key;

        if(buff_len < start_key_pos)
        {
            break;
        }

        key = key|0x01;

        for(i = 0; i<buff_len; ++i)
        {
            if( i != 1 && i != start_key_pos)
            {
                buff[i] = buff[i]^key;
                key = buff[i];
            }
        }

        //加密成功后再置算法位
        algo <<= 3;
        buff[1] |= algo;

        return 0;
    } while (0);
    return -1;
}

int CAYCrypt::DecryUdpMsg(char * buff, int buff_len)
{
    unsigned char testkey;
    unsigned char get_algo;
    int device_id_len_pos = 0;
    int start_key_pos = 0;
    unsigned char key;
    int i = 0;
    do
    {
        get_algo = buff[1];
        get_algo >>= 3;

        if (get_algo == 1)
        {
            //设备ID长度所在的位置
            start_key_pos = 2+4+8-1;

            if(buff_len < start_key_pos)
            {
                break;
            }

            //选定设备ID的最后一个字节为KEY
            key = buff[start_key_pos];
            key = key|0x01;

            for( i = 0; i<buff_len; ++i)
            {
                if( i != 1 && i != start_key_pos)
                {
                    testkey = buff[i];
                    buff[i] = buff[i]^key;
                    key = testkey;
                }
            }

            buff[1] &= 0x07;//((~get_algo)|0x07);//0x7F;
            buff[start_key_pos] &=0x0;
        }
        return 0;
    } while (0);
    return -1;
}

int CAYCrypt::EncryTcpMsg(unsigned char* buf, int buf_len, int key_pos, unsigned char* pkey, int key_len)
{
    int i = 0;
    int start_key_pos = 0;
    int encry_len = 0;
    unsigned char key = 0;
    unsigned char encry_key[64] = {0};
    do 
    {
        {
            if(buf == NULL || pkey == NULL)
            {
                break;
            }

            start_key_pos = key_pos;
            if(buf_len < start_key_pos)
            {
                break;
            }

            if(buf_len > 128)
                encry_len = 128;
            else
                encry_len = buf_len;

            memcpy(&encry_key[0],pkey,key_len);

            //产生随机key
            key = rand()%256;

            //保存key
            buf[start_key_pos] = key;

            if(buf_len < start_key_pos)
            {
                break;
            }

            key = key|0x01;

            for(i = 0; i<key_len; i++)
            {
                encry_key[i] = encry_key[i]^key;
                key = encry_key[i];
            }

            for(i = 2; i < encry_len; ++i)
            {
                if(i != start_key_pos)
                {
                    buf[i] = buf[i]^encry_key[i%key_len];
                    //key = buf[i];
                }
            }

            //加密成功后再置算法位
            //algo <<= 3;
            //f[1] |= algo;
        }

        return 0;
    } while (0);
    return -1;
}

int CAYCrypt::DecryTcpMsg(unsigned char * buf, int buf_len, int key_pos, unsigned char * pkey, int key_len)
{
    //	unsigned char testkey;
    int device_id_len_pos = 0;
    int start_key_pos = 0;
    unsigned char key;
    int encry_len = 0;
    unsigned char encry_key[64] = {0};
    int i = 0;
    do 
    {

        if (buf == NULL || pkey == NULL)
        {
            break;
        }

        //key所在的位置
        start_key_pos = key_pos;
        if(buf_len < start_key_pos)
        {
            break;
        }

        memcpy(&encry_key[0],pkey,key_len);

        if (buf_len > 128)
        {
            encry_len = 128;
        }
        else
        {
            encry_len = buf_len;
        }

        //选定设备ID的最后一个字节为KEY
        key = buf[start_key_pos];
        key = key|0x01;

        for(i = 0; i<key_len; i++)
        {
            encry_key[i] = encry_key[i]^key;
            key = encry_key[i];
        }

        for(i = 2; i < encry_len; ++i)
        {
            if(i != start_key_pos)
            {
                buf[i] = buf[i]^encry_key[i%key_len];
                //key = buf[i];
            }	 
        }

        buf[start_key_pos] &=0x0;
        return 0;
    } while (0);
    return -1;
}
