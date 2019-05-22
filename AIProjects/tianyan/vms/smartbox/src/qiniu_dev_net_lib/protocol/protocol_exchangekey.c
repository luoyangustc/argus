#define __PROTOCOL_EXCHANGEKEY_C__

#include "vos_bit_t.h"
#include "protocol_header.h"
#include "protocol_exchangekey.h"

int  Pack_MsgExchangeKeyRequest(char *buf, int buflen, ExchangeKeyRequest *req)
{
    //char* head = NULL; 
    char* pos = NULL;
    char* start_pos = NULL;
    do
    {
        if ( !buf || !req )
        {
            break;
        }

        start_pos = buf;
        pos = buf;

        S4Bytes(pos, req->mask);


        //0x01
        if( 0x01 & req->mask )
        {
            S1Bytes(pos, req->keyPA_01.key_P_length);
            SnBytes(pos, req->keyPA_01.key_P, req->keyPA_01.key_P_length);

            S1Bytes(pos, req->keyPA_01.key_A_length);
            SnBytes(pos, req->keyPA_01.key_A, req->keyPA_01.key_A_length);
        }

        if( 0x02 & req->mask )
        {
            S2Bytes(pos, req->except_algorithm);
            S4Bytes(pos, req->algorithm_param);
        }

        return (pos-start_pos);
        
    }while (0);
    
    return -1;
}

int Unpack_MsgExchangeKeyResponse(char *pdata, vos_uint16_t datalen, ExchangeKeyResponse *rsp)
{
    char 	*pos = pdata;
    vos_uint32_t   mask;
    //vos_uint16_t   read_len = 0;

    //FILE* file = NULL;

    do
    {

        if ( !pdata || !rsp)
        {
            return -1;
        }

		if ( datalen < sizeof(MsgHeader ) )
		{
			return -1;
		}

        {
            vos_uint16_t msg_size = 0;
            vos_uint32_t msg_id, msg_type, msg_seq;

            //__ADD_CHK(read_len, sizeof(MsgHeader), datalen);
            R2Bytes(pos, msg_size);
            R4Bytes(pos, msg_id);
            R4Bytes(pos, msg_type);
            R4Bytes(pos, msg_seq);
        }        

        {
            vos_int32_t resp_code = 0;
            //__ADD_CHK(read_len, 4, datalen);
            R4Bytes(pos, mask);

            //__ADD_CHK(read_len, 4, datalen);
            R4Bytes(pos, resp_code);
            if (resp_code != EN_SUCCESS)
            {
                break;
            }
        }
        

        if(mask & 0x01)
        {
            //__ADD_CHK(read_len, 1, datalen);
            R1Bytes(pos, rsp->keyB_01.key_B_length);
            if ( !rsp->keyB_01.key_B_length || rsp->keyB_01.key_B_length > sizeof(rsp->keyB_01.key_B) )
            {
                break;
            }

            //__ADD_CHK(read_len, rsp->keyB_01.key_B_length, datalen);
            RnBytes(pos, rsp->keyB_01.key_B, rsp->keyB_01.key_B_length);
            
            //__ADD_CHK(read_len, 2, datalen);
            R2Bytes(pos, rsp->keyB_01.key_size);
        }

        if(mask & 0x02)
        {
            //__ADD_CHK(read_len, 2, datalen);
            R2Bytes(pos, rsp->encry_algorithm);

            //__ADD_CHK(read_len, 4, datalen);
            R4Bytes(pos, rsp->algorithm_param);
        }
        
        return 0;
        
    }while (0);

    return -1;

}


