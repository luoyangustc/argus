#define  __CDIFFIE_HELL_MAN_C__

#include "cdiffie_hell_man.h"
#include <string.h>
#include "../inc/dnl_log.h"

static DiffieHellManDataList dif_hellman_list; 

int GetDataBlock_index(int socket_fd)
{
	int i = 0;
	do 
	{
		for (i = 0;i < MAX_SOCET_FD;i++)
		{
			if (dif_hellman_list._data[i].socket_fd == socket_fd)
			{
				return i;
			}
		}
	} while (0);

	return -1;
}

void InitDHDataList()
{
	memset(&dif_hellman_list,0,sizeof(DiffieHellManDataList));
}

int InitDiffieHellman(UINT nSize, int socket_fd)
{
	int index = 0;
	if (dif_hellman_list.connect_num < 0)
	{
		dif_hellman_list.connect_num = 0;
	}
	//index = dif_hellman_list.connect_num%MAX_SOCET_FD;
	do
	{
	   if(dif_hellman_list._data[index].connecting != 1)
	   {
	      break;
	   }
	   index++;
       if(index >= MAX_SOCET_FD)
       {
           return -1;
       }

	   DNL_TRACE_LOG("InitDiffieHellman-->nSize=%u, fd=%d, index=%d, connecting=%d, connect_num=%d\n",
	            nSize, socket_fd, index, dif_hellman_list._data[index].connecting, dif_hellman_list.connect_num);
	}while(1);

    dif_hellman_list._data[index].connecting = 1;
	dif_hellman_list._data[index].socket_fd = socket_fd;

	dif_hellman_list._data[index].size_ = nSize + 1;
    dif_hellman_list._data[index].key_size_ = nSize;
	memset(dif_hellman_list._data[index].p_,0,sizeof(dif_hellman_list._data[index].p_));
	memset(dif_hellman_list._data[index].g_,0,sizeof(dif_hellman_list._data[index].g_));
	memset(dif_hellman_list._data[index].A_,0,sizeof(dif_hellman_list._data[index].A_));
	memset(dif_hellman_list._data[index].B_,0,sizeof(dif_hellman_list._data[index].B_));
	memset(dif_hellman_list._data[index].a_,0,sizeof(dif_hellman_list._data[index].a_));
	memset(dif_hellman_list._data[index].b_,0,sizeof(dif_hellman_list._data[index].b_));
	memset(dif_hellman_list._data[index].S1_,0,sizeof(dif_hellman_list._data[index].S1_));
	memset(dif_hellman_list._data[index].S2_,0,sizeof(dif_hellman_list._data[index].S2_));
	
	InitCDHCryptLib(&dif_hellman_list._data[index].dh_crypt_data);
	BNSetEqualdw(dif_hellman_list._data[index].g_, 5, dif_hellman_list._data[index].key_size_);
	dif_hellman_list.connect_num++;
    DNL_TRACE_LOG("InitDiffieHellman-->nSize=%u, fd=%d, index=%d, connecting=%d, connect_num=%d\n",
        nSize, socket_fd, index, dif_hellman_list._data[index].connecting, dif_hellman_list.connect_num);
	return 0;
}

int MakePrime(DWORD socket_fd)
{
	int index = 0;
	index = GetDataBlock_index(socket_fd);
	if (index == -1)return -1;

	return BNMakePrime(dif_hellman_list._data[index].p_,dif_hellman_list._data[index].size_,NULL,0,&dif_hellman_list._data[index].dh_crypt_data);
}

int ComputesA(DWORD socket_fd)
{
	UINT j = 0; 
	int index = 0;
	index = GetDataBlock_index(socket_fd);
	if (index == -1)return -1;
	for (j = 0; j <dif_hellman_list._data[index].key_size_ ; j++)
		dif_hellman_list._data[index].a_[j] = MTRandom(&dif_hellman_list._data[index].dh_crypt_data);	
	return BNModExp(dif_hellman_list._data[index].A_, dif_hellman_list._data[index].g_, dif_hellman_list._data[index].a_, dif_hellman_list._data[index].p_,dif_hellman_list._data[index].key_size_);
}
int ComputesB(DWORD socket_fd)
{
	UINT j = 0;
	int index = 0;
	index = GetDataBlock_index(socket_fd);
	if (index == -1)return -1;
	for (j = 0; j <dif_hellman_list._data[index].key_size_ ; j++)
		dif_hellman_list._data[index].b_[j] = MTRandom(&dif_hellman_list._data[index].dh_crypt_data);	
	return BNModExp(dif_hellman_list._data[index].B_, dif_hellman_list._data[index].g_, dif_hellman_list._data[index].b_, dif_hellman_list._data[index].p_,dif_hellman_list._data[index].key_size_);
}
int ComputesS1(DWORD socket_fd)
{
	int index = 0;
	index = GetDataBlock_index(socket_fd);
	if (index == -1)return -1;
	return BNModExp(dif_hellman_list._data[index].S1_, dif_hellman_list._data[index].B_, dif_hellman_list._data[index].a_, dif_hellman_list._data[index].p_,dif_hellman_list._data[index].key_size_);
}
int ComputesS2(DWORD socket_fd)
{
	int index = 0;
	index = GetDataBlock_index(socket_fd);
	if (index == -1)return -1;
	return BNModExp(dif_hellman_list._data[index].S2_, dif_hellman_list._data[index].A_, dif_hellman_list._data[index].b_, dif_hellman_list._data[index].p_,dif_hellman_list._data[index].key_size_);
}

int Printf_a_p(ExchangeKeyRequest *pMsg,DWORD socket_fd)
{
	int index = 0;
//	int i = 0;
	index = GetDataBlock_index(socket_fd);
	if (index == -1)return -1;
	pMsg->keyPA_01.key_A_length = dif_hellman_list._data[index].key_size_ * 4;
	pMsg->keyPA_01.key_P_length = dif_hellman_list._data[index].size_ * 4;
	if ( dif_hellman_list._data[index].key_size_ > 16 || dif_hellman_list._data[index].size_ > 16)
	{
		return -1;
	}

	memcpy(pMsg->keyPA_01.key_P,dif_hellman_list._data[index].p_,pMsg->keyPA_01.key_P_length);
	memcpy(pMsg->keyPA_01.key_A,dif_hellman_list._data[index].A_,pMsg->keyPA_01.key_A_length);

	return 0;
}

void Set_B(ExchangeKeyResponse *pMsg,DWORD socket_fd)
{
	int index = 0;
	index = GetDataBlock_index(socket_fd);
	if(index == -1)
		return;
	if (pMsg->keyB_01.key_B_length > 63 || pMsg == NULL)
	{
		return;
	}
	memcpy(dif_hellman_list._data[index].B_,&pMsg->keyB_01.key_B[0],pMsg->keyB_01.key_B_length);
	dif_hellman_list._data[index].use_key_size_ = pMsg->keyB_01.key_size;
	return;
}

int Get_size_(DWORD socket_fd)
{
	int index = 0;
//	int i = 0;
//	int copy_len = 0;
	do 
	{
		index = GetDataBlock_index(socket_fd);
		if(index == -1) break;

		return dif_hellman_list._data[index].size_;
	} while (0);
	return -1;
}

int Get_exchangekey_(int socket_fd, char *pS_)
{
	int index = 0;
//	int i = 0;
//	int copy_len = 0;
	do 
	{
		index = GetDataBlock_index(socket_fd);
		if(index == -1) break;
		//*pS_ = &(dif_hellman_list._data[index].S1_);
		//DNL_DEBUG_LOG(" S1:%x",&(dif_hellman_list._data[index].S1_));
		memcpy(pS_, (char*)(&dif_hellman_list._data[index].S1_[0]), dif_hellman_list._data[index].use_key_size_);
		return dif_hellman_list._data[index].use_key_size_;
	} while (0);
	return -1;

}

int Printf_S1(DWORD socket_fd,DWORD *pdw_Buf,int buf_len)
{
	int index = 0;
//	int i = 0;
	int copy_len = 0;
	do 
	{
		index = GetDataBlock_index(socket_fd);
		if(index == -1 || buf_len <= 0) break;
		if (buf_len < sizeof(dif_hellman_list._data[index].S1_))
		{
			copy_len = buf_len;
		}
		else
		{
			copy_len =  sizeof(dif_hellman_list._data[index].S1_);
		}
		memcpy(pdw_Buf,&dif_hellman_list._data[index].S1_,copy_len);
/*
		for (i = 0;i<dif_hellman_list._data[index].size_;i++)
		{
			DNL_DEBUG_LOG("S1.%d:%u\n",i,dif_hellman_list._data[index].S1_[i]);
		}
		*/
		return 0;
	} while (0);
	return -1;
}

//int Set_Connect_status(DWORD socket_fd,int status)
int Clear_DH_conn_status(DWORD socket_fd)
{
		int index = 0;
		do 
		{
			index = GetDataBlock_index((int)socket_fd);
			if(index == -1) break;
			dif_hellman_list._data[index].connecting = 0;
            dif_hellman_list.connect_num--;
            DNL_TRACE_LOG("Clear_DH_conn_status-->fd=%d, index=%d, connecting=%d, connect_num=%d\n",
                socket_fd, index, dif_hellman_list._data[index].connecting, dif_hellman_list.connect_num);
			return 0;
		} while (0);
		
		return -1;
}

